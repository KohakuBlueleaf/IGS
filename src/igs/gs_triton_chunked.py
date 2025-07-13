import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function
from igs.gs_triton import gaussian_splatting_fused_backward_kernel


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


@triton.jit
def gaussian_splatting_fused_forward_kernel_chunked(
    # Pointers to tensors
    x_ptr,
    y_ptr,
    position_ptr,
    cov_inv_00_ptr,
    cov_inv_01_ptr,
    cov_inv_11_ptr,
    alphas_ptr,
    colors_ptr,
    result_ptr,
    result_sum_ptr,
    # Dimensions
    B,
    H,
    W,
    N,
    C,
    C_padded,
    N_padded,
    N_CHUNK_SIZE: tl.constexpr,
    # Strides
    stride_x_b,
    stride_x_h,
    stride_x_w,
    stride_y_b,
    stride_y_h,
    stride_y_w,
    stride_pos_b,
    stride_pos_n,
    stride_pos_d,
    stride_cov_b,
    stride_cov_n,
    stride_alpha_b,
    stride_alpha_n,
    stride_colors_b,
    stride_colors_n,
    stride_colors_c,
    stride_result_b,
    stride_result_c,
    stride_result_h,
    stride_result_w,
    stride_result_sum_b,
    stride_result_sum_h,
    stride_result_sum_w,
    # Block size for the inner loop
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get the program IDs for batch and chunk
    b = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    # Calculate the range of Gaussians for this chunk
    gaussian_start = chunk_idx * N_CHUNK_SIZE
    gaussian_offsets = gaussian_start + tl.arange(0, N_CHUNK_SIZE)
    gaussian_mask = gaussian_offsets < N  # Handle padding for N

    # Load Gaussian parameters for the entire chunk
    pos_ptr_base = position_ptr + b * stride_pos_b + gaussian_offsets * stride_pos_n
    pos_x = tl.load(
        pos_ptr_base + 0 * stride_pos_d, mask=gaussian_mask, other=0.0
    )  # [N_CHUNK_SIZE]
    pos_y = tl.load(
        pos_ptr_base + 1 * stride_pos_d, mask=gaussian_mask, other=0.0
    )  # [N_CHUNK_SIZE]

    cov_inv_00 = tl.load(
        cov_inv_00_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    cov_inv_01 = tl.load(
        cov_inv_01_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    cov_inv_11 = tl.load(
        cov_inv_11_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    alpha = tl.load(
        alphas_ptr + b * stride_alpha_b + gaussian_offsets * stride_alpha_n,
        mask=gaussian_mask,
        other=0.0,
    )

    # Load colors for the current chunk [N_CHUNK_SIZE, C_padded]
    color_offsets = tl.arange(0, BLOCK_SIZE_C)
    color_mask = color_offsets < C
    colors_ptr_base = (
        colors_ptr + b * stride_colors_b + gaussian_offsets[:, None] * stride_colors_n
    )
    colors_chunk = tl.load(
        colors_ptr_base + color_offsets[None, :] * stride_colors_c,
        mask=gaussian_mask[:, None] & color_mask[None, :],
        other=0.0,
    )  # [N_CHUNK_SIZE, BLOCK_SIZE_C]

    # Iterate over the spatial dimensions (H, W) in blocks
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            spatial_mask = h_mask[:, None] & w_mask[None, :]

            # Load a block of x and y coordinates
            x_ptr_block = (
                x_ptr
                + b * stride_x_b
                + h_offsets[:, None] * stride_x_h
                + w_offsets[None, :] * stride_x_w
            )
            y_ptr_block = (
                y_ptr
                + b * stride_y_b
                + h_offsets[:, None] * stride_y_h
                + w_offsets[None, :] * stride_y_w
            )
            x_block = tl.load(
                x_ptr_block, mask=spatial_mask, other=0.0
            )  # [BLOCK_SIZE_H, BLOCK_SIZE_W]
            y_block = tl.load(
                y_ptr_block, mask=spatial_mask, other=0.0
            )  # [BLOCK_SIZE_H, BLOCK_SIZE_W]

            # Compute Gaussian weights for all Gaussians in chunk
            # Broadcasting: [N_CHUNK_SIZE, 1, 1] with [1, BLOCK_SIZE_H, BLOCK_SIZE_W]
            dx = (
                x_block[None, :, :] - pos_x[:, None, None]
            )  # [N_CHUNK_SIZE, BLOCK_SIZE_H, BLOCK_SIZE_W]
            dy = (
                y_block[None, :, :] - pos_y[:, None, None]
            )  # [N_CHUNK_SIZE, BLOCK_SIZE_H, BLOCK_SIZE_W]

            distance = (
                cov_inv_00[:, None, None] * (dx * dx)
                + cov_inv_11[:, None, None] * (dy * dy)
                + cov_inv_01[:, None, None] * (2 * dx * dy)
            )  # [N_CHUNK_SIZE, BLOCK_SIZE_H, BLOCK_SIZE_W]

            weight_chunk = (
                tl.exp(-0.5 * distance) * alpha[:, None, None]
            )  # [N_CHUNK_SIZE, BLOCK_SIZE_H, BLOCK_SIZE_W]

            # Mask out padded Gaussians
            weight_chunk = tl.where(gaussian_mask[:, None, None], weight_chunk, 0.0)

            # Update result sum - sum over the chunk dimension
            weight_sum_block = tl.sum(
                weight_chunk, axis=0
            )  # [BLOCK_SIZE_H, BLOCK_SIZE_W]

            result_sum_ptr_block = (
                result_sum_ptr
                + b * stride_result_sum_b
                + h_offsets[:, None] * stride_result_sum_h
                + w_offsets[None, :] * stride_result_sum_w
            )
            tl.atomic_add(result_sum_ptr_block, weight_sum_block, mask=spatial_mask)

            # Use tl.dot for efficient matrix multiplication
            # Reshape weight_chunk from [N_CHUNK_SIZE, BLOCK_SIZE_H, BLOCK_SIZE_W] to [BLOCK_SIZE_H*BLOCK_SIZE_W, N_CHUNK_SIZE]
            weight_reshaped = tl.trans(
                weight_chunk.reshape(N_CHUNK_SIZE, BLOCK_SIZE_H * BLOCK_SIZE_W)
            )

            # Perform matrix multiplication: [H*W, N_CHUNK_SIZE] @ [N_CHUNK_SIZE, C] -> [H*W, C]
            result_block_flat = tl.dot(
                weight_reshaped, colors_chunk
            )  # [BLOCK_SIZE_H*BLOCK_SIZE_W, BLOCK_SIZE_C]

            # Reshape back to [BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C]
            result_block = result_block_flat.reshape(
                BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
            )

            # Write result to global memory - since BLOCK_SIZE_C == C_padded, no loop needed
            c_offsets = tl.arange(0, BLOCK_SIZE_C)
            result_ptr_block = (
                result_ptr
                + b * stride_result_b
                + c_offsets[None, None, :] * stride_result_c
                + h_offsets[:, None, None] * stride_result_h
                + w_offsets[None, :, None] * stride_result_w
            )

            tl.atomic_add(
                result_ptr_block,
                result_block,
                mask=spatial_mask[:, :, None],
            )


@triton.jit
def gaussian_splatting_fused_backward_kernel_chunked(
    # Saved tensors from forward
    x_ptr,
    y_ptr,
    position_ptr,
    cov_inv_00_ptr,
    cov_inv_01_ptr,
    cov_inv_11_ptr,
    alphas_ptr,
    colors_ptr,
    result_sum_ptr,
    # Upstream gradient
    grad_result_ptr,
    # Output Gradients
    grad_x_ptr,
    grad_y_ptr,
    grad_position_ptr,
    grad_cov_inv_00_ptr,
    grad_cov_inv_01_ptr,
    grad_cov_inv_11_ptr,
    grad_alphas_ptr,
    grad_colors_ptr,
    # Dimensions
    B,
    H,
    W,
    N,
    C,
    C_padded,
    N_padded,
    N_CHUNK_SIZE: tl.constexpr,
    # Strides (same as forward)
    stride_x_b,
    stride_x_h,
    stride_x_w,
    stride_y_b,
    stride_y_h,
    stride_y_w,
    stride_pos_b,
    stride_pos_n,
    stride_pos_d,
    stride_cov_b,
    stride_cov_n,
    stride_alpha_b,
    stride_alpha_n,
    stride_colors_b,
    stride_colors_n,
    stride_colors_c,
    stride_result_sum_b,
    stride_result_sum_h,
    stride_result_sum_w,
    stride_grad_result_b,
    stride_grad_result_c,
    stride_grad_result_h,
    stride_grad_result_w,
    # Epsilon
    eps,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    THIS ONE have performance issue, currently we use not chunked impl for bwd
    """
    # Program IDs
    b = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    # Calculate the range of Gaussians for this chunk
    gaussian_start = chunk_idx * N_CHUNK_SIZE
    gaussian_offsets = gaussian_start + tl.arange(0, N_CHUNK_SIZE)
    gaussian_mask = gaussian_offsets < N

    # Load Gaussian parameters for the chunk
    pos_ptr_base = position_ptr + b * stride_pos_b + gaussian_offsets * stride_pos_n
    pos_x = tl.load(pos_ptr_base + 0 * stride_pos_d, mask=gaussian_mask, other=0.0)
    pos_y = tl.load(pos_ptr_base + 1 * stride_pos_d, mask=gaussian_mask, other=0.0)

    cov_inv_00 = tl.load(
        cov_inv_00_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    cov_inv_01 = tl.load(
        cov_inv_01_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    cov_inv_11 = tl.load(
        cov_inv_11_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n,
        mask=gaussian_mask,
        other=0.0,
    )
    alpha = tl.load(
        alphas_ptr + b * stride_alpha_b + gaussian_offsets * stride_alpha_n,
        mask=gaussian_mask,
        other=0.0,
    )

    # Load colors for the chunk
    color_offsets = tl.arange(0, BLOCK_SIZE_C)
    color_mask = color_offsets < C
    colors_ptr_base = (
        colors_ptr + b * stride_colors_b + gaussian_offsets[:, None] * stride_colors_n
    )
    colors_chunk = tl.load(
        colors_ptr_base + color_offsets[None, :] * stride_colors_c,
        mask=gaussian_mask[:, None] & color_mask[None, :],
        other=0.0,
    )

    # Initialize gradient accumulators for the chunk
    grad_pos_x_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_pos_y_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_cov_inv_00_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_cov_inv_11_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_cov_inv_01_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_alpha_acc = tl.zeros((N_CHUNK_SIZE,), dtype=tl.float32)
    grad_colors_acc = tl.zeros((N_CHUNK_SIZE, BLOCK_SIZE_C), dtype=tl.float32)

    # tl.static_print(b, chunk_idx, "start")

    # Iterate over spatial blocks
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            spatial_mask = h_mask[:, None] & w_mask[None, :]

            # Load spatial coordinates
            x_ptr_block = (
                x_ptr
                + b * stride_x_b
                + h_offsets[:, None] * stride_x_h
                + w_offsets[None, :] * stride_x_w
            )
            y_ptr_block = (
                y_ptr
                + b * stride_y_b
                + h_offsets[:, None] * stride_y_h
                + w_offsets[None, :] * stride_y_w
            )
            x_block = tl.load(x_ptr_block, mask=spatial_mask, other=0.0)
            y_block = tl.load(y_ptr_block, mask=spatial_mask, other=0.0)

            # Load result_sum and grad_result
            result_sum_ptr_block = (
                result_sum_ptr
                + b * stride_result_sum_b
                + h_offsets[:, None] * stride_result_sum_h
                + w_offsets[None, :] * stride_result_sum_w
            )
            result_sum_block = tl.load(
                result_sum_ptr_block, mask=spatial_mask, other=0.0
            )

            grad_result_ptr_block = (
                grad_result_ptr
                + b * stride_grad_result_b
                + color_offsets[None, None, :] * stride_grad_result_c
                + h_offsets[:, None, None] * stride_grad_result_h
                + w_offsets[None, :, None] * stride_grad_result_w
            )
            grad_result_block = tl.load(
                grad_result_ptr_block,
                mask=spatial_mask[:, :, None] & color_mask[None, None, :],
                other=0.0,
            )  # [BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C]

            # Recompute forward pass values
            dx = x_block[None, :, :] - pos_x[:, None, None]
            dy = y_block[None, :, :] - pos_y[:, None, None]
            distance = (
                cov_inv_00[:, None, None] * (dx * dx)
                + cov_inv_11[:, None, None] * (dy * dy)
                + cov_inv_01[:, None, None] * (2 * dx * dy)
            )
            gaussian_val = tl.exp(-0.5 * distance)
            gaussian_val_alpha = gaussian_val * alpha[:, None, None]

            # Mask out padded Gaussians
            gaussian_val_alpha = tl.where(
                gaussian_mask[:, None, None], gaussian_val_alpha, 0.0
            )

            # Gradient calculations
            denom = result_sum_block + eps
            grad_result_norm = grad_result_block / denom[:, :, None]

            # grad w.r.t colors using tl.dot
            # grad_result_norm: [H, W, C], gaussian_val_alpha: [N_CHUNK_SIZE, H, W]
            # Need: [N_CHUNK_SIZE, C] = sum over (H,W) of gaussian_val_alpha[n,h,w] * grad_result_norm[h,w,c]

            # Reshape for tl.dot: [N_CHUNK_SIZE, H*W] @ [H*W, C] -> [N_CHUNK_SIZE, C]
            gaussian_val_alpha_flat = gaussian_val_alpha.reshape(
                N_CHUNK_SIZE, BLOCK_SIZE_H * BLOCK_SIZE_W
            )
            grad_result_norm_flat = grad_result_norm.reshape(
                BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_C
            )

            # Apply spatial mask
            spatial_mask_flat = spatial_mask.reshape(BLOCK_SIZE_H * BLOCK_SIZE_W)
            grad_result_norm_masked = tl.where(
                spatial_mask_flat[:, None], grad_result_norm_flat, 0.0
            )

            grad_colors_term = tl.dot(gaussian_val_alpha_flat, grad_result_norm_masked)
            grad_colors_acc += grad_colors_term

            # grad w.r.t result_sum and other parameters
            result_block = (
                gaussian_val_alpha[:, :, :, None] * colors_chunk[:, None, None, :]
            )  # [N_CHUNK_SIZE, H, W, C]
            grad_result_sum_block = (
                -tl.sum(grad_result_norm[None, :, :, :] * result_block, axis=3)
                / denom[None, :, :]
            )
            grad_result_sum_block = tl.sum(
                grad_result_sum_block, axis=0
            )  # Sum over chunk dimension

            grad_result_curr = tl.sum(
                grad_result_norm[None, :, :, :] * colors_chunk[:, None, None, :], axis=3
            )

            # Combined gradient for weight
            grad_gaussian = grad_result_curr + grad_result_sum_block[None, :, :]

            # Apply spatial mask
            grad_gaussian = tl.where(spatial_mask[None, :, :], grad_gaussian, 0.0)

            # Accumulate gradients
            grad_alpha_acc += tl.sum(
                tl.sum(grad_gaussian * gaussian_val, axis=1), axis=1
            )

            common_factor = grad_gaussian * gaussian_val_alpha * -0.5

            # Gradients for x, y
            grad_dist_dx = common_factor * (
                2 * cov_inv_00[:, None, None] * dx + 2 * cov_inv_01[:, None, None] * dy
            )
            grad_dist_dy = common_factor * (
                2 * cov_inv_11[:, None, None] * dy + 2 * cov_inv_01[:, None, None] * dx
            )

            # Sum over chunk dimension for spatial gradients
            grad_x_sum = tl.sum(grad_dist_dx, axis=0)
            grad_y_sum = tl.sum(grad_dist_dy, axis=0)

            grad_x_ptr_block = (
                grad_x_ptr
                + b * stride_x_b
                + h_offsets[:, None] * stride_x_h
                + w_offsets[None, :] * stride_x_w
            )
            grad_y_ptr_block = (
                grad_y_ptr
                + b * stride_y_b
                + h_offsets[:, None] * stride_y_h
                + w_offsets[None, :] * stride_y_w
            )
            tl.atomic_add(grad_x_ptr_block, grad_x_sum, mask=spatial_mask)
            tl.atomic_add(grad_y_ptr_block, grad_y_sum, mask=spatial_mask)

            # Accumulate gradients for position and covariance
            grad_pos_x_acc -= tl.sum(tl.sum(grad_dist_dx, axis=1), axis=1)
            grad_pos_y_acc -= tl.sum(tl.sum(grad_dist_dy, axis=1), axis=1)
            grad_cov_inv_00_acc += tl.sum(
                tl.sum(common_factor * (dx * dx), axis=1), axis=1
            )
            grad_cov_inv_11_acc += tl.sum(
                tl.sum(common_factor * (dy * dy), axis=1), axis=1
            )
            grad_cov_inv_01_acc += tl.sum(
                tl.sum(common_factor * (2 * dx * dy), axis=1), axis=1
            )

    # tl.static_print(b, chunk_idx, "done")
    # Write accumulated gradients to global memory
    grad_pos_ptr_base = (
        grad_position_ptr + b * stride_pos_b + gaussian_offsets * stride_pos_n
    )
    tl.store(grad_pos_ptr_base + 0 * stride_pos_d, grad_pos_x_acc, mask=gaussian_mask)
    tl.store(grad_pos_ptr_base + 1 * stride_pos_d, grad_pos_y_acc, mask=gaussian_mask)

    grad_cov_ptr_base = (
        grad_cov_inv_00_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n
    )
    tl.store(grad_cov_ptr_base, grad_cov_inv_00_acc, mask=gaussian_mask)

    grad_cov_ptr_base_11 = (
        grad_cov_inv_11_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n
    )
    tl.store(grad_cov_ptr_base_11, grad_cov_inv_11_acc, mask=gaussian_mask)

    grad_cov_ptr_base_01 = (
        grad_cov_inv_01_ptr + b * stride_cov_b + gaussian_offsets * stride_cov_n
    )
    tl.store(grad_cov_ptr_base_01, grad_cov_inv_01_acc, mask=gaussian_mask)

    grad_alpha_ptr_base = (
        grad_alphas_ptr + b * stride_alpha_b + gaussian_offsets * stride_alpha_n
    )
    tl.store(grad_alpha_ptr_base, grad_alpha_acc, mask=gaussian_mask)

    grad_colors_ptr_base = (
        grad_colors_ptr
        + b * stride_colors_b
        + gaussian_offsets[:, None] * stride_colors_n
    )
    tl.store(
        grad_colors_ptr_base + color_offsets[None, :] * stride_colors_c,
        grad_colors_acc,
        mask=gaussian_mask[:, None] & color_mask[None, :],
    )
    # tl.static_print(b, chunk_idx, "saved")


class TritonGaussianSplatting2DChunked(Function):
    # Block sizes optimized for chunked processing
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 32
    BWD_BLOCK_SIZE_H = 8
    BWD_BLOCK_SIZE_W = 64
    N_CHUNK_SIZE = 32

    @staticmethod
    def forward(
        ctx,
        x,
        y,
        position,
        cov_inv_00,
        cov_inv_01,
        cov_inv_11,
        alphas,
        colors,
        eps=1e-6,
    ):
        # Make inputs contiguous
        dtype = torch.float32
        x = x.contiguous().to(dtype)
        y = y.contiguous().to(dtype)
        position = position.contiguous().to(dtype)
        cov_inv_00 = cov_inv_00.contiguous().to(dtype)
        cov_inv_01 = cov_inv_01.contiguous().to(dtype)
        cov_inv_11 = cov_inv_11.contiguous().to(dtype)
        alphas = alphas.contiguous().to(dtype)
        colors = colors.contiguous().to(dtype)
        B, _, H, W = x.shape
        N = position.size(1)
        C = colors.size(2)

        # Pad N to be divisible by N_CHUNK_SIZE
        N_padded = (
            (N + TritonGaussianSplatting2DChunked.N_CHUNK_SIZE - 1)
            // TritonGaussianSplatting2DChunked.N_CHUNK_SIZE
        ) * TritonGaussianSplatting2DChunked.N_CHUNK_SIZE

        if N != N_padded:
            padding_n = N_padded - N
            position_padded = F.pad(position, (0, 0, 0, padding_n), "constant", 0)
            cov_inv_00_padded = F.pad(cov_inv_00, (0, 0, 0, padding_n), "constant", 0)
            cov_inv_01_padded = F.pad(cov_inv_01, (0, 0, 0, padding_n), "constant", 0)
            cov_inv_11_padded = F.pad(cov_inv_11, (0, 0, 0, padding_n), "constant", 0)
            alphas_padded = F.pad(alphas, (0, padding_n), "constant", 0)
            colors_padded = F.pad(colors, (0, 0, 0, padding_n), "constant", 0)
        else:
            position_padded = position
            cov_inv_00_padded = cov_inv_00
            cov_inv_01_padded = cov_inv_01
            cov_inv_11_padded = cov_inv_11
            alphas_padded = alphas
            colors_padded = colors

        # Pad colors to minimum 16 and ensure power of 2
        C_padded = max(16, next_power_of_2(C))
        if C != C_padded:
            padding_c = C_padded - C
            colors_padded = F.pad(colors_padded, (0, padding_c), "constant", 0)

        result = torch.zeros(B, C_padded, H, W, device=x.device, dtype=x.dtype)
        result_sum = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)

        grid = (B, N_padded // TritonGaussianSplatting2DChunked.N_CHUNK_SIZE)
        BLOCK_SIZE_C = C_padded
        N_CHUNK_SIZE = TritonGaussianSplatting2DChunked.N_CHUNK_SIZE

        gaussian_splatting_fused_forward_kernel_chunked[grid](
            x.squeeze(1),
            y.squeeze(1),
            position_padded,
            cov_inv_00_padded.squeeze(-1),
            cov_inv_01_padded.squeeze(-1),
            cov_inv_11_padded.squeeze(-1),
            alphas_padded,
            colors_padded,
            result,
            result_sum,
            B,
            H,
            W,
            N,
            C,
            C_padded,
            N_padded,
            N_CHUNK_SIZE,
            x.stride(0),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(2),
            y.stride(3),
            position_padded.stride(0),
            position_padded.stride(1),
            position_padded.stride(2),
            cov_inv_00_padded.stride(0),
            cov_inv_00_padded.stride(1),
            alphas_padded.stride(0),
            alphas_padded.stride(1),
            colors_padded.stride(0),
            colors_padded.stride(1),
            colors_padded.stride(2),
            result.stride(0),
            result.stride(1),
            result.stride(2),
            result.stride(3),
            result_sum.stride(0),
            result_sum.stride(1),
            result_sum.stride(2),
            BLOCK_SIZE_H=TritonGaussianSplatting2DChunked.BLOCK_SIZE_H,
            BLOCK_SIZE_W=TritonGaussianSplatting2DChunked.BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

        result_sum_plus_eps = result_sum + eps
        result_normalized = result / result_sum_plus_eps[:, None, :, :]

        # Truncate result back to original channel size
        result_final = result_normalized[:, :C, :, :]

        ctx.save_for_backward(
            x,
            y,
            position_padded,
            cov_inv_00_padded,
            cov_inv_01_padded,
            cov_inv_11_padded,
            alphas_padded,
            colors_padded,
            result_sum,
        )
        ctx.eps = eps
        ctx.C, ctx.C_padded = C, C_padded
        ctx.N, ctx.N_padded = N, N_padded
        return result_final

    @staticmethod
    def _backward(ctx, grad_result):
        # print("Start Backward")
        (
            x,
            y,
            position_padded,
            cov_inv_00_padded,
            cov_inv_01_padded,
            cov_inv_11_padded,
            alphas_padded,
            colors_padded,
            result_sum,
        ) = ctx.saved_tensors
        eps = ctx.eps
        C, C_padded = ctx.C, ctx.C_padded
        N, N_padded = ctx.N, ctx.N_padded

        B, _, H, W = x.shape

        grad_result = grad_result.contiguous()

        # Pad grad_result to match padded colors
        if C != C_padded:
            padding = C_padded - C
            grad_result_padded = F.pad(
                grad_result, (0, 0, 0, 0, 0, padding), "constant", 0
            )
        else:
            grad_result_padded = grad_result

        # Initialize gradient tensors
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_position_padded = torch.zeros_like(position_padded)
        grad_cov_inv_00_padded = torch.zeros_like(cov_inv_00_padded)
        grad_cov_inv_01_padded = torch.zeros_like(cov_inv_01_padded)
        grad_cov_inv_11_padded = torch.zeros_like(cov_inv_11_padded)
        grad_alphas_padded = torch.zeros_like(alphas_padded)
        grad_colors_padded = torch.zeros_like(colors_padded)

        grid = (B, N_padded // TritonGaussianSplatting2DChunked.N_CHUNK_SIZE)
        BLOCK_SIZE_C = C_padded
        N_CHUNK_SIZE = TritonGaussianSplatting2DChunked.N_CHUNK_SIZE
        BLOCK_SIZE_H = TritonGaussianSplatting2DChunked.BWD_BLOCK_SIZE_H
        BLOCK_SIZE_W = TritonGaussianSplatting2DChunked.BWD_BLOCK_SIZE_W

        gaussian_splatting_fused_backward_kernel_chunked[grid](
            x.squeeze(1),
            y.squeeze(1),
            position_padded,
            cov_inv_00_padded.squeeze(-1),
            cov_inv_01_padded.squeeze(-1),
            cov_inv_11_padded.squeeze(-1),
            alphas_padded,
            colors_padded,
            result_sum,
            grad_result_padded,
            grad_x.squeeze(1),
            grad_y.squeeze(1),
            grad_position_padded,
            grad_cov_inv_00_padded.squeeze(-1),
            grad_cov_inv_01_padded.squeeze(-1),
            grad_cov_inv_11_padded.squeeze(-1),
            grad_alphas_padded,
            grad_colors_padded,
            B,
            H,
            W,
            N,
            C,
            C_padded,
            N_padded,
            N_CHUNK_SIZE,
            x.stride(0),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(2),
            y.stride(3),
            position_padded.stride(0),
            position_padded.stride(1),
            position_padded.stride(2),
            cov_inv_00_padded.stride(0),
            cov_inv_00_padded.stride(1),
            alphas_padded.stride(0),
            alphas_padded.stride(1),
            colors_padded.stride(0),
            colors_padded.stride(1),
            colors_padded.stride(2),
            result_sum.stride(0),
            result_sum.stride(1),
            result_sum.stride(2),
            grad_result_padded.stride(0),
            grad_result_padded.stride(1),
            grad_result_padded.stride(2),
            grad_result_padded.stride(3),
            eps,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

        # Truncate gradients back to original sizes
        grad_position_final = grad_position_padded[:, :N, :]
        grad_cov_inv_00_final = grad_cov_inv_00_padded[:, :N, :]
        grad_cov_inv_01_final = grad_cov_inv_01_padded[:, :N, :]
        grad_cov_inv_11_final = grad_cov_inv_11_padded[:, :N, :]
        grad_alphas_final = grad_alphas_padded[:, :N]
        grad_colors_final = grad_colors_padded[:, :N, :C]

        return (
            grad_x,
            grad_y,
            grad_position_final,
            grad_cov_inv_00_final,
            grad_cov_inv_01_final,
            grad_cov_inv_11_final,
            grad_alphas_final,
            grad_colors_final,
            None,
        )

    @staticmethod
    def backward(ctx, grad_result):
        (
            x,
            y,
            position,
            cov_inv_00,
            cov_inv_01,
            cov_inv_11,
            alphas,
            colors_padded,
            result_sum,
        ) = ctx.saved_tensors
        eps = ctx.eps
        C, C_padded = ctx.C, ctx.C_padded
        C_padded = next_power_of_2(C)

        B, _, H, W = x.shape
        N = position.size(1)

        grad_result = grad_result.contiguous()

        # Pad grad_result to match padded colors
        if C != C_padded:
            padding = C_padded - C
            grad_result_padded = F.pad(
                grad_result, (0, 0, 0, 0, 0, padding), "constant", 0
            )
        else:
            grad_result_padded = grad_result

        # Initialize gradient tensors
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_position = torch.zeros_like(position)
        grad_cov_inv_00 = torch.zeros_like(cov_inv_00)
        grad_cov_inv_01 = torch.zeros_like(cov_inv_01)
        grad_cov_inv_11 = torch.zeros_like(cov_inv_11)
        grad_alphas = torch.zeros_like(alphas)
        grad_colors = torch.zeros_like(colors_padded)

        grid = (B, N)
        BLOCK_SIZE_C = C_padded
        BLOCK_SIZE_H = (
            TritonGaussianSplatting2DChunked.BWD_BLOCK_SIZE_H
            or TritonGaussianSplatting2DChunked.BLOCK_SIZE_H
        )
        BLOCK_SIZE_W = (
            TritonGaussianSplatting2DChunked.BWD_BLOCK_SIZE_W
            or TritonGaussianSplatting2DChunked.BLOCK_SIZE_W
        )

        gaussian_splatting_fused_backward_kernel[grid](
            x.squeeze(1),
            y.squeeze(1),
            position,
            cov_inv_00.squeeze(-1),
            cov_inv_01.squeeze(-1),
            cov_inv_11.squeeze(-1),
            alphas,
            colors_padded,
            result_sum,
            grad_result_padded,
            grad_x.squeeze(1),
            grad_y.squeeze(1),
            grad_position,
            grad_cov_inv_00.squeeze(-1),
            grad_cov_inv_01.squeeze(-1),
            grad_cov_inv_11.squeeze(-1),
            grad_alphas,
            grad_colors,
            B,
            H,
            W,
            N,
            C,
            C_padded,
            x.stride(0),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(2),
            y.stride(3),
            position.stride(0),
            position.stride(1),
            position.stride(2),
            cov_inv_00.stride(0),
            cov_inv_00.stride(1),
            alphas.stride(0),
            alphas.stride(1),
            colors_padded.stride(0),
            colors_padded.stride(1),
            colors_padded.stride(2),
            result_sum.stride(0),
            result_sum.stride(1),
            result_sum.stride(2),
            grad_result_padded.stride(0),
            grad_result_padded.stride(1),
            grad_result_padded.stride(2),
            grad_result_padded.stride(3),
            eps,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

        # Truncate grad_colors back to original size
        grad_colors_final = grad_colors[:, :, :C]

        return (
            grad_x,
            grad_y,
            grad_position,
            grad_cov_inv_00,
            grad_cov_inv_01,
            grad_cov_inv_11,
            grad_alphas,
            grad_colors_final,
            None,
        )


# Test code
if __name__ == "__main__":
    from igs.gs2d import GaussianSplatting2DKernel, GaussianSplatting2DChunkedKernel
    from igs.gs_triton import TritonGaussianSplatting2D

    TritonGaussianSplatting2DChunked.backward = TritonGaussianSplatting2D.backward

    B, N, H, W = 4, 2048, 256, 256

    x = torch.rand(B, 1, H, W).cuda().requires_grad_(True)
    y = torch.rand(B, 1, H, W).cuda().requires_grad_(True)
    cov_inv_00 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    cov_inv_01 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    cov_inv_11 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    alphas = torch.rand(B, N).cuda().requires_grad_(True)
    position = torch.rand(B, N, 2).cuda().requires_grad_(True)
    colors = torch.rand(B, N, 3).cuda().requires_grad_(True)

    # Test chunked Triton implementation
    output_chunked = TritonGaussianSplatting2DChunked.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    output_chunked.mean().backward()
    grads_chunked = (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    )

    # Clear gradients
    (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    ) = [None] * 8
    torch.cuda.empty_cache()

    # Test original PyTorch chunked implementation
    output_torch_chunked = GaussianSplatting2DChunkedKernel.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    output_torch_chunked.mean().backward()
    grads_torch_chunked = (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    )

    print("Comparing Triton Chunked vs PyTorch Chunked:")
    print(f"Output MSE: {F.mse_loss(output_chunked, output_torch_chunked)}")
    print(f"Output L1: {F.l1_loss(output_chunked, output_torch_chunked)}")

    for i, (g1, g2) in enumerate(zip(grads_chunked, grads_torch_chunked)):
        grad_names = [
            "x",
            "y",
            "position",
            "cov_inv_00",
            "cov_inv_01",
            "cov_inv_11",
            "alphas",
            "colors",
        ]
        print(
            f"{grad_names[i]} MSE: {F.mse_loss(g1, g2):.6f}, L1: {F.l1_loss(g1, g2):.6f}"
        )
