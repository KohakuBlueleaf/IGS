import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


# Helper function to find the next power of 2
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
def gaussian_splatting_fused_forward_kernel(
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
    # Get the program IDs for batch and Gaussian
    b = tl.program_id(0)
    gaussian = tl.program_id(1)

    # Load Gaussian parameters
    pos_ptr_base = position_ptr + b * stride_pos_b + gaussian * stride_pos_n
    pos_x = tl.load(pos_ptr_base)
    pos_y = tl.load(pos_ptr_base + stride_pos_d)

    cov_inv_00 = tl.load(cov_inv_00_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    cov_inv_01 = tl.load(cov_inv_01_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    cov_inv_11 = tl.load(cov_inv_11_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    alpha = tl.load(alphas_ptr + b * stride_alpha_b + gaussian * stride_alpha_n)

    # Load colors for the current Gaussian
    color_offsets = tl.arange(0, BLOCK_SIZE_C)
    color_mask = color_offsets < C
    colors_ptr_base = colors_ptr + b * stride_colors_b + gaussian * stride_colors_n
    color = tl.load(
        colors_ptr_base + color_offsets * stride_colors_c, mask=color_mask, other=0.0
    )

    # Iterate over the spatial dimensions (H, W) in blocks
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            mask = h_mask[:, None] & w_mask[None, :]

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
            x_block = tl.load(x_ptr_block, mask=mask, other=0.0)
            y_block = tl.load(y_ptr_block, mask=mask, other=0.0)

            # Compute Gaussian weights
            dx = x_block - pos_x
            dy = y_block - pos_y
            distance = (
                cov_inv_00 * (dx * dx)
                + cov_inv_11 * (dy * dy)
                + cov_inv_01 * (2 * dx * dy)
            )
            weight_block = tl.exp(-0.5 * distance) * alpha

            # Update result sum
            result_sum_ptr_block = (
                result_sum_ptr
                + b * stride_result_sum_b
                + h_offsets[:, None] * stride_result_sum_h
                + w_offsets[None, :] * stride_result_sum_w
            )
            tl.atomic_add(result_sum_ptr_block, weight_block, mask=mask)

            # Update final result (color weighted sum)
            for c_idx in range(0, C_padded, BLOCK_SIZE_C):
                c_offsets = c_idx + tl.arange(0, BLOCK_SIZE_C)
                c_mask = c_offsets < C_padded

                # Pointers to the output result tensor
                result_ptr_block = (
                    result_ptr
                    + b * stride_result_b
                    + c_offsets[None, None, :] * stride_result_c
                    + h_offsets[:, None, None] * stride_result_h
                    + w_offsets[None, :, None] * stride_result_w
                )

                # Perform matmul and atomically add
                # weight_block is [BLOCK_H, BLOCK_W], color is [BLOCK_C]
                # we need to add weight_block[:, :, None] * color[None, None, :]
                update = weight_block[:, :, None] * color[None, None, :]

                tl.atomic_add(
                    result_ptr_block,
                    update,
                    mask=mask[:, :, None] & c_mask[None, None, :],
                )


@triton.jit
def gaussian_splatting_fused_backward_kernel(
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
    # Program IDs
    b = tl.program_id(0)
    gaussian = tl.program_id(1)

    # Load Gaussian parameters
    pos_ptr_base = position_ptr + b * stride_pos_b + gaussian * stride_pos_n
    pos_x = tl.load(pos_ptr_base)
    pos_y = tl.load(pos_ptr_base + stride_pos_d)

    cov_inv_00 = tl.load(cov_inv_00_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    cov_inv_01 = tl.load(cov_inv_01_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    cov_inv_11 = tl.load(cov_inv_11_ptr + b * stride_cov_b + gaussian * stride_cov_n)
    alpha = tl.load(alphas_ptr + b * stride_alpha_b + gaussian * stride_alpha_n)

    # Load colors
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    colors_ptr_base = colors_ptr + b * stride_colors_b + gaussian * stride_colors_n
    color = tl.load(
        colors_ptr_base + c_offsets * stride_colors_c, mask=c_mask, other=0.0
    )

    # Initialize gradient accumulators
    grad_pos_x_acc = 0.0
    grad_pos_y_acc = 0.0
    grad_cov_inv_00_acc = 0.0
    grad_cov_inv_11_acc = 0.0
    grad_cov_inv_01_acc = 0.0
    grad_alpha_acc = 0.0
    grad_colors_acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)

    # Iterate over spatial blocks
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            mask = h_mask[:, None] & w_mask[None, :]

            # Load data for the block
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
            x_block = tl.load(x_ptr_block, mask=mask, other=0.0)
            y_block = tl.load(y_ptr_block, mask=mask, other=0.0)

            result_sum_ptr_block = (
                result_sum_ptr
                + b * stride_result_sum_b
                + h_offsets[:, None] * stride_result_sum_h
                + w_offsets[None, :] * stride_result_sum_w
            )
            result_sum_block = tl.load(result_sum_ptr_block, mask=mask, other=0.0)

            # Recompute forward pass values
            dx = x_block - pos_x
            dy = y_block - pos_y
            distance = (
                cov_inv_00 * (dx * dx)
                + cov_inv_11 * (dy * dy)
                + cov_inv_01 * (2 * dx * dy)
            )
            gaussian_val = tl.exp(-0.5 * distance)
            gaussian_val_alpha = gaussian_val * alpha

            # Load upstream gradient
            grad_result_c_offsets = tl.arange(0, BLOCK_SIZE_C)
            grad_result_c_mask = grad_result_c_offsets < C_padded
            grad_result_ptr_block = (
                grad_result_ptr
                + b * stride_grad_result_b
                + grad_result_c_offsets[None, None, :] * stride_grad_result_c
                + h_offsets[:, None, None] * stride_grad_result_h
                + w_offsets[None, :, None] * stride_grad_result_w
            )
            grad_result_block = tl.load(
                grad_result_ptr_block,
                mask=mask[:, :, None] & grad_result_c_mask[None, None, :],
                other=0.0,
            )

            # --- Start Gradient Calculation ---
            denom = result_sum_block + eps
            grad_result_norm = grad_result_block / denom[:, :, None]

            # grad w.r.t colors
            # Original term to sum over H and W dimensions
            grad_colors_term = (
                grad_result_norm * gaussian_val_alpha[:, :, None]
            ) * mask[:, :, None]

            # Perform summation in two steps since axis=(0, 1) is not supported
            sum_over_h = tl.sum(
                grad_colors_term, axis=0
            )  # Sum over H, result is [BLOCK_W, BLOCK_C]
            sum_over_hw = tl.sum(sum_over_h, axis=0)  # Sum over W, result is [BLOCK_C]

            grad_colors_acc += sum_over_hw

            # grad w.r.t result_sum
            # Simplified from -(grad_result * result).sum(dim=1) / (denom)
            # which becomes -(grad_result_norm * color_weighted_gaussian).sum(dim=1) / denom
            # Let's follow the user's PyTorch code for consistency
            result_block = gaussian_val_alpha[:, :, None] * color[None, None, :]
            grad_result_sum_block = (
                -tl.sum(grad_result_norm * result_block, axis=2) / denom
            )

            # grad w.r.t individual gaussian contribution
            grad_result_curr = tl.sum(grad_result_norm * color[None, None, :], axis=2)

            # Combined gradient for the weight
            grad_gaussian = grad_result_curr + grad_result_sum_block

            # grad w.r.t alpha
            grad_alpha_acc += tl.sum((grad_gaussian * gaussian_val) * mask)

            common_factor = grad_gaussian * gaussian_val_alpha * -0.5

            # Gradients for x, y
            grad_dist_dx = common_factor * (2 * cov_inv_00 * dx + 2 * cov_inv_01 * dy)
            grad_dist_dy = common_factor * (2 * cov_inv_11 * dy + 2 * cov_inv_01 * dx)

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
            tl.atomic_add(grad_x_ptr_block, grad_dist_dx, mask=mask)
            tl.atomic_add(grad_y_ptr_block, grad_dist_dy, mask=mask)

            # Accumulate gradients for position and covariance
            grad_pos_x_acc -= tl.sum(grad_dist_dx * mask)
            grad_pos_y_acc -= tl.sum(grad_dist_dy * mask)
            grad_cov_inv_00_acc += tl.sum((common_factor * (dx * dx)) * mask)
            grad_cov_inv_11_acc += tl.sum((common_factor * (dy * dy)) * mask)
            grad_cov_inv_01_acc += tl.sum((common_factor * (2 * dx * dy)) * mask)

    # Write accumulated gradients to global memory
    grad_pos_ptr_base = grad_position_ptr + b * stride_pos_b + gaussian * stride_pos_n
    tl.store(grad_pos_ptr_base, grad_pos_x_acc)
    tl.store(grad_pos_ptr_base + stride_pos_d, grad_pos_y_acc)

    grad_cov_ptr_base = grad_cov_inv_00_ptr + b * stride_cov_b + gaussian * stride_cov_n
    tl.store(grad_cov_ptr_base, grad_cov_inv_00_acc)
    grad_cov_ptr_base_11 = (
        grad_cov_inv_11_ptr + b * stride_cov_b + gaussian * stride_cov_n
    )
    tl.store(grad_cov_ptr_base_11, grad_cov_inv_11_acc)
    grad_cov_ptr_base_01 = (
        grad_cov_inv_01_ptr + b * stride_cov_b + gaussian * stride_cov_n
    )
    tl.store(grad_cov_ptr_base_01, grad_cov_inv_01_acc)

    grad_alpha_ptr_base = (
        grad_alphas_ptr + b * stride_alpha_b + gaussian * stride_alpha_n
    )
    tl.store(grad_alpha_ptr_base, grad_alpha_acc)

    grad_colors_ptr_base = (
        grad_colors_ptr + b * stride_colors_b + gaussian * stride_colors_n
    )
    grad_c_offsets = tl.arange(0, BLOCK_SIZE_C)
    grad_c_mask = grad_c_offsets < C
    tl.store(
        grad_colors_ptr_base + grad_c_offsets * stride_colors_c,
        grad_colors_acc,
        mask=grad_c_mask,
    )


class TritonGaussianSplatting2D(Function):
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

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
        x = x.contiguous()
        y = y.contiguous()
        position = position.contiguous()
        cov_inv_00 = cov_inv_00.contiguous()
        cov_inv_01 = cov_inv_01.contiguous()
        cov_inv_11 = cov_inv_11.contiguous()
        alphas = alphas.contiguous()
        colors = colors.contiguous()

        B, _, H, W = x.shape
        N = position.size(1)
        C = colors.size(2)

        # Pad colors if C is not a power of 2
        C_padded = next_power_of_2(C) if C > 1 else 1
        if C != C_padded:
            padding = C_padded - C
            colors_padded = F.pad(colors, (0, padding), "constant", 0)
        else:
            colors_padded = colors

        result = torch.zeros(B, C_padded, H, W, device=x.device, dtype=x.dtype)
        result_sum = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)

        grid = (B, N)
        BLOCK_SIZE_C = C_padded

        gaussian_splatting_fused_forward_kernel[grid](
            x.squeeze(1),
            y.squeeze(1),
            position,
            cov_inv_00.squeeze(-1),
            cov_inv_01.squeeze(-1),
            cov_inv_11.squeeze(-1),
            alphas,
            colors_padded,
            result,
            result_sum,
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
            result.stride(0),
            result.stride(1),
            result.stride(2),
            result.stride(3),
            result_sum.stride(0),
            result_sum.stride(1),
            result_sum.stride(2),
            BLOCK_SIZE_H=TritonGaussianSplatting2D.BLOCK_SIZE_H,
            BLOCK_SIZE_W=TritonGaussianSplatting2D.BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

        result_sum_plus_eps = result_sum + eps
        result_normalized = result / result_sum_plus_eps[:, None, :, :]

        # Truncate result back to original channel size
        result_final = result_normalized[:, :C, :, :]

        ctx.save_for_backward(
            x,
            y,
            position,
            cov_inv_00,
            cov_inv_01,
            cov_inv_11,
            alphas,
            colors_padded,
            result_sum,
        )
        ctx.eps = eps
        ctx.C, ctx.C_padded = C, C_padded
        return result_final

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
            BLOCK_SIZE_H=TritonGaussianSplatting2D.BLOCK_SIZE_H,
            BLOCK_SIZE_W=TritonGaussianSplatting2D.BLOCK_SIZE_W,
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


if __name__ == "__main__":
    from .gs2d import GaussianSplatting2DKernel

    B, N, H, W = 4, 2048, 256, 256

    x = torch.rand(B, 1, H, W).cuda().requires_grad_(True)
    y = torch.rand(B, 1, H, W).cuda().requires_grad_(True)
    cov_inv_00 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    cov_inv_01 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    cov_inv_11 = torch.randn(B, N, 1, 1).cuda().requires_grad_(True)
    alphas = torch.rand(B, N).cuda().requires_grad_(True)
    position = torch.rand(B, N, 2).cuda().requires_grad_(True)
    colors = torch.rand(B, N, 3).cuda().requires_grad_(True)

    output1 = TritonGaussianSplatting2D.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    output1.mean().backward()
    grads = (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    )
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

    output2 = GaussianSplatting2DKernel.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    output2.mean().backward()
    grads2 = (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    )

    print(F.mse_loss(output1, output2), F.l1_loss(output1, output2))
    for g1, g2 in zip(grads, grads2):
        print(F.mse_loss(g1, g2), F.l1_loss(g1, g2))
