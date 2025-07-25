import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import torch.autograd as autograd

try:
    import triton
    import triton.language as tl
    from .gs_triton import TritonGaussianSplatting2D
    from .gs_triton_chunked import TritonGaussianSplatting2DChunked
except ImportError:
    TritonGaussianSplatting2D = None
    pass


torch.set_float32_matmul_precision("high")


def log_vram(prefix=""):
    return


def naive_gaussian_2d(
    x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors, eps=1e-6
):
    dx = x - position[..., 0, None, None]
    dy = y - position[..., 1, None, None]
    weight = (
        torch.exp(
            -0.5
            * (dx * dx * cov_inv_00 + 2 * dx * dy * cov_inv_01 + dy * dy * cov_inv_11)
        )
        * alphas[..., None, None]
    )
    weight_sum = weight.sum(dim=1, keepdim=True)

    b, ng, h, w = weight.shape
    normalized_weights = weight / (weight_sum + eps)
    normalized_weights = normalized_weights.permute(0, 2, 3, 1).flatten(1, 2)
    result = torch.bmm(
        normalized_weights, colors
    )  # [b, H*W, ng] @ [b, ng, 3] = [b, H*W, 3]
    return result.unflatten(1, (h, w)).permute(0, 3, 1, 2)


class GaussianSplatting2DKernel(autograd.Function):
    """
    A totally fused custom Autograd function for 2D gaussian splatting

    The overall idea is:

    weight, weight_sum = GaussianWeight()
    normalized_weight = weight/(weight_sum + eps)
    result = weight @ color

    ^ The size of "weight" is large, so we try to avoid it

    Since shape of weight_sum is [B, H, W], we can actually do things like:
    weight, weight_sum = GaussianWeight()
    result = (weight @ color) / (weight_sum[:, None] + eps)

    and if we fuse the weigh @ color into one kernel
    which done matmul when we iterate through all the gaussians
    we can avoid B, N, H, W intermediate state
    """

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
        """
        x, y: [B, 1, H, W]
        position: [B, N, 2]
        cov_inv_00: [B, N, 1]
        cov_inv_01: [B, N, 1]
        cov_inv_11: [B, N, 1]
        alphas: [B, N]
        colors: [B, N, 3]

        dx = x - position
        dy = y - position
        distance_matrix = cov_inv_00 * (dx ** 2) + cov_inv_11 * (dy ** 2) + cov_inv_01 * (2 * dx * dy)
        weight = exp(-0.5 * distance_matrix) * alphas
        weight = weight.sum(dim=1, keepdim=True)
        norm_weight = weight / (weight_sum+eps)
        result = einsum("bnhw, bnc -> bchw", norm_weight, colors)
        # You can also say it is:
        result = einsum("bnhw, bnc -> bchw", weight, colors) / (weight_sum + eps)
        # This allow us to seperate the operation of "final output" and "norm weight"
        """
        x = x[:, 0]
        y = y[:, 0]
        B, H, W = x.shape
        N = position.size(1)
        B, N, C = colors.shape

        result = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        weight_sum = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
        log_vram("Result Init")
        for gaussian in range(N):
            dx = x - position[:, gaussian, 0, None, None]  # [B, H, W]
            dy = y - position[:, gaussian, 1, None, None]  # [B, H, W]
            weight_block = (
                torch.exp(
                    -0.5
                    * (
                        cov_inv_00[:, gaussian] * (dx**2)
                        + cov_inv_11[:, gaussian] * (dy**2)
                        + cov_inv_01[:, gaussian] * (2 * dx * dy)
                    )
                )
                * alphas[:, gaussian, None, None]
            )  # [B, H, W]
            # perform matmul which iterate through N dim
            result += (
                weight_block[:, None] * colors[:, gaussian, :, None, None]
            )  # [B, C, H, W]
            weight_sum += weight_block

        result /= (weight_sum + eps)[:, None, :, :]

        ctx.save_for_backward(
            x,
            y,
            position,
            cov_inv_00,
            cov_inv_01,
            cov_inv_11,
            alphas,
            colors,
            weight_sum,
        )
        ctx.eps = eps
        return result

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
            colors,
            result_sum,
        ) = ctx.saved_tensors
        eps = ctx.eps
        B, H, W = x.shape
        N = position.size(1)

        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_position = torch.zeros_like(position)
        grad_cov_inv_00 = torch.zeros_like(cov_inv_00)
        grad_cov_inv_01 = torch.zeros_like(cov_inv_01)
        grad_cov_inv_11 = torch.zeros_like(cov_inv_11)
        grad_alphas = torch.zeros_like(alphas)
        grad_colors = torch.zeros_like(colors)

        # grad result is [B, C, H, W] where C=3 so we directly use it
        # if C is large than we should do iterative process just like for g in range(N)
        result_sum += eps
        grad_result_sum = -(grad_result * grad_result).sum(dim=1) / (result_sum**2)
        grad_result /= result_sum[:, None, :, :]

        # Process each Gaussian separately to save memory
        for gaussian in range(N):
            color = colors[:, gaussian, :, None, None]  # [B, 3, 1, 1]
            grad_result_curr = (grad_result * color).sum(dim=1)  # [B, 3, H, W]

            # Recompute forward values for this Gaussian
            dx = x - position[:, gaussian, 0, None, None]  # [B, H, W]
            dy = y - position[:, gaussian, 1, None, None]  # [B, H, W]

            cov_inv_00_cur = cov_inv_00[:, gaussian]
            cov_inv_01_cur = cov_inv_01[:, gaussian]
            cov_inv_11_cur = cov_inv_11[:, gaussian]

            # Mahalanobis distance
            distance = (
                cov_inv_00_cur * (dx**2)
                + cov_inv_11_cur * (dy**2)
                + cov_inv_01_cur * (2 * dx * dy)
            )

            # Gaussian value (without alpha)
            gaussian_val = torch.exp(-0.5 * distance)  # [B, H, W]

            # Gaussian value with alpha
            gaussian_val_alpha = (
                gaussian_val * alphas[:, gaussian, None, None]
            )  # [B, H, W]
            # print(gaussian_val_alpha.shape, grad_result.shape)
            grad_colors[:, gaussian] = (grad_result * gaussian_val_alpha[:, None]).sum(
                dim=(2, 3)
            )

            # Combined gradient from both grad_result and grad_result_sum
            grad_gaussian = grad_result_curr + grad_result_sum  # [B, H, W]

            # Gradient w.r.t alpha
            # print(grad_gaussian.shape, gaussian_val.shape, alphas.shape)
            grad_alphas[:, gaussian] = (grad_gaussian * gaussian_val).sum(dim=(1, 2))

            # Common factor for spatial and covariance gradients
            common_factor = grad_gaussian * gaussian_val_alpha * (-0.5)  # [B, H, W]

            # Gradients w.r.t distance components
            grad_distance_dx = common_factor * (
                2 * cov_inv_00_cur * dx + 2 * cov_inv_01_cur * dy
            )
            grad_distance_dy = common_factor * (
                2 * cov_inv_11_cur * dy + 2 * cov_inv_01_cur * dx
            )

            # Gradients w.r.t spatial coordinates
            grad_x += grad_distance_dx
            grad_y += grad_distance_dy

            # Gradients w.r.t position (negative of spatial gradients)
            grad_position[:, gaussian, 0] = -grad_distance_dx.sum(dim=(1, 2))
            grad_position[:, gaussian, 1] = -grad_distance_dy.sum(dim=(1, 2))

            # Gradients w.r.t inverse covariance matrix elements
            grad_cov_inv_00[:, gaussian] = (common_factor * (dx**2)).sum(
                dim=(1, 2), keepdim=True
            )
            grad_cov_inv_11[:, gaussian] = (common_factor * (dy**2)).sum(
                dim=(1, 2), keepdim=True
            )
            grad_cov_inv_01[:, gaussian] = (common_factor * (2 * dx * dy)).sum(
                dim=(1, 2), keepdim=True
            )

        # Reshape gradients to match input format
        grad_x = grad_x.unsqueeze(1)  # [B, 1, H, W]
        grad_y = grad_y.unsqueeze(1)  # [B, 1, H, W]

        return (
            grad_x,
            grad_y,
            grad_position,
            grad_cov_inv_00,
            grad_cov_inv_01,
            grad_cov_inv_11,
            grad_alphas,
            grad_colors,
            None,
        )


class GaussianSplatting2DChunkedKernel(autograd.Function):
    """
    A pytorch implementation of potentially faster version of GaussianSplatting2DKernel
    Which splits gaussian into chunks and use matmul to get output of current chunk.
    This approach utilize matmul (tensor core) in einsum(bchw, bnc -> bnhw) part
    while saving the vram since the max intermediate size is B, N_CHUNK_SIZE, H, W
    """

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
        B, _, H, W = x.shape
        N = position.size(1)
        B, N, C = colors.shape
        chunk_size = GaussianSplatting2DChunkedKernel.N_CHUNK_SIZE

        result = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        weight_sum = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
        log_vram("Result Init")
        for gaussian in range(0, N, chunk_size):
            dx = (
                x - position[:, gaussian : gaussian + chunk_size, 0, None, None]
            )  # [B, cn, H, W]
            dy = (
                y - position[:, gaussian : gaussian + chunk_size, 1, None, None]
            )  # [B, cn, H, W]
            weight_block = (
                torch.exp(
                    -0.5
                    * (
                        cov_inv_00[:, gaussian : gaussian + chunk_size] * (dx**2)
                        + cov_inv_11[:, gaussian : gaussian + chunk_size] * (dy**2)
                        + cov_inv_01[:, gaussian : gaussian + chunk_size]
                        * (2 * dx * dy)
                    )
                )
                * alphas[:, gaussian : gaussian + chunk_size, None, None]
            )  # [B, H, W]
            # perform matmul which iterate through N dim
            result += torch.einsum(
                "bnhw, bnc -> bchw",
                weight_block,
                colors[:, gaussian : gaussian + chunk_size],
            )  # [B, C, H, W]
            weight_sum += weight_block.sum(dim=1)

        result /= (weight_sum + eps)[:, None, :, :]

        ctx.save_for_backward(
            x,
            y,
            position,
            cov_inv_00,
            cov_inv_01,
            cov_inv_11,
            alphas,
            colors,
            weight_sum,
        )
        ctx.eps = eps
        return result

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
            colors,
            result_sum,
        ) = ctx.saved_tensors
        eps = ctx.eps
        B, _, H, W = x.shape
        N = position.size(1)
        chunk_size = GaussianSplatting2DChunkedKernel.N_CHUNK_SIZE

        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)
        grad_position = torch.zeros_like(position)
        grad_cov_inv_00 = torch.zeros_like(cov_inv_00)
        grad_cov_inv_01 = torch.zeros_like(cov_inv_01)
        grad_cov_inv_11 = torch.zeros_like(cov_inv_11)
        grad_alphas = torch.zeros_like(alphas)
        grad_colors = torch.zeros_like(colors)

        # grad result is [B, C, H, W] where C=3 so we directly use it
        # if C is large than we should do iterative process just like for g in range(N)
        result_sum += eps
        grad_result_sum = -(grad_result * grad_result).sum(dim=1) / (result_sum**2)
        grad_result /= result_sum[:, None, :, :]

        # Process each Gaussian separately to save memory
        for gaussian in range(0, N, chunk_size):
            color = colors[:, gaussian : gaussian + chunk_size, :]  # [B, cn, 3]
            grad_result_curr = torch.einsum(
                "bchw, bnc -> bnhw", grad_result, color
            )  # [B, cn, H, W]

            # Recompute forward values for this Gaussian
            dx = (
                x - position[:, gaussian : gaussian + chunk_size, 0, None, None]
            )  # [B, cn, H, W]
            dy = (
                y - position[:, gaussian : gaussian + chunk_size, 1, None, None]
            )  # [B, cn, H, W]

            cov_inv_00_cur = cov_inv_00[:, gaussian : gaussian + chunk_size]
            cov_inv_01_cur = cov_inv_01[:, gaussian : gaussian + chunk_size]
            cov_inv_11_cur = cov_inv_11[:, gaussian : gaussian + chunk_size]

            # Mahalanobis distance
            distance = (
                cov_inv_00_cur * (dx**2)
                + cov_inv_11_cur * (dy**2)
                + cov_inv_01_cur * (2 * dx * dy)
            )

            # Gaussian value (without alpha)
            gaussian_val = torch.exp(-0.5 * distance)  # [B, cn, H, W]

            # Gaussian value with alpha
            gaussian_val_alpha = (
                gaussian_val * alphas[:, gaussian : gaussian + chunk_size, None, None]
            )  # [B, H, W]
            grad_colors[:, gaussian : gaussian + chunk_size] = (
                grad_result[:, None] * gaussian_val_alpha[:, :, None]
            ).sum(dim=(3, 4))

            # Combined gradient from both grad_result and grad_result_sum
            grad_gaussian = (
                grad_result_curr + grad_result_sum[:, None, :, :]
            )  # [B, cn, H, W]

            # Gradient w.r.t alpha
            # print(grad_gaussian.shape, gaussian_val.shape, alphas.shape)
            grad_alphas[:, gaussian : gaussian + chunk_size] = (
                grad_gaussian * gaussian_val
            ).sum(dim=(2, 3))

            # Common factor for spatial and covariance gradients
            common_factor = grad_gaussian * gaussian_val_alpha * (-0.5)  # [B, H, W]

            # Gradients w.r.t distance components
            grad_distance_dx = common_factor * (
                2 * cov_inv_00_cur * dx + 2 * cov_inv_01_cur * dy
            )
            grad_distance_dy = common_factor * (
                2 * cov_inv_11_cur * dy + 2 * cov_inv_01_cur * dx
            )

            # Gradients w.r.t spatial coordinates
            grad_x += grad_distance_dx.sum(dim=1, keepdim=True)
            grad_y += grad_distance_dy.sum(dim=1, keepdim=True)

            # Gradients w.r.t position (negative of spatial gradients)
            grad_position[:, gaussian : gaussian + chunk_size, 0] = (
                -grad_distance_dx.sum(dim=(2, 3))
            )
            grad_position[:, gaussian : gaussian + chunk_size, 1] = (
                -grad_distance_dy.sum(dim=(2, 3))
            )

            # Gradients w.r.t inverse covariance matrix elements
            grad_cov_inv_00[:, gaussian : gaussian + chunk_size] = (
                common_factor * (dx**2)
            ).sum(dim=(2, 3), keepdim=True)
            grad_cov_inv_11[:, gaussian : gaussian + chunk_size] = (
                common_factor * (dy**2)
            ).sum(dim=(2, 3), keepdim=True)
            grad_cov_inv_01[:, gaussian : gaussian + chunk_size] = (
                common_factor * (2 * dx * dy)
            ).sum(dim=(2, 3), keepdim=True)

        return (
            grad_x,
            grad_y,
            grad_position,
            grad_cov_inv_00,
            grad_cov_inv_01,
            grad_cov_inv_11,
            grad_alphas,
            grad_colors,
            None,
        )


TorchGaussianSplatting2DKernel = GaussianSplatting2DKernel
TorchGaussianSplatting2DChunked = GaussianSplatting2DChunkedKernel

GaussianSplatting2DKernel = (
    TritonGaussianSplatting2DChunked
    or TritonGaussianSplatting2D
    or GaussianSplatting2DChunkedKernel
    or GaussianSplatting2DKernel
)


class GaussianSplatting2D(nn.Module):
    def __init__(
        self, num_gaussians_per_emb, emb_dim, output_dim=3, pos_map_offset=False
    ):
        super().__init__()
        self.num_gaussians_per_emb = num_gaussians_per_emb
        self.emb_dim = emb_dim
        self.chunk_size = 0
        self.output_dim = output_dim
        self.pos_map_offset = pos_map_offset

        self.proj = nn.Linear(emb_dim, num_gaussians_per_emb * (6 + output_dim))
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

        # positions (2)
        # log_scales (2)
        # rotations (1)
        # logit_opacity (1)
        # colors (output_dim)
        gaussian_bias = torch.randn(num_gaussians_per_emb, 6 + output_dim) * 0.01
        gaussian_bias[:, :2] = 0
        gaussian_bias[:, 2:4] = -4
        self.proj.bias.data.copy_(gaussian_bias.view_as(self.proj.bias))

    @staticmethod
    def xy_grid(size=None, device=None):
        h, w = size
        aspect_ratio = h / w
        # x/y = h/w, x*y = 1
        h_range = aspect_ratio**0.5
        w_range = 1 / aspect_ratio**0.5
        x_grid = (
            torch.linspace(-w_range, w_range, w)
            .view(1, 1, w)
            .expand(1, h, w)
            .to(device)
        )
        y_grid = (
            torch.linspace(-h_range, h_range, h)
            .view(1, h, 1)
            .expand(1, h, w)
            .to(device)
        )
        return x_grid, y_grid

    @staticmethod
    @torch.compile(mode="default", dynamic=True)
    def render(
        positions,
        colors,
        scales,
        rotations,
        alphas,
        x_grid,
        y_grid,
        eps=1e-6,
        fused=True,
    ):
        """
        scales: [B, N, 2]
        alphas: [B, N]
        positions: [B, N, 2]
        rotations: [B, N]
        x_grid: [B, 1, H, W]
        y_grid: [B, 1, H, W]
        """
        b, ng, _ = scales.shape
        _, _, h, w = x_grid.shape

        # Compute rotation matrices [ng, 2, 2]
        cos_rots = torch.cos(rotations)
        sin_rots = torch.sin(rotations)
        rotation_matrices = torch.stack(
            [
                torch.stack([cos_rots, -sin_rots], dim=2),
                torch.stack([sin_rots, cos_rots], dim=2),
            ],
            dim=2,
        )  # [b, ng, 2, 2]
        log_vram("Rotate Mat")

        # Scale matrices [b, ng, 2, 2]
        scale_matrices = torch.zeros(b, ng, 2, 2, device=scales.device)
        scale_matrices[:, :, 0, 0] = scales[:, :, 0]
        scale_matrices[:, :, 1, 1] = scales[:, :, 1]
        log_vram("Scale Mat")

        # Compute covariances [b*ng, 2, 2]
        RS = torch.bmm(rotation_matrices.flatten(0, 1), scale_matrices.flatten(0, 1))
        covariances = torch.bmm(RS, RS.transpose(-2, -1)).unflatten(0, (b, ng))
        log_vram("Convariances")

        # Inverse covariances with regularization [b, ng, 2, 2]
        eye_batch = torch.eye(2, device=covariances.device).expand(b, ng, -1, -1)
        cov_inv_batch = torch.inverse(covariances + eps * eye_batch)
        log_vram("Inverse Cov")

        # Vectorized quadratic form computation [b, ng, H, W]
        # For each gaussian: [dx, dy] @ cov_inv @ [dx, dy]^T
        cov_inv_00 = cov_inv_batch[..., 0, 0, None, None]  # [b, ng, 1, 1]
        cov_inv_01 = cov_inv_batch[..., 0, 1, None, None]  # [b, ng, 1, 1]
        cov_inv_11 = cov_inv_batch[..., 1, 1, None, None]  # [b, ng, 1, 1]
        log_vram("cov inv chunk")

        if fused:
            result = GaussianSplatting2DKernel.apply(
                x_grid,
                y_grid,
                positions,
                cov_inv_00,
                cov_inv_01,
                cov_inv_11,
                alphas,
                colors,
                eps,
            )
            log_vram("fused gaussian")
            return result

        ### THIS IS HEAVY, JUST FOR DEBUGGING
        return naive_gaussian_2d(
            x_grid,
            y_grid,
            positions,
            cov_inv_00,
            cov_inv_01,
            cov_inv_11,
            alphas,
            colors,
            eps,
        )

    def forward(
        self,
        feature,
        size=None,
        grid=None,
        pos_map=None,
        eps=1e-6,
        fused=True,
        scale=None,
    ):
        """
        feature: [B, N, D]
        gs_feat: [B, N, (n*9)] -> [B, N*n, 9]
        grid: optional, [B, 2, H, W]

        colors: [B, N, 3]
        scales: [B, N, 2]
        alphas: [B, N]
        positions: [B, N, 2]
        rotations: [B, N]
        x_grid: [B, H, W]
        y_grid: [B, H, W]
        """

        gs_feature = self.proj(feature).unflatten(-1, (-1, 6+self.output_dim)).flatten(1, 2)
        b, ng, _ = gs_feature.shape

        (
            positions,
            log_scales,
            rotations,
            logit_opacity,
            colors,
        ) = gs_feature.split([2, 2, 1, 1, self.output_dim], dim=-1)

        if grid is None:
            x_grid, y_grid = self.xy_grid(size, feature.device)
            x_grid = x_grid[None].repeat(b, 1, 1, 1)
            y_grid = y_grid[None].repeat(b, 1, 1, 1)
        else:
            # pos_map will put h_pos before w_pos which means y_grid is first
            grid = grid.flip(1)
            x_grid, y_grid = grid.split([1, 1], dim=1)
        if self.pos_map_offset:
            pos_map = torch.concat([x_grid, y_grid], dim=1)
            if scale is not None:
                pos_map = F.interpolate(
                    pos_map,
                    scale_factor=1 / scale,
                    mode="bilinear",
                    align_corners=True,
                )
            pos_map = pos_map.flatten(2, 3).transpose(1, 2)
            assert pos_map.size(1) == feature.size(1)
            pos_map = pos_map[..., None, :].repeat(1, 1, self.num_gaussians_per_emb, 1)
            pos_map = pos_map.flatten(1, 2)
            positions = positions + pos_map

        ## Preprocess
        # Opacity: Not really meaningful in 2DGS for image, skip at first
        # scales: add eps to avoid near 0 scales which cause NaN output
        # rotation: squeeze(-1)
        alphas = torch.ones_like(logit_opacity[..., 0])
        scales = torch.exp(log_scales) + eps
        rotations = rotations[..., 0]
        log_vram("Preprocess")

        output = self.render(
            positions,
            colors,
            scales,
            rotations,
            alphas,
            x_grid,
            y_grid,
            eps,
            fused,
        )
        return output, gs_feature


gaussian_splatting_2d = GaussianSplatting2D.render


if __name__ == "__main__":
    device = "cuda"

    def log_vram(prefix=""):
        if "cuda" in device:
            dev = torch.device(device)
            torch.cuda.synchronize(dev)
            free, total = torch.cuda.mem_get_info(dev)
            mem_used_mb = (total - free) / 1024**2
            print(prefix, mem_used_mb)
        elif "mps" in device:
            mem_used_mb = torch.mps.current_allocated_memory() / 1024**2
            print(prefix, mem_used_mb)
        else:
            mem_used_mb = None
            print(prefix, "N/A")
        return mem_used_mb

    test_gs = GaussianSplatting2D(32, 1024).to(device)
    log_vram("Model Init")

    # shapes:
    B, N, H, W = 3, 4096, 128, 128
    x = torch.rand(B, 1, H, W).to(device).requires_grad_(True)
    y = torch.rand(B, 1, H, W).to(device).requires_grad_(True)
    position = torch.rand(B, N, 2).to(device).requires_grad_(True)
    cov_inv_00 = torch.rand(B, N, 1, 1).to(device).requires_grad_(True)
    cov_inv_01 = torch.rand(B, N, 1, 1).to(device).requires_grad_(True)
    cov_inv_11 = torch.rand(B, N, 1, 1).to(device).requires_grad_(True)
    alphas = torch.rand(B, N).to(device).requires_grad_(True)
    colors = torch.rand(B, N, 3).to(device).requires_grad_(True)

    log_vram("Input")

    torch.cuda.empty_cache()
    log_vram("Naive Gaussian st")
    output1 = naive_gaussian_2d(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    log_vram("Naive Gaussian fwd")
    output1.mean().backward()
    log_vram("Naive Gaussian bwd")
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

    log_vram("Gaussian Fused st")
    output2 = GaussianSplatting2DKernel.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    log_vram("Gaussian Fused fwd")
    output2.mean().backward()
    log_vram("Gaussian Fused bwd")
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

    log_vram("Gaussian Chunked st")
    output3 = GaussianSplatting2DChunkedKernel.apply(
        x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors
    )
    log_vram("Gaussian Chunked fwd")
    output3.mean().backward()
    log_vram("Gaussian Chunked bwd")
    grads3 = (
        x.grad,
        y.grad,
        position.grad,
        cov_inv_00.grad,
        cov_inv_01.grad,
        cov_inv_11.grad,
        alphas.grad,
        colors.grad,
    )

    print("-" * 10)

    print(F.mse_loss(output1, output2), F.l1_loss(output1, output2))
    for g1, g2 in zip(grads, grads2):
        print(F.mse_loss(g1, g2), F.l1_loss(g1, g2))

    print("-" * 10)

    print(F.mse_loss(output1, output3), F.l1_loss(output1, output3))
    for g1, g2 in zip(grads, grads3):
        print(F.mse_loss(g1, g2), F.l1_loss(g1, g2))
