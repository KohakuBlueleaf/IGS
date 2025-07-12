# IGS: Image Gaussian Splatting

This repository contains some implementation for image gaussian splatting or more specifically: 2D gaussian splatting.

An example result which use 256 features and 64 gaussian per feature to direct match a 512x512 image:
![example result](/images/99.png)
![example result vid](/images/output4.mp4)

## Usage
Install it from source
```bash
pip install git+https://github.com/KohakuBlueleaf/IGS
```

Usage example:
```python
import torch
from igs.gs2d import GaussianSplatting2D, gaussian_splatting_2d

## Module usage
# 32 gaussians per 1024dim feature
# Input: [batch_size, num_features, feature_dim]
# gs_feature: [B, num_features * num_gaussians, 9]
# generated: [B, 3, H, W]
gs = GaussianSplatting2D(32, 1024)
test_x = torch.randn(1, 128, 1024) 
generated, gs_feature = gs(test_x, size=(256, 256))

## Function usage
# generated: [1, 3, 256, 256]
positions = torch.rand(1, 2048, 2) * 2 - 1
colors = torch.rand(1, 2048, 3)
scales = torch.rand(1, 2048, 2) * 0.5 + 0.25
rotations = torch.rand(1, 2048) * 2 * torch.pi
alphas = torch.rand(1, 2048)
x_grid, y_grid = GaussianSplatting2D.xy_grid(size=(256, 256), device=positions.device)
generated = gaussian_splatting_2d(
    positions, colors, scales, rotations, alphas, x_grid, y_grid
)
```

For more detail example, please refer to [this script](/scripts/gs2d_test.py).

## 2DGS implementation

### Intro
In 2DGS, we consider each "sphere" (circle in 2D) as a gaussian distribution. The splatting is performed by summing the gaussian values for each pixel.

The idea is for each gaussian we have:
| Name | Shape | Value Range | Description |
| --- | --- | --- | --- |
| Position | [B, N, 2] | Usually ne [-1, 1], will be larger/smaller for non-square image | The center of the gaussian, also the center of gaussian can be outside the image, but should not be too far |
| Scale | [B, N, 2] | Usually be (0,1], can be (0, inf), but should not be too large | The radius of the gaussian (On each axis, we have 2 since we are 2DGS) |
| rotations | [B, N] or [B, N, 1] | [0, 2pi], but since we use cos/sin, (-inf, inf) is ok | The rotation of the gaussian |
| colors | [B, N, 3] or [B, N, C] | Depends on your target | The "color" of each gaussian |
| opacity | [B, N] or [B, N, 1] | [0, 1] | The opacity of each gaussian |

And for output image, we use `pos_grid - position` as distance from center, than use covariance matrix obtained from `scale` and `rotation` to compute the weight of corresponding color.

And the weight will be multiplied with `opacity` and summed up for each pixel. (With normalization to ensure the sum is 1)

### Naive Implementation
The overall implementation can be look like:
```python
def naive_gaussian_2d(
    x, y, position, cov_inv_00, cov_inv_01, cov_inv_11, alphas, colors, eps=1e-6
):
    """
    x: [B, 1, H, W] -> axis on w
    y: [B, 1, H, W] -> axis on h

    position: [B, N, 2]
    cov_inv_00: [B, N, 1]
    cov_inv_01: [B, N, 1]
    cov_inv_11: [B, N, 1]
    alphas: [B, N]
    colors: [B, N, C]

    weight: [B, N, H, W]
    result: [B, C, H, W]

    result = einsum("bnhw, bnc -> bchw", weight, colors)
    """
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
```

The problem of above implementation is, it will consume ***TONS*** of vram.
Since the shape of weight is [B, N, H, W], and if we use some common numbers, like batch size 16, 8192 gaussians for 256x256 image.
It will consume 16 * 8192 * 256 * 256 * 2or4 bytes = 16GiB of vram.
And this is only for a single intermediate state, you will need more same size state for autograd backward.

### Chunked Implementation
The idea of chunked or iterative implementation is, we can process a small amount or even just one gaussian at a time.

Which means we do a for loop to go through all the gaussians, when we get the weight matrix for current gaussians, we directly calculate its corresponding color output.

Then we sum up all the color outputs together and divide by the weight_sum to get the final result.

This means the largest intermedate state is [B, C, H, W] which is directly the final result, so it will consume less vram.

But if you do this directly, the autograd will still cache [B, N, H, W] state for autograd bwd, therefore we will need to implement the bwd by ourselves.

You can check [the implementation](/src/igs/gs2d.py) for more details.

### Triton Kernel
Since for chunked/iterative implementation we will need to use for-loops, this makes the whole implementation become slow.

Therefore we did a triton kernel which basically convert the custom autograd function directly into a kernel.
And we use a 2d triton grid on (batch_size, num_gaussians) so the operation inside the kernel match our custom autograd function.

The implementation is [here](/src/igs/gs_triton.py)