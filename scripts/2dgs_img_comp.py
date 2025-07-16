import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as trnsF
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from torchao.float8.float8_utils import amax_to_scale, tensor_to_scale, to_fp8_saturated
from pytorch_msssim import ms_ssim, ssim
from PIL import Image
from tqdm import trange
from lpips import LPIPS
from PIL import Image

from anyschedule import AnySchedule
from igs.gs2d import GaussianSplatting2D


torch.set_float32_matmul_precision("medium")
DEVICE = "cuda:0"
EPS = 1e-6


def quant(x, target_dtype=torch.float8_e4m3fn):
    if target_dtype in [torch.int8]:
        max_value = torch.iinfo(target_dtype).max
    else:
        max_value = torch.finfo(target_dtype).max
    if x.ndim == 3:
        amax = torch.max(torch.abs(x), dim=1, keepdim=True)[0].to(torch.float64)
        scale = max_value / torch.clamp(amax, min=1e-12)
    else:
        amax = torch.max(torch.abs(x)).to(torch.float64)
        scale = max_value / torch.clamp(amax, min=1e-12)
    x_scaled = x.to(scale) * scale
    x_quant = x_scaled.clamp(-max_value, max_value).to(target_dtype)
    return x_quant, scale


def dequant(x, scale):
    x = x.to(torch.float64)
    scale = scale.to(torch.float64)
    return (x / scale).float()


class ProgressiveGaussianSplatter(nn.Module):
    def __init__(self, num_gaussians, image=None):
        super().__init__()
        self.num_gaussians = num_gaussians

        position_grid_size = int(num_gaussians**0.5)
        position_grid_h = torch.linspace(-1, 1, position_grid_size)[:, None].repeat(
            1, position_grid_size
        )
        position_grid_v = torch.linspace(-1, 1, position_grid_size)[None, :].repeat(
            position_grid_size, 1
        )
        position_grid = torch.stack([position_grid_v, position_grid_h], dim=-1).view(
            -1, 2
        )
        position_grid_addon = torch.linspace(
            -1 + EPS, 1 - EPS, (num_gaussians - position_grid.size(0)) // 2
        )[:, None]
        addon_h = (
            torch.concat(
                [position_grid_addon, torch.zeros_like(position_grid_addon)], dim=1
            )
            + EPS
        )
        addon_v = (
            torch.concat(
                [torch.zeros_like(position_grid_addon), position_grid_addon], dim=1
            )
            - EPS
        )
        position_grid = torch.concat([position_grid, addon_h, addon_v], dim=0)
        position_grid = torch.concat(
            [position_grid, torch.zeros(num_gaussians - position_grid.shape[0], 2)],
            dim=0,
        )

        if image is not None:
            raw_colors = (
                F.interpolate(
                    image,
                    (position_grid_size, position_grid_size),
                    mode="bicubic",
                    antialias=True,
                )
                .flatten(2, 3)
                .permute(0, 2, 1)
            )
            raw_colors = torch.concat(
                [
                    raw_colors,
                    torch.zeros(1, num_gaussians - raw_colors.shape[1], 3).to(image),
                ],
                dim=1,
            )
        else:
            raw_colors = torch.rand(1, num_gaussians, 3)

        # Initialize Gaussian parameters
        self.position = nn.Parameter(position_grid[None])
        self.log_scale = nn.Parameter(torch.zeros(1, num_gaussians, 2) - 5)
        self.rotation = nn.Parameter(torch.zeros(1, num_gaussians))
        self.color = nn.Parameter(raw_colors + 1)
        self.register_buffer("alphas", torch.ones(1, num_gaussians))

    def forward(self, num_active_features=None, size=None):
        if num_active_features is None:
            num_active_features = self.num_gaussians
        else:
            num_active_features = min(num_active_features, self.num_gaussians)
        x_grid, y_grid = GaussianSplatting2D.xy_grid(size, self.position.device)
        x_grid = x_grid[None]
        y_grid = y_grid[None]
        return (
            GaussianSplatting2D.render(
                self.position[:, :num_active_features],
                self.color[:, :num_active_features],
                torch.exp(self.log_scale[:, :num_active_features]) + EPS,
                self.rotation[:, :num_active_features],
                self.alphas[:, :num_active_features],
                x_grid,
                y_grid,
            )
            - 1
        )


lpips = LPIPS(net="alex").to(DEVICE).requires_grad_(False).eval()
lpips2 = LPIPS(net="vgg").to(DEVICE).requires_grad_(False).eval()


def gaussian_splatting_loss(rendered_image, target_image):
    """Combined L1 + L2 loss"""
    l1 = F.l1_loss(rendered_image, target_image)
    l2 = F.mse_loss(rendered_image, target_image)
    msssim = 1 - ms_ssim(
        rendered_image, target_image, data_range=1.0, win_size=7, size_average=True
    )
    ssim_loss = 1 - ssim(
        rendered_image, target_image, data_range=1.0,size_average=True
    )
    lpips_loss1 = lpips(rendered_image, target_image, normalize=True)
    lpips_loss2 = lpips2(rendered_image, target_image, normalize=True)
    return l1 + l2 * 0.5 + ssim_loss * 0.5 + msssim * 0.3 + lpips_loss1 * 0.1 + lpips_loss2 * 0.1


def psnr(image1, image2):
    return 20 * torch.log10(1.0 / torch.mean((image1 - image2) ** 2))


def progressive_training():
    # Parameters
    # 256px -> 512px -> 1024px, 800/400/200 step
    height, width = 256, 256
    scales = [1, 2, 4]
    steps = [1400, 400, 200]
    # scales = [1]
    # steps = [1000]
    max_height, max_width = max(scales) * height, max(scales) * width
    num_gaussians = 8192
    log_period = 20
    lr = 0.05 * 4096 / num_gaussians

    # Create target image
    # target_image = create_target_image(height, width)
    image = Image.open("./data/target-real-sq.jpg").convert("RGB")
    target_image_base = (
        trnsF.to_tensor(
            trnsF.center_crop(
                trnsF.resize(
                    image,
                    max(max_height, max_width),
                    interpolation=trnsF.InterpolationMode.BOX,
                    antialias=True,
                ),
                (max_height, max_width),
            )
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Initialize splatter
    splatter = ProgressiveGaussianSplatter(num_gaussians, target_image_base).to(DEVICE)
    optimizer = torch.optim.AdamW(
        splatter.parameters(),
        lr=lr,
        eps=EPS,
        betas=(0.9, 0.995),
        weight_decay=0.0001,
        # maximize=True,
    )
    scheduler = AnySchedule(
        optimizer,
        config={
            "lr": {
                "mode": "cosine",
                "end": sum(steps) + 1,
                "min_value": 0.1,
                "warmup": 0,
            }
        },
    )

    # Training with progressive k selection
    losses = []
    prev_step = 0
    for img_scale, step in zip(scales, steps):
        h, w = int(height * img_scale), int(width * img_scale)
        target_image = F.interpolate(
            target_image_base,
            size=(h, w),
            mode="bicubic",
            antialias=True,
        )
        for s in (pbar := trange(prev_step, prev_step + step, smoothing=0.01)):
            rendered_image = splatter(size=(h, w))
            recon_loss = gaussian_splatting_loss(
                rendered_image,
                target_image,
            )
            # recon_loss = ssim(
            #     rendered_image,
            #     target_image,
            #     data_range=1.0,
            #     size_average=True,
            # )

            ## regularization
            position_penalty = torch.mean(F.relu(splatter.position.abs() - 1.0))
            reg_loss = position_penalty

            loss = recon_loss - reg_loss * 0.01
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix(
                loss=loss.item(),
            )
            if s % log_period == 0:
                coarse_to_fine_visualization(
                    splatter, target_image, file_name=f"log_images2/{s//log_period}"
                )
        prev_step += step

    coarse_to_fine_visualization(
        splatter,
        target_image_base,
        file_name=f"log_images2/{(prev_step//log_period) + int(prev_step%log_period > 0)}",
    )

    return splatter, target_image_base, losses


@torch.no_grad()
def coarse_to_fine_visualization(model, target_image, file_name=None):
    """
    Visualize coarse-to-fine AR generation
    """
    # clone splatter to avoid affect training
    splatter = ProgressiveGaussianSplatter(model.num_gaussians).to(DEVICE).eval()
    splatter.load_state_dict(model.state_dict())
    fractions = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
    num_gaussians = splatter.num_gaussians
    height, width = target_image.shape[2:]

    # Calculate number of Gaussians for each fraction
    features_counts = [max(1, int(frac * num_gaussians)) for frac in fractions]

    # Generate images for each level
    with torch.no_grad():
        rendered_images = []
        for count in features_counts:
            rendered = splatter(num_active_features=count, size=(height, width)).cpu()
            rendered_images.append(rendered)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Target image
    img = target_image[0].cpu().permute(1, 2, 0).clamp(0, 1)
    h, w = img.shape[:2]
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Target Image ({w}x{h})")
    axes[0, 0].axis("off")

    # Progressive generations
    titles = ["1/16", "1/8", "1/4", "1/2", "Full"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for i, (rendered, title, pos) in enumerate(zip(rendered_images, titles, positions)):
        row, col = pos
        axes[row, col].imshow(rendered[0].permute(1, 2, 0).clamp(0, 1))
        axes[row, col].set_title(f"{title}\n(First {features_counts[i]} Gaussians)")
        axes[row, col].axis("off")

    plt.tight_layout()
    file_name = file_name or "output/coarse_to_fine"
    plt.savefig(f"{file_name}.png", dpi=200)
    plt.close()

    return rendered_images, features_counts


@torch.no_grad()
def save_tensor_image(tensor, file_name, **kwargs):
    tensor = tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().detach()
    img = Image.fromarray((tensor * 255).numpy().astype(np.uint8))
    img.save(file_name, **kwargs)


def main():
    pl.seed_everything(0)
    """Main training and visualization"""
    print("=== Progressive Gaussian Splatting Training ===")

    # Run progressive training
    splatter, target_image, losses = progressive_training()

    eval(splatter, target_image)

    torch.save(splatter.state_dict(), "output/features.pt")


def sim_to_db(sim_val):
    return 10 * math.log10(1/(1-sim_val))


@torch.no_grad()
def eval(splatter: ProgressiveGaussianSplatter, target_image: torch.Tensor):
    save_tensor_image(target_image, "output/target.png")
    print("\n=== Statistics ===")
    b, c, h, w = target_image.shape
    pixel_count = h * w
    gaussian_per_pixel = splatter.num_gaussians / pixel_count
    ppp = gaussian_per_pixel * 8
    bpp = ppp * 8
    print(f"Gaussians per pixel: {gaussian_per_pixel:.2f}")
    print(f"Parameters per pixel: {ppp:.2f}")

    root_state_dict = {k: v.clone().cpu() for k, v in splatter.state_dict().items()}

    print("\n=== Evaluation ===")
    recon = splatter(size=(h, w)).clamp(0, 1)
    psnr_val = psnr(recon, target_image).item()
    ssim_val = ssim(recon, target_image, data_range=1, size_average=True).item()
    msssim_val = ms_ssim(recon, target_image, data_range=1, size_average=True).item()
    lpips_alex = 1 - lpips(recon, target_image, normalize=True).item()
    lpips_vgg = 1 - lpips2(recon, target_image, normalize=True).item()
    print(f"PSNR: {psnr_val:5.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM(db): {sim_to_db(ssim_val):.2f}")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"MS-SSIM(db): {sim_to_db(msssim_val):.2f}")
    print(f"LPIPS (AlexNet): {lpips_alex:.4f}")
    print(f"LPIPS (VGG): {lpips_vgg:.4f}")
    save_tensor_image(recon, "output/recon.png")

    print("\n=== Evaluation (16bit) ===")
    splatter.load_state_dict(
        {k: dequant(*quant(v, torch.float16)) for k, v in root_state_dict.items()}
    )
    b, c, h, w = target_image.shape
    recon = splatter(size=(h, w)).clamp(0, 1)
    psnr_val = psnr(recon, target_image).item()
    ssim_val = ssim(recon, target_image, data_range=1, size_average=True).item()
    msssim_val = ms_ssim(recon, target_image, data_range=1, size_average=True).item()
    lpips_alex = 1 - lpips(recon, target_image, normalize=True).item()
    lpips_vgg = 1 - lpips2(recon, target_image, normalize=True).item()
    print(f"PSNR: {psnr_val:5.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM(db): {sim_to_db(ssim_val):.2f}")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"MS-SSIM(db): {sim_to_db(msssim_val):.2f}")
    print(f"LPIPS (AlexNet): {lpips_alex:.4f}")
    print(f"LPIPS (VGG): {lpips_vgg:.4f}")
    save_tensor_image(recon, "output/recon_fp16.png")

    print("\n=== Evaluation (12bit) ===")
    state_dict = {
        k: dequant(*quant(v, torch.int8)) for k, v in splatter.state_dict().items()
    }
    state_dict["position"] = root_state_dict["position"]
    state_dict["log_scale"] = root_state_dict["log_scale"]
    splatter.load_state_dict(state_dict)
    b, c, h, w = target_image.shape
    recon = splatter(size=(h, w)).clamp(0, 1)
    psnr_val = psnr(recon, target_image).item()
    ssim_val = ssim(recon, target_image, data_range=1, size_average=True).item()
    msssim_val = ms_ssim(recon, target_image, data_range=1, size_average=True).item()
    lpips_alex = 1 - lpips(recon, target_image, normalize=True).item()
    lpips_vgg = 1 - lpips2(recon, target_image, normalize=True).item()
    print(f"PSNR: {psnr_val:5.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM(db): {sim_to_db(ssim_val):.2f}")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"MS-SSIM(db): {sim_to_db(msssim_val):.2f}")
    print(f"LPIPS (AlexNet): {lpips_alex:.4f}")
    print(f"LPIPS (VGG): {lpips_vgg:.4f}")
    save_tensor_image(recon, "output/recon_12bit.png")

    print("\n=== Evaluation (10bit) ===")
    state_dict = {
        k: dequant(*quant(v, torch.int8)) for k, v in splatter.state_dict().items()
    }
    state_dict["position"] = root_state_dict["position"]
    splatter.load_state_dict(state_dict)
    b, c, h, w = target_image.shape
    recon = splatter(size=(h, w)).clamp(0, 1)
    psnr_val = psnr(recon, target_image).item()
    ssim_val = ssim(recon, target_image, data_range=1, size_average=True).item()
    msssim_val = ms_ssim(recon, target_image, data_range=1, size_average=True).item()
    lpips_alex = 1 - lpips(recon, target_image, normalize=True).item()
    lpips_vgg = 1 - lpips2(recon, target_image, normalize=True).item()
    print(f"PSNR: {psnr_val:5.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM(db): {sim_to_db(ssim_val):.2f}")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"MS-SSIM(db): {sim_to_db(msssim_val):.2f}")
    print(f"LPIPS (AlexNet): {lpips_alex:.4f}")
    print(f"LPIPS (VGG): {lpips_vgg:.4f}")
    save_tensor_image(recon, "output/recon_10bit.png")

    print("\n=== Evaluation (int8) ===")
    splatter.load_state_dict(
        {k: dequant(*quant(v, torch.int8)) for k, v in root_state_dict.items()}
    )
    b, c, h, w = target_image.shape
    recon = splatter(size=(h, w)).clamp(0, 1)
    psnr_val = psnr(recon, target_image).item()
    ssim_val = ssim(recon, target_image, data_range=1, size_average=True).item()
    msssim_val = ms_ssim(recon, target_image, data_range=1, size_average=True).item()
    lpips_alex = 1 - lpips(recon, target_image, normalize=True).item()
    lpips_vgg = 1 - lpips2(recon, target_image, normalize=True).item()
    print(f"PSNR: {psnr_val:5.2f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"SSIM(db): {sim_to_db(ssim_val):.2f}")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"MS-SSIM(db): {sim_to_db(msssim_val):.2f}")
    print(f"LPIPS (AlexNet): {lpips_alex:.4f}")
    print(f"LPIPS (VGG): {lpips_vgg:.4f}")
    save_tensor_image(recon, "output/recon_int8.png")


if __name__ == "__main__":
    main()
