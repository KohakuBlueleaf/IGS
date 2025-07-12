import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import torchvision.transforms.functional as trnsF
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import random

from pytorch_msssim import ssim, ms_ssim
from PIL import Image
from tqdm import trange
from lpips import LPIPS

from anyschedule import AnySchedule
from anyschedule.utils import get_scheduler

from reg import koleo_diversity_loss
from igs.gs2d import GaussianSplatting2D


torch.set_float32_matmul_precision("medium")
DEVICE = "cuda:0"
EPS = 1e-6


class ProgressiveGaussianSplatter(nn.Module):
    def __init__(self, num_features, feature_dim, gaussians_per_feature):
        super().__init__()
        self.num_features = num_features
        self.gaussians_per_feature = gaussians_per_feature

        # Initialize Gaussian parameters
        self.feature = nn.Parameter(torch.randn(1, num_features, feature_dim))
        self.gs = GaussianSplatting2D(gaussians_per_feature, feature_dim)

    def forward(self, num_active_features=None, size=None):
        """
        Fully vectorized rendering - no for loops!
        """
        if num_active_features is None:
            num_active_features = self.num_features
        else:
            num_active_features = min(num_active_features, self.num_features)
        feature = self.feature[:, :num_active_features]

        return self.gs(feature, size)


def gaussian_splatting_loss(rendered_image, target_image):
    """Combined L1 + L2 loss"""
    l1 = F.l1_loss(rendered_image, target_image)
    l2 = F.mse_loss(rendered_image, target_image)
    msssim_loss = 1 - ms_ssim(
        rendered_image, target_image, data_range=1.0, size_average=True
    )
    return l1 + l2 * 0.5 + msssim_loss * 0.5


def progressive_training():
    """
    Progressive training with random k selection
    """
    # Parameters
    height, width = 512, 512
    num_features = 128
    feature_dim = 1024
    num_gaussians_per_feature = 64
    chunk_size = 4
    steps = 10000
    grad_acc = 1
    log_period = 100
    lr = 5e-5

    # Create target image
    # target_image = create_target_image(height, width)
    image = Image.open("./data/target.png").convert("RGB")
    target_image = (
        trnsF.to_tensor(
            trnsF.center_crop(
                trnsF.resize(
                    image,
                    max(height, width),
                    interpolation=trnsF.InterpolationMode.BOX,
                    antialias=True,
                ),
                (height, width),
            )
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Initialize splatter
    splatter = ProgressiveGaussianSplatter(
        num_features, feature_dim, num_gaussians_per_feature
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        splatter.parameters(), lr=lr, eps=EPS, betas=(0.9, 0.98), weight_decay=0.005
    )
    scheduler = AnySchedule(
        optimizer,
        config={
            "lr": {
                # WSD scheduler
                "mode": "composer",
                "schedules": [
                    {
                        "mode": "constant",
                        "end": int(steps // grad_acc * 0.4),
                        "warmup": int(steps // grad_acc * 0.05),
                    },
                    {
                        "mode": "cosine",
                        "end": steps // grad_acc,
                        "min_value": 0.05,
                    },
                ],
                "min_value": 0.05,
            }
        },
    )
    diversity_loss_weight_scheduler = get_scheduler(
        config={
            "mode": "composer",
            "schedules": [
                {
                    "mode": "cosine",
                    "end": int(steps * 0.5),
                    "value": 0.1,
                    "min_value": 0.01,
                }
            ],
            "end": steps + 1,
            "min_value": 0.01,
            "value": 0.01,
        }
    )

    print("Warming up...")
    rendered, gs_feature = splatter(
        num_active_features=num_features, size=(height, width)
    )
    recon_loss = gaussian_splatting_loss(rendered, target_image)
    reg_loss = koleo_diversity_loss(gs_feature[0], eps=EPS)
    (recon_loss + reg_loss).backward()
    optimizer.zero_grad()
    print("Done!")

    # Training with progressive k selection
    losses = []

    for step in (pbar := trange(steps, desc="Progressive training", smoothing=0.01)):
        ratio = 1 - torch.rand(1).item() ** 1.5
        # ratio = random.choice([1/16, 1/8, 1/4, 1/2, 1.0])
        # ratio = step/steps
        k = int(round((ratio * num_features / chunk_size)) * chunk_size)
        if k > num_features:
            k = num_features
        elif k < chunk_size:
            k = chunk_size

        # Render with only first k Gaussians
        rendered_image, gs_feature = splatter(
            num_active_features=k, size=(height, width)
        )
        recon_loss = gaussian_splatting_loss(
            rendered_image,
            target_image,
        )
        reg_loss = koleo_diversity_loss(
            splatter.feature,
        )
        loss = recon_loss
        loss = loss + reg_loss * diversity_loss_weight_scheduler(step)
        loss = loss / grad_acc
        loss.backward()
        if step % grad_acc == 0:
            # do gradient clipping
            torch.nn.utils.clip_grad_norm_(splatter.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(
            recon_loss=recon_loss.item(),
            reg_loss=reg_loss.item(),
            k=k,
        )
        if step % log_period == 0:
            coarse_to_fine_visualization(
                splatter, target_image, file_name=f"log_images2/{step//log_period}"
            )
    coarse_to_fine_visualization(
        splatter,
        target_image,
        file_name=f"log_images2/{(steps-1//log_period) + ((steps-1)%log_period > 0)}",
    )

    return splatter, target_image, losses


def coarse_to_fine_visualization(splatter, target_image, file_name=None):
    """
    Visualize coarse-to-fine AR generation
    """
    fractions = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
    num_features = splatter.num_features
    height, width = target_image.shape[2:]

    # Calculate number of Gaussians for each fraction
    features_counts = [max(1, int(frac * num_features)) for frac in fractions]

    # Generate images for each level
    with torch.no_grad():
        rendered_images = []
        for count in features_counts:
            rendered = splatter(num_active_features=count, size=(height, width))[
                0
            ].cpu()
            rendered_images.append(rendered)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Target image
    axes[0, 0].imshow(target_image.cpu()[0].permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title("Target Image")
    axes[0, 0].axis("off")

    # Progressive generations
    titles = ["1/16", "1/8", "1/4", "1/2", "Full"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for i, (rendered, title, pos) in enumerate(zip(rendered_images, titles, positions)):
        row, col = pos
        axes[row, col].imshow(rendered[0].permute(1, 2, 0).clamp(0, 1))
        axes[row, col].set_title(f"{title}\n(First {features_counts[i]} Features)")
        axes[row, col].axis("off")

    plt.tight_layout()
    file_name = file_name or "coarse_to_fine"
    plt.savefig(f"{file_name}.png", dpi=200)
    plt.close()

    return rendered_images, features_counts


def main():
    pl.seed_everything(random.randint(0, 2**31 - 1))
    """Main training and visualization"""
    print("=== Progressive Gaussian Splatting Training ===")

    # Run progressive training
    splatter, target_image, losses = progressive_training()

    print("\n=== Coarse-to-Fine AR Visualization ===")

    # Show coarse-to-fine generation
    rendered_images, gaussian_counts = coarse_to_fine_visualization(
        splatter, target_image
    )

    return splatter, target_image, rendered_images


if __name__ == "__main__":
    main()
