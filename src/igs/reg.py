import torch
import torch.nn.functional as F


@torch.compile(mode="default", dynamic=True)
def koleo_diversity_loss(features):
    """
    KoLeo regularization: Encourages uniform distribution by penalizing
    close nearest neighbors. Much cheaper than full pairwise distances.

    Args:
        features: [B, N, D] tensor of features to regularize
        eps: small constant for numerical stability

    Returns:
        loss: scalar diversity loss
    """
    N = features.shape[1]
    if N <= 1:
        return torch.tensor(0.0, device=features.device)

    # Normalize features to unit sphere
    features_norm = F.normalize(features, p=2, dim=-1)  # [N, D]

    # Compute pairwise cosine similarities
    similarities = torch.bmm(features_norm, features_norm.permute(0, 2, 1))  # [B, N, N]

    # Set diagonal to -inf to exclude self-similarity
    similarities.diagonal(dim1=-2, dim2=-1).fill_(-float("inf"))

    # Find nearest neighbor similarities (highest cosine similarity)
    nearest_similarities, _ = torch.max(similarities, dim=-1)  # [B, N]

    # KoLeo loss: penalize high nearest neighbor similarities
    # We want nearest neighbors to be far apart (low similarity)
    diversity_loss = torch.mean(
        F.relu(nearest_similarities + 1.0)
    )  # ReLU(sim + 1) since sim ∈ [-1, 1]

    return diversity_loss
