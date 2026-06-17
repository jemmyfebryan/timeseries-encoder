"""
Soft Triplet Loss with continuous similarity weights.

Adaptive margin based on actual distances instead of fixed hard margins.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SoftTripletLoss(nn.Module):
    """
    Soft triplet loss with adaptive margins.

    Instead of fixed margin, uses continuous similarity scores
    to weight the loss adaptively.
    """

    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
        hard_mining: bool = False,
    ):
        """
        Initialize soft triplet loss.

        Args:
            margin: Base margin for triplet loss
            temperature: Temperature for soft weighting
            hard_mining: If True, use hard negative mining
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.hard_mining = hard_mining

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        pos_distances: torch.Tensor = None,
        neg_distances: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute soft triplet loss.

        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
            pos_distances: Optional pre-computed positive distances (B,)
            neg_distances: Optional pre-computed negative distances (B,)

        Returns:
            Triplet loss scalar
        """
        # Compute pairwise distances
        pos_dist = 1 - torch.sum(anchor * positive, dim=1)  # Cosine distance
        neg_dist = 1 - torch.sum(anchor * negative, dim=1)

        # Adaptive margin: smaller margin for closer pairs
        # This preserves continuous relationship
        adaptive_margin = self.margin * (neg_dist - pos_dist)
        adaptive_margin = torch.clamp(adaptive_margin, min=0.1, max=2.0)

        # Triplet loss with adaptive margin
        loss = torch.clamp(pos_dist - neg_dist + adaptive_margin, min=0.0)

        return loss.mean()

    def compute_weighted_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        anchor_returns: torch.Tensor,
        positive_returns: torch.Tensor,
        negative_returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss weighted by actual return distances.

        This preserves the continuous relationship - closer returns
        should have stronger similarity signals.
        """
        batch_size = anchor.size(0)

        # Compute embedding distances
        pos_dist = 1 - torch.sum(anchor * positive, dim=1)
        neg_dist = 1 - torch.sum(anchor * negative, dim=1)

        # Compute target distances (actual return differences)
        pos_target_dist = torch.abs(anchor_returns - positive_returns)
        neg_target_dist = torch.abs(anchor_returns - negative_returns)

        # Weight loss by target distance (closer = higher weight)
        # Normalize by range (approximately)
        pos_weight = 1.0 / (pos_target_dist + 0.1)
        neg_weight = 1.0 / (neg_target_dist + 0.1)

        # Base triplet loss
        base_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        # Apply weights
        weighted_loss = base_loss * pos_weight * neg_weight

        return weighted_loss.mean()


class AdaptiveTripletLoss(nn.Module):
    """
    Triplet loss with adaptive margins based on target distance.

    The margin adapts based on how different the actual targets are:
    - Very different targets → larger margin
    - Similar targets → smaller margin
    """

    def __init__(
        self,
        base_margin: float = 1.0,
        min_margin: float = 0.2,
        max_margin: float = 2.0,
        scale_factor: float = 10.0,
    ):
        """
        Initialize adaptive triplet loss.

        Args:
            base_margin: Base margin for triplet loss
            min_margin: Minimum margin (for very close targets)
            max_margin: Maximum margin (for very different targets)
            scale_factor: How quickly margin adapts with distance
        """
        super().__init__()
        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.scale_factor = scale_factor

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        anchor_targets: torch.Tensor,
        positive_targets: torch.Tensor,
        negative_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive triplet loss.

        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
            anchor_targets: Target values for anchors (B,)
            positive_targets: Target values for positives (B,)
            negative_targets: Target values for negatives (B,)

        Returns:
            Triplet loss scalar
        """
        # Compute embedding distances
        pos_dist = 1 - torch.sum(anchor * positive, dim=1)
        neg_dist = 1 - torch.sum(anchor * negative, dim=1)

        # Compute target distance for adaptive margin
        # If targets are very different, use larger margin
        # If targets are close, use smaller margin
        pos_target_dist = torch.abs(anchor_targets - positive_targets)
        neg_target_dist = torch.abs(anchor_targets - negative_targets)

        # Adaptive margin based on target difference
        # Further apart targets → larger margin
        target_diff = pos_target_dist + neg_target_dist
        adaptive_margin = self.base_margin + torch.tanh(target_diff / self.scale_factor)

        # Clamp margin to reasonable range
        adaptive_margin = torch.clamp(adaptive_margin, self.min_margin, self.max_margin)

        # Triplet loss
        loss = torch.clamp(pos_dist - neg_dist + adaptive_margin, min=0.0)

        return loss.mean()


class ContinuousSimilarityLoss(nn.Module):
    """
    Loss that directly optimizes continuous similarity.

    Instead of margin-based loss, optimizes embeddings to directly
    reflect the continuous similarity of target values.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 0.1,
    ):
        """
        Initialize continuous similarity loss.

        Args:
            alpha: Weight for return similarity
            beta: Weight for volatility similarity
            temperature: Temperature for softening
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
        self,
        anchor_return_emb: torch.Tensor,
        pos_return_emb: torch.Tensor,
        neg_return_emb: torch.Tensor,
        anchor_vol_emb: torch.Tensor,
        pos_vol_emb: torch.Tensor,
        neg_vol_emb: torch.Tensor,
        anchor_return_target: torch.Tensor,
        pos_return_target: torch.Tensor,
        neg_return_target: torch.Tensor,
        anchor_vol_target: torch.Tensor,
        pos_vol_target: torch.Tensor,
        neg_vol_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute continuous similarity loss.

        Args:
            All embeddings and target values for anchor, positive, negative

        Returns:
            Total loss combining return and volatility losses
        """
        # Compute target-based similarities (continuous)
        pos_return_sim = 1.0 - torch.abs(anchor_return_target - pos_return_target)
        neg_return_sim = 1.0 - torch.abs(anchor_return_target - neg_return_target)

        pos_vol_sim = 1.0 - torch.abs(anchor_vol_target - pos_vol_target)
        neg_vol_sim = 1.0 - torch.abs(anchor_vol_target - neg_vol_target)

        # Normalize by temperature
        pos_return_sim = pos_return_sim / self.temperature
        neg_return_sim = neg_return_sim / self.temperature
        pos_vol_sim = pos_vol_sim / self.temperature
        neg_vol_sim = neg_vol_sim / self.temperature

        # Desired: positive sim > negative sim
        return_loss = torch.clamp(
            (1 - torch.sum(anchor_return_emb * pos_return_emb, dim=1)) -
            (1 - torch.sum(anchor_return_emb * neg_return_emb, dim=1)) +
            pos_return_sim - neg_return_sim,
            min=0.0
        )

        vol_loss = torch.clamp(
            (1 - torch.sum(anchor_vol_emb * pos_vol_emb, dim=1)) -
            (1 - torch.sum(anchor_vol_emb * neg_vol_emb, dim=1)) +
            pos_vol_sim - neg_vol_sim,
            min=0.0
        )

        # Combine losses
        total_loss = self.alpha * return_loss + self.beta * vol_loss

        return total_loss
