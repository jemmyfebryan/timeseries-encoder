"""
Soft triplet mining with continuous similarity labels.

Preserves continuous relationships instead of hard binary binning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class SoftTripletDataset(Dataset):
    """
    Triplet dataset with soft similarity labels.

    Uses continuous similarity scores instead of hard bin membership,
    preserving ordinal relationships in the data.
    """

    def __init__(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        n_triplets: int = 10000,
        temperature: float = 0.1,
    ):
        """
        Initialize SoftTripletDataset.

        Args:
            features: Feature windows (N, lookback, C)
            return_targets: Future returns (N,)
            volatility_targets: Future volatility (N,)
            asset_ids: Asset IDs (N,)
            n_triplets: Number of triplets to generate
            temperature: Temperature for soft weighting
        """
        self.features = torch.from_numpy(features).float()
        self.return_targets = torch.from_numpy(return_targets).float()
        self.volatility_targets = torch.from_numpy(volatility_targets).float()
        self.asset_ids = asset_ids
        self.n_triplets = n_triplets
        self.temperature = temperature

        # Pre-compute statistics for normalization
        self.return_min = return_targets.min()
        self.return_max = return_targets.max()
        self.return_range = self.return_max - self.return_min

        self.vol_min = volatility_targets.min()
        self.vol_max = volatility_targets.max()
        self.vol_range = self.vol_max - self.vol_min

        # Generate soft triplets
        logger.info("Mining soft triplets with continuous similarity...")
        self.anchor_idx, self.positive_idx, self.negative_idx = self._mine_soft_triplets(
            features, return_targets, volatility_targets, asset_ids, n_triplets
        )
        logger.info(f"Mined {len(self.anchor_idx)} soft triplets")

        # Pre-compute soft similarity weights
        self.return_sim_weights = self._compute_similarity_weights(
            return_targets[self.anchor_idx],
            return_targets[self.positive_idx],
            return_targets[self.negative_idx],
        )

        self.vol_sim_weights = self._compute_similarity_weights(
            volatility_targets[self.anchor_idx],
            volatility_targets[self.positive_idx],
            volatility_targets[self.negative_idx],
        )

    def _mine_soft_triplets(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        n_triplets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mine triplets with soft continuous similarity."""
        n_samples = len(return_targets)

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        # Use coarse bins for efficiency (but soft labels for similarity)
        num_bins = 20

        # Create bins
        return_bins = self._create_bins(return_targets, num_bins)
        vol_bins = self._create_bins(volatility_targets, num_bins)

        # Convert to numpy arrays for faster operations
        return_bins_array = np.array(return_bins)
        vol_bins_array = np.array(vol_bins)
        asset_ids_array = np.array(asset_ids)

        # Build bin groups
        return_bin_groups: Dict[int, np.ndarray] = {}
        vol_bin_groups: Dict[int, np.ndarray] = {}

        unique_bins = np.unique(return_bins_array)
        for bin_id in unique_bins:
            mask = return_bins_array == bin_id
            return_bin_groups[bin_id] = np.where(mask)[0]

        unique_bins = np.unique(vol_bins_array)
        for bin_id in unique_bins:
            mask = vol_bins_array == bin_id
            vol_bin_groups[bin_id] = np.where(mask)[0]

        return_bin_keys = list(return_bin_groups.keys())
        vol_bin_keys = list(vol_bin_groups.keys())

        # Log progress
        log_interval = max(1, n_triplets // 10)

        for i in range(n_triplets):
            if i % log_interval == 0 and i > 0:
                logger.info(f"  Mining progress: {i}/{n_triplets} triplets")

            # Random anchor
            anchor_idx = np.random.randint(0, n_samples)

            # Choose which target to use
            if np.random.rand() < 0.5:
                # Use return similarity
                anchor_bin = return_bins[anchor_idx]
                same_bin_samples = return_bin_groups.get(anchor_bin, np.array([]))

                # Get different bin for negative
                different_bins = [b for b in return_bin_keys if b != anchor_bin]
            else:
                # Use volatility similarity
                anchor_bin = vol_bins[anchor_idx]
                same_bin_samples = vol_bin_groups.get(anchor_bin, np.array([]))

                # Get different bin for negative
                different_bins = [b for b in vol_bin_keys if b != anchor_bin]

            if len(same_bin_samples) < 2 or len(different_bins) == 0:
                continue

            # Sample positive from same bin
            mask = (same_bin_samples != anchor_idx) & (asset_ids_array[same_bin_samples] != asset_ids_array[anchor_idx])
            valid_positives = same_bin_samples[mask]

            if len(valid_positives) == 0:
                valid_positives = same_bin_samples[same_bin_samples != anchor_idx]

            if len(valid_positives) == 0:
                continue

            positive_idx = np.random.choice(valid_positives)

            # Sample negative from different bin
            neg_bin = np.random.choice(different_bins)

            if anchor_bin in return_bin_groups:
                different_bin_samples = return_bin_groups[neg_bin]
            else:
                different_bin_samples = vol_bin_groups[neg_bin]

            mask = asset_ids_array[different_bin_samples] != asset_ids_array[anchor_idx]
            valid_negatives = different_bin_samples[mask]

            if len(valid_negatives) == 0:
                valid_negatives = different_bin_samples

            if len(valid_negatives) == 0:
                continue

            negative_idx = np.random.choice(valid_negatives)

            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)

        return (
            np.array(anchor_indices),
            np.array(positive_indices),
            np.array(negative_indices),
        )

    def _create_bins(self, targets: np.ndarray, num_bins: int) -> np.ndarray:
        """Create coarse bins for efficient sampling."""
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(targets, quantiles)
        bin_indices = np.digitize(targets, bin_edges[1:-1], right=False)
        return bin_indices

    def _compute_similarity_weights(
        self,
        anchor_targets: np.ndarray,
        positive_targets: np.ndarray,
        negative_targets: np.ndarray,
    ) -> np.ndarray:
        """
        Compute soft similarity weights based on actual distance.

        Args:
            anchor_targets: Target values for anchors
            positive_targets: Target values for positives
            negative_targets: Target values for negatives

        Returns:
            Similarity weights: (N, 2) where [:, 0] is pos_sim, [:, 1] is neg_sim
        """
        # Compute positive similarity: closer = more similar
        pos_dist = np.abs(anchor_targets - positive_targets)
        pos_sim = 1.0 - (pos_dist / (self.return_range + 1e-8))

        # Compute negative similarity: further = more dissimilar
        neg_dist = np.abs(anchor_targets - negative_targets)
        neg_sim = neg_dist / (self.return_range + 1e-8)  # Higher = more different

        # Normalize
        weights = np.stack([pos_sim, neg_sim], axis=1)

        # Apply temperature scaling
        weights = weights / self.temperature

        return weights

    def __len__(self) -> int:
        """Return number of triplets."""
        return len(self.anchor_idx)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single soft triplet."""
        anchor_i = self.anchor_idx[idx]
        positive_i = self.positive_idx[idx]
        negative_i = self.negative_idx[idx]

        return (
            self.features[anchor_i],
            self.features[positive_i],
            self.features[negative_i],
            self.return_targets[anchor_i],
            self.return_targets[positive_i],
            self.return_targets[negative_i],
            self.volatility_targets[anchor_i],
            self.volatility_targets[positive_i],
            self.volatility_targets[negative_i],
            self.asset_ids[anchor_i],
            self.asset_ids[positive_i],
            self.asset_ids[negative_i],
        )
