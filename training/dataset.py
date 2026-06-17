"""
Triplet dataset module.

Implements efficient triplet sampling for Siamese training.
Supports multiple strategies: binning, random, and hard negative mining.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List
import logging

from .config import TripletConfig

logger = logging.getLogger(__name__)


class TripletMiningStrategy:
    """Base class for triplet mining strategies."""

    def __init__(self, config: TripletConfig):
        self.config = config

    def mine_triplets(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        n_triplets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mine triplets from the data.

        Args:
            features: Feature windows (N, lookback, 1)
            return_targets: Future returns (N,)
            volatility_targets: Future volatility (N,)
            asset_ids: Asset IDs for each sample (N,)
            n_triplets: Number of triplets to mine

        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        raise NotImplementedError


class BinningStrategy(TripletMiningStrategy):
    """
    Binning strategy for efficient triplet mining.

    Bins targets into discrete buckets and samples positives/negatives
    from the same/different bins.
    """

    def __init__(self, config: TripletConfig):
        super().__init__(config)

    def _create_bins(
        self,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Create discrete bins from continuous targets.

        Args:
            targets: Continuous target values (N,)

        Returns:
            Bin indices (N,)
        """
        # Use quantile-based binning for robust handling of outliers
        quantiles = np.linspace(0, 1, self.config.num_bins + 1)
        bin_edges = np.quantile(targets, quantiles)
        bin_indices = np.digitize(targets, bin_edges[1:-1], right=False)
        return bin_indices

    def mine_triplets(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        n_triplets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mine triplets using optimized binning strategy."""
        n_samples = len(return_targets)

        # Create bins for return and volatility
        return_bins = self._create_bins(return_targets)
        vol_bins = self._create_bins(volatility_targets)

        # Convert to numpy arrays for faster operations
        return_bins_array = np.array(return_bins)
        vol_bins_array = np.array(vol_bins)
        asset_ids_array = np.array(asset_ids)

        # Build bin groups using numpy (much faster than Python loops)
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

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        # Pre-compute list of available bins
        return_bin_keys = list(return_bin_groups.keys())
        vol_bin_keys = list(vol_bin_groups.keys())

        # Log progress every 1000 triplets
        log_interval = max(1, n_triplets // 10)

        for i in range(n_triplets):
            if i % log_interval == 0 and i > 0:
                logger.info(f"  Mining progress: {i}/{n_triplets} triplets")
            # Random anchor
            anchor_idx = np.random.randint(0, n_samples)

            # Choose which target to use for this triplet (50/50)
            if np.random.rand() < 0.5:
                # Use return similarity
                anchor_bin = return_bins[anchor_idx]
                same_bin_samples = return_bin_groups.get(anchor_bin, np.array([]))

                # Get different bin for negative (use numpy)
                different_bins = [b for b in return_bin_keys if b != anchor_bin]
            else:
                # Use volatility similarity
                anchor_bin = vol_bins[anchor_idx]
                same_bin_samples = vol_bin_groups.get(anchor_bin, np.array([]))

                # Get different bin for negative
                different_bins = [b for b in vol_bin_keys if b != anchor_bin]

            if len(same_bin_samples) < 2 or len(different_bins) == 0:
                # Skip this triplet
                continue

            # Sample positive from same bin (use boolean indexing - much faster)
            mask = (same_bin_samples != anchor_idx) & (asset_ids_array[same_bin_samples] != asset_ids_array[anchor_idx])
            valid_positives = same_bin_samples[mask]

            if len(valid_positives) == 0:
                # Fallback to any sample from same bin
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

            # Prefer different asset for negative (use boolean indexing)
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


class RandomStrategy(TripletMiningStrategy):
    """
    Random sampling strategy for triplet mining.

    Randomly samples anchors, positives, and negatives.
    Less efficient but simple.
    """

    def __init__(self, config: TripletConfig):
        super().__init__(config)

    def mine_triplets(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        n_triplets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mine triplets using random sampling."""
        n_samples = len(return_targets)

        anchor_indices = []
        positive_indices = []
        negative_indices = []

        for _ in range(n_triplets):
            anchor_idx = np.random.randint(0, n_samples)

            # Find positives (similar targets, preferably different asset)
            if np.random.rand() < 0.5:
                # Use return similarity
                anchor_target = return_targets[anchor_idx]
                targets = return_targets
            else:
                # Use volatility similarity
                anchor_target = volatility_targets[anchor_idx]
                targets = volatility_targets

            # Compute distances
            distances = np.abs(targets - anchor_target)

            # Prefer different asset
            mask = asset_ids != asset_ids[anchor_idx]
            masked_distances = distances.copy()
            masked_distances[~mask] = np.inf

            # Find positive (closest, different asset)
            if np.isfinite(masked_distances).any():
                positive_idx = np.argmin(masked_distances)
            else:
                positive_idx = np.argmin(distances)
                if positive_idx == anchor_idx:
                    distances[anchor_idx] = np.inf
                    positive_idx = np.argmin(distances)

            # Find negative (furthest, different asset)
            if np.isfinite(masked_distances).any():
                negative_idx = np.argmax(masked_distances)
            else:
                negative_idx = np.argmax(distances)
                if negative_idx == anchor_idx:
                    distances[anchor_idx] = -np.inf
                    negative_idx = np.argmax(distances)

            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)

        return (
            np.array(anchor_indices),
            np.array(positive_indices),
            np.array(negative_indices),
        )


class TripletDataset(Dataset):
    """
    PyTorch Dataset for triplet learning.

    Creates triplets (anchor, positive, negative) for training.
    """

    def __init__(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        asset_ids: np.ndarray,
        config: TripletConfig,
        strategy: Optional[TripletMiningStrategy] = None,
    ):
        """
        Initialize TripletDataset.

        Args:
            features: Feature windows (N, lookback, 1)
            return_targets: Future returns (N,)
            volatility_targets: Future volatility (N,)
            asset_ids: Asset IDs (N,)
            config: TripletConfig instance
            strategy: Optional mining strategy (defaults to BinningStrategy)
        """
        self.features = torch.from_numpy(features).float()
        self.return_targets = torch.from_numpy(return_targets).float()
        self.volatility_targets = torch.from_numpy(volatility_targets).float()
        self.asset_ids = asset_ids
        self.config = config

        # Initialize strategy
        if strategy is None:
            if config.strategy == "binning":
                strategy = BinningStrategy(config)
            elif config.strategy == "random":
                strategy = RandomStrategy(config)
            else:
                raise ValueError(f"Unknown strategy: {config.strategy}")

        self.strategy = strategy

        # Pre-compute triplets
        logger.info("Mining triplets...")
        self.anchor_idx, self.positive_idx, self.negative_idx = self.strategy.mine_triplets(
            features,
            return_targets,
            volatility_targets,
            asset_ids,
            config.n_triplets,
        )
        logger.info(f"Mined {len(self.anchor_idx)} triplets")

    def __len__(self) -> int:
        """Return number of triplets."""
        return len(self.anchor_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single triplet.

        Returns:
            Tuple of (anchor_features, positive_features, negative_features,
                     anchor_return_target, positive_return_target, negative_return_target,
                     anchor_vol_target, positive_vol_target, negative_vol_target,
                     anchor_asset_id, positive_asset_id, negative_asset_id)
        """
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


def create_triplet_datasets(
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: TripletConfig,
) -> Tuple[TripletDataset, TripletDataset, TripletDataset]:
    """
    Create triplet datasets for train/val/test splits.

    Args:
        train_data: Tuple of (features, return_targets, volatility_targets, asset_ids)
        val_data: Same structure as train_data
        test_data: Same structure as train_data
        config: TripletConfig instance

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = TripletDataset(*train_data, config=config)
    val_dataset = TripletDataset(*val_data, config=config)
    test_dataset = TripletDataset(*test_data, config=config)

    return train_dataset, val_dataset, test_dataset
