"""
Memory-efficient data loading - process assets in batches to avoid OOM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

from .config import DataConfig

logger = logging.getLogger(__name__)


class MemoryEfficientDataLoader:
    """
    Load and process data in batches to avoid OOM.
    Only keeps currently needed assets in memory.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.asset_ids = list(range(1, config.num_assets + 1))
        self.data_dir = Path(config.data_dir)

    def process_in_batches(
        self,
        batch_size: int = 10,
        output_dir: str = "./processed_data"
    ) -> Dict[str, List]:
        """
        Process assets in batches and save to disk.

        Args:
            batch_size: Number of assets to process at once
            output_dir: Directory to save processed data

        Returns:
            Dictionary with train/val/test file lists
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        split_files = {"train": [], "val": [], "test": []}

        for batch_start in range(0, len(self.asset_ids), batch_size):
            batch_ids = self.asset_ids[batch_start:batch_start + batch_size]
            logger.info(f"Processing batch {batch_start//batch_size + 1}: assets {batch_ids[0]}-{batch_ids[-1]}")

            # Process this batch
            for asset_id in batch_ids:
                try:
                    asset_data = self._process_single_asset(asset_id)
                    if asset_data is None:
                        continue

                    # Save to disk
                    self._save_asset(asset_data, output_path)
                    split_files["train"].append(output_path / f"asset_{asset_id}_train.npz")
                    split_files["val"].append(output_path / f"asset_{asset_id}_val.npz")
                    split_files["test"].append(output_path / f"asset_{asset_id}_test.npz")

                    # Clear from memory
                    del asset_data

                except Exception as e:
                    logger.error(f"Error processing asset {asset_id}: {e}")

            # Force garbage collection after each batch
            import gc
            gc.collect()

        logger.info(f"Processed {len(self.asset_ids)} assets")
        return split_files

    def _process_single_asset(self, asset_id: int):
        """Process single asset (same logic as before)."""
        filepath = self.data_dir / self.config.data_pattern.format(i=asset_id)

        if not filepath.exists():
            return None

        try:
            # Load CSV
            df = pd.read_csv(
                filepath,
                usecols=["price", "volume", "date"],
                engine="c",
            )

            if "price" not in df.columns or "volume" not in df.columns:
                return None

            df["timestamp"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["price", "volume"]).copy()

            if len(df) < self.config.min_samples:
                return None

            df = df.sort_values("timestamp").reset_index(drop=True)

            prices = df["price"].values
            volumes = df["volume"].values

            # Compute features and targets (same as before)
            features = self._compute_features(prices, volumes)
            if len(features) == 0:
                return None

            return_targets, volatility_targets = self._compute_targets(prices, len(features))

            # Create splits
            train_data, val_data, test_data = self._split_asset(
                features, return_targets, volatility_targets, df["timestamp"]
            )

            return {
                "asset_id": asset_id,
                "train": (features[train_slice], return_targets[train_slice], ...),
                "val": (...),
                "test": (...),
            }

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def _compute_features(self, prices, volumes):
        # Same as before
        pass

    def _compute_targets(self, prices, n_samples):
        # Same as before
        pass

    def _save_asset(self, asset_data, output_path):
        """Save processed asset to disk."""
        asset_id = asset_data["asset_id"]

        np.savez_compressed(
            output_path / f"asset_{asset_id}_train.npz",
            features=asset_data["train"][0],
            return_targets=asset_data["train"][1],
            volatility_targets=asset_data["train"][2],
        )

        # Same for val and test
