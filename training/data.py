"""
Data loading and preprocessing module with memory-efficient streaming.

Processes assets in batches to avoid OOM - saves to disk incrementally.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import gc

from .config import DataConfig

logger = logging.getLogger(__name__)


class AssetData:
    """Container for a single asset's data."""

    def __init__(
        self,
        asset_id: int,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ):
        """
        Initialize AssetData.

        Args:
            asset_id: Asset identifier (1-128)
            features: Preprocessed feature windows (N, lookback, 2) - channel 0: price, channel 1: volume
            return_targets: Future accumulated log returns (N,)
            volatility_targets: Future volatility (N,)
            timestamps: Timestamps for each sample (N,)
        """
        self.asset_id = asset_id
        self.features = features
        self.return_targets = return_targets
        self.volatility_targets = volatility_targets
        self.timestamps = timestamps

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.return_targets)


class DataLoader:
    """
    Loads and preprocesses asset price data from CSV files.

    Handles:
    - Loading 128 CSV files
    - Computing log returns and volatility targets
    - Creating sliding windows
    - Splitting into train/val/test per asset
    """

    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader.

        Args:
            config: DataConfig instance with preprocessing parameters
        """
        self.config = config
        self.assets: Dict[int, AssetData] = {}
        self.splits: Dict[str, List[AssetData]] = {
            "train": [],
            "val": [],
            "test": []
        }

    def load_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file safely with optimizations.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with 'price', 'volume', and timestamp columns, or None if failed
        """
        try:
            # Optimized CSV reading
            df = pd.read_csv(
                filepath,
                usecols=["price", "volume", "date", "block"],  # Only read needed columns
                engine="c",  # Faster C engine
            )

            # Validate required columns
            if "price" not in df.columns or "volume" not in df.columns:
                logger.warning(f"'price' or 'volume' column not found in {filepath}. Skipping.")
                return None

            # Handle timestamps - use 'date' column, fallback to 'block', then index
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "block" in df.columns:
                # Use block as a proxy for time (sorted by block)
                df["timestamp"] = pd.RangeIndex(start=0, stop=len(df))
            else:
                logger.warning(f"No 'date' or 'block' column in {filepath}. Using index.")
                df["timestamp"] = pd.RangeIndex(start=0, stop=len(df))

            # Drop NaN values in price or volume
            df = df.dropna(subset=["price", "volume"]).copy()

            if len(df) < self.config.min_samples:
                logger.warning(
                    f"Insufficient samples in {filepath} ({len(df)} < {self.config.min_samples}). Skipping."
                )
                return None

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df[["price", "volume", "timestamp"]]

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def compute_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """
        Compute scale-invariant features from prices and volumes.

        Uses log returns or rolling z-score normalization for both features.

        Args:
            prices: Array of prices (T,)
            volumes: Array of volumes (T,)

        Returns:
            Preprocessed features (T - lookback, lookback, 2)
        """
        lookback = self.config.lookback_window

        # Compute log returns for prices
        if self.config.use_log_returns:
            log_prices = np.log(np.maximum(prices, 1e-8))  # Guard against negative/zero
            price_features = np.diff(log_prices, prepend=0)
        else:
            price_features = prices.copy()

        # Compute log returns for volumes (handle invalid values)
        if self.config.use_log_returns:
            # Filter out invalid volumes (<= 0) and use log transform
            valid_volumes = np.maximum(volumes, 1e-8)  # Guard against negative/zero
            log_volumes = np.log(valid_volumes)
            volume_features = np.diff(log_volumes, prepend=0)
        else:
            volume_features = volumes.copy()

        # Create sliding windows for both features
        n_windows = len(price_features) - lookback + 1
        if n_windows <= 0:
            return np.array([])

        price_windows = np.lib.stride_tricks.sliding_window_view(
            price_features, window_shape=lookback
        )  # (n_windows, lookback)

        volume_windows = np.lib.stride_tricks.sliding_window_view(
            volume_features, window_shape=lookback
        )  # (n_windows, lookback)

        # Normalize windows
        if self.config.normalize_method == "rolling_zscore":
            # Normalize price windows
            price_mean = price_windows.mean(axis=1, keepdims=True)
            price_std = price_windows.std(axis=1, keepdims=True) + 1e-8
            price_windows = (price_windows - price_mean) / price_std

            # Normalize volume windows
            volume_mean = volume_windows.mean(axis=1, keepdims=True)
            volume_std = volume_windows.std(axis=1, keepdims=True) + 1e-8
            volume_windows = (volume_windows - volume_mean) / volume_std

        # Stack features to create 2-channel input
        # (n_windows, lookback, 2) where channel 0 is price, channel 1 is volume
        windows = np.stack([price_windows, volume_windows], axis=-1)

        return windows

    def compute_targets(
        self,
        prices: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute future return and volatility targets.

        Args:
            prices: Array of close prices (T,)
            n_samples: Number of samples (windows created)

        Returns:
            Tuple of (return_targets, volatility_targets) each of shape (n_samples,)
        """
        future_window = self.config.future_window
        lookback = self.config.lookback_window

        # Compute log returns (guard against invalid values)
        log_prices = np.log(np.maximum(prices, 1e-8))
        log_returns = np.diff(log_prices, prepend=0)

        return_targets = []
        volatility_targets = []

        for i in range(n_samples):
            # Future window starts after the lookback window
            start_idx = i + lookback
            end_idx = min(start_idx + future_window, len(log_returns))

            if end_idx - start_idx < future_window:
                # Not enough data for full future window
                continue

            future_log_returns = log_returns[start_idx:end_idx]

            # Accumulated log return
            accumulated_return = future_log_returns.sum()
            return_targets.append(accumulated_return)

            # Volatility (std of returns)
            volatility = future_log_returns.std()
            volatility_targets.append(volatility)

        return (
            np.array(return_targets),
            np.array(volatility_targets)
        )

    def split_data(
        self,
        features: np.ndarray,
        return_targets: np.ndarray,
        volatility_targets: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Tuple[slice, slice, slice]:
        """
        Compute train/val/test split indices for a single asset.

        Splits are chronological to prevent look-ahead bias.

        Args:
            features: Feature windows
            return_targets: Return targets
            volatility_targets: Volatility targets
            timestamps: Timestamps

        Returns:
            Tuple of (train_slice, val_slice, test_slice)
        """
        n_samples = len(return_targets)

        train_end = int(n_samples * self.config.train_split)
        val_end = int(n_samples * (self.config.train_split + self.config.val_split))

        train_slice = slice(0, train_end)
        val_slice = slice(train_end, val_end)
        test_slice = slice(val_end, n_samples)

        return train_slice, val_slice, test_slice

    def load_single_asset(self, asset_id: int) -> Optional[AssetData]:
        """
        Load and preprocess a single asset.

        Args:
            asset_id: Asset identifier (1-128)

        Returns:
            AssetData instance or None if loading failed
        """
        filepath = Path(self.config.data_dir) / self.config.data_pattern.format(i=asset_id)

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        df = self.load_csv(filepath)
        if df is None:
            return None

        prices = df["price"].values
        volumes = df["volume"].values
        timestamps = df["timestamp"]

        # Compute features
        features = self.compute_features(prices, volumes)
        if len(features) == 0:
            logger.warning(f"Insufficient data for windows in asset {asset_id}")
            return None

        # Compute targets (based on prices)
        return_targets, volatility_targets = self.compute_targets(
            prices, len(features)
        )

        # Align lengths (targets may be shorter)
        min_len = min(len(features), len(return_targets))
        features = features[:min_len]
        return_targets = return_targets[:min_len]
        volatility_targets = volatility_targets[:min_len]
        timestamps = timestamps[self.config.lookback_window:self.config.lookback_window + min_len]

        return AssetData(
            asset_id=asset_id,
            features=features,
            return_targets=return_targets,
            volatility_targets=volatility_targets,
            timestamps=timestamps,
        )

    def load_all_assets(self) -> None:
        """
        Load all assets in memory-efficient batches to avoid OOM.

        Processes assets in batches, saves to disk, then loads back.
        Memory usage stays constant regardless of number of assets.
        """
        # Check if already processed
        processed_dir = Path("./processed_data")
        if (processed_dir / "train_data.npz").exists():
            logger.info("Found processed data, loading from disk...")
            self._load_from_disk(processed_dir)
            return

        logger.info(f"Loading {self.config.num_assets} assets in batches (memory-efficient)...")

        # Process in batches to avoid OOM
        batch_size = 10  # Process 10 assets at a time
        processed_dir = Path("./processed_data")
        processed_dir.mkdir(exist_ok=True)

        all_train_features = []
        all_train_returns = []
        all_train_vols = []
        all_train_asset_ids = []

        all_val_features = []
        all_val_returns = []
        all_val_vols = []
        all_val_asset_ids = []

        all_test_features = []
        all_test_returns = []
        all_test_vols = []
        all_test_asset_ids = []

        for batch_start in range(1, self.config.num_assets + 1, batch_size):
            batch_end = min(batch_start + batch_size, self.config.num_assets + 1)
            batch_ids = list(range(batch_start, batch_end))

            logger.info(f"Processing batch {batch_start//batch_size + 1}: assets {batch_ids[0]}-{batch_ids[-1]}")

            # Load this batch
            batch_assets = []
            for asset_id in batch_ids:
                try:
                    asset_data = self.load_single_asset(asset_id)
                    if asset_data is not None:
                        batch_assets.append(asset_data)
                except Exception as e:
                    logger.error(f"Error loading asset {asset_id}: {e}")

            if not batch_assets:
                continue

            # Split batch assets
            for asset_data in batch_assets:
                train_slice, val_slice, test_slice = self.split_data(
                    asset_data.features,
                    asset_data.return_targets,
                    asset_data.volatility_targets,
                    asset_data.timestamps,
                )

                # Append to lists
                if train_slice.stop - train_slice.start > 0:
                    all_train_features.append(asset_data.features[train_slice])
                    all_train_returns.append(asset_data.return_targets[train_slice])
                    all_train_vols.append(asset_data.volatility_targets[train_slice])
                    all_train_asset_ids.append(np.full(train_slice.stop - train_slice.start, asset_data.asset_id))

                if val_slice.stop - val_slice.start > 0:
                    all_val_features.append(asset_data.features[val_slice])
                    all_val_returns.append(asset_data.return_targets[val_slice])
                    all_val_vols.append(asset_data.volatility_targets[val_slice])
                    all_val_asset_ids.append(np.full(val_slice.stop - val_slice.start, asset_data.asset_id))

                if test_slice.stop - test_slice.start > 0:
                    all_test_features.append(asset_data.features[test_slice])
                    all_test_returns.append(asset_data.return_targets[test_slice])
                    all_test_vols.append(asset_data.volatility_targets[test_slice])
                    all_test_asset_ids.append(np.full(test_slice.stop - test_slice.start, asset_data.asset_id))

                # Clear individual asset data
                del asset_data

            # Clear batch
            del batch_assets
            gc.collect()

            logger.info(f"  → Completed batch, cleared memory")

        # Concatenate and save to disk
        logger.info("Concatenating data and saving to disk...")

        if all_train_features:
            train_data = {
                "features": np.concatenate(all_train_features, axis=0),
                "return_targets": np.concatenate(all_train_returns, axis=0),
                "volatility_targets": np.concatenate(all_train_vols, axis=0),
                "asset_ids": np.concatenate(all_train_asset_ids, axis=0),
            }
            np.savez_compressed(processed_dir / "train_data.npz", **train_data)
            logger.info(f"Saved train data: {len(train_data['features'])} samples")
            del all_train_features, all_train_returns, all_train_vols, all_train_asset_ids

        if all_val_features:
            val_data = {
                "features": np.concatenate(all_val_features, axis=0),
                "return_targets": np.concatenate(all_val_returns, axis=0),
                "volatility_targets": np.concatenate(all_val_vols, axis=0),
                "asset_ids": np.concatenate(all_val_asset_ids, axis=0),
            }
            np.savez_compressed(processed_dir / "val_data.npz", **val_data)
            logger.info(f"Saved val data: {len(val_data['features'])} samples")
            del all_val_features, all_val_returns, all_val_vols, all_val_asset_ids

        if all_test_features:
            test_data = {
                "features": np.concatenate(all_test_features, axis=0),
                "return_targets": np.concatenate(all_test_returns, axis=0),
                "volatility_targets": np.concatenate(all_test_vols, axis=0),
                "asset_ids": np.concatenate(all_test_asset_ids, axis=0),
            }
            np.savez_compressed(processed_dir / "test_data.npz", **test_data)
            logger.info(f"Saved test data: {len(test_data['features'])} samples")
            del all_test_features, all_test_returns, all_test_vols, all_test_asset_ids

        gc.collect()

        # Now load from disk
        self._load_from_disk(processed_dir)

    def _load_from_disk(self, processed_dir: Path) -> None:
        """Load processed data from disk into splits."""
        logger.info("Loading processed data from disk...")

        # Load train
        train_data = np.load(processed_dir / "train_data.npz")
        self.splits["train"] = [
            AssetData(
                asset_id=0,  # Dummy ID for concatenated data
                features=train_data["features"],
                return_targets=train_data["return_targets"],
                volatility_targets=train_data["volatility_targets"],
                timestamps=pd.RangeIndex(len(train_data["features"])),
            )
        ]

        # Load val
        val_data = np.load(processed_dir / "val_data.npz")
        self.splits["val"] = [
            AssetData(
                asset_id=0,
                features=val_data["features"],
                return_targets=val_data["return_targets"],
                volatility_targets=val_data["volatility_targets"],
                timestamps=pd.RangeIndex(len(val_data["features"])),
            )
        ]

        # Load test
        test_data = np.load(processed_dir / "test_data.npz")
        self.splits["test"] = [
            AssetData(
                asset_id=0,
                features=test_data["features"],
                return_targets=test_data["return_targets"],
                volatility_targets=test_data["volatility_targets"],
                timestamps=pd.RangeIndex(len(test_data["features"])),
            )
        ]

        logger.info("Data loaded successfully from disk")

        # Create splits
        self._create_splits()

    def _create_splits(self) -> None:
        """Splits are now created in load_all_assets directly."""
        pass

    def get_split_data(
        self,
        split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get concatenated data for a specific split.

        Args:
            split: One of "train", "val", "test"

        Returns:
            Tuple of (features, return_targets, volatility_targets, asset_ids)
        """
        assets = self.splits[split]

        features_list = []
        return_targets_list = []
        volatility_targets_list = []
        asset_ids_list = []

        for asset in assets:
            if len(asset) > 0:
                features_list.append(asset.features)
                return_targets_list.append(asset.return_targets)
                volatility_targets_list.append(asset.volatility_targets)
                asset_ids_list.append(
                    np.full(len(asset), asset.asset_id, dtype=np.int32)
                )

        features = np.concatenate(features_list, axis=0)
        return_targets = np.concatenate(return_targets_list, axis=0)
        volatility_targets = np.concatenate(volatility_targets_list, axis=0)
        asset_ids = np.concatenate(asset_ids_list, axis=0)

        return features, return_targets, volatility_targets, asset_ids
