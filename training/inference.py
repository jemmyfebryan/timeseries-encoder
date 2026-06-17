"""
Inference module with K-NN Memory Bank.

Implements weighted K-NN prediction using cosine similarity
and softmax weighting over historical embeddings.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import pickle

from .config import Config, InferenceConfig
from .model import DualHeadSiameseTCN

logger = logging.getLogger(__name__)


@dataclass
class MemoryBank:
    """
    Memory bank for storing historical embeddings and targets.

    Stores embeddings for the test set to use as historical reference
    for K-NN predictions.
    """

    return_embeddings: np.ndarray  # (N, return_dim)
    volatility_embeddings: np.ndarray  # (N, vol_dim)
    return_targets: np.ndarray  # (N,)
    volatility_targets: np.ndarray  # (N,)
    asset_ids: np.ndarray  # (N,)
    timestamps: Optional[np.ndarray] = None  # (N,)

    def __len__(self) -> int:
        """Return number of samples in memory bank."""
        return len(self.return_targets)

    def save(self, filepath: Path) -> None:
        """Save memory bank to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Memory bank saved: {filepath}")

    @staticmethod
    def load(filepath: Path) -> "MemoryBank":
        """Load memory bank from disk."""
        with open(filepath, "rb") as f:
            bank = pickle.load(f)
        logger.info(f"Memory bank loaded: {filepath} ({len(bank)} samples)")
        return bank


class KNNInference:
    """
    K-NN Inference with weighted averaging using softmax.

    For a query sample:
    1. Compute embeddings using trained network
    2. Find Top-K most similar historical states (cosine similarity)
    3. Apply softmax to convert similarities to weights
    4. Predict as weighted average of targets
    """

    def __init__(
        self,
        model: DualHeadSiameseTCN,
        memory_bank: MemoryBank,
        config: InferenceConfig,
        device: str = "cuda",
    ):
        """
        Initialize KNNInference.

        Args:
            model: Trained DualHeadSiameseTCN model
            memory_bank: MemoryBank with historical embeddings
            config: InferenceConfig instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.memory_bank = memory_bank
        self.config = config
        self.device = torch.device(device)

        # Convert memory bank to tensors for efficient computation
        self.return_bank = torch.from_numpy(memory_bank.return_embeddings).float().to(self.device)
        self.vol_bank = torch.from_numpy(memory_bank.volatility_embeddings).float().to(self.device)
        self.return_targets = torch.from_numpy(memory_bank.return_targets).float().to(self.device)
        self.vol_targets = torch.from_numpy(memory_bank.volatility_targets).float().to(self.device)

    @torch.no_grad()
    def compute_embeddings(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute embeddings for input features.

        Args:
            features: Input features (N, T, C)

        Returns:
            Tuple of (return_embeddings, volatility_embeddings)
        """
        self.model.eval()

        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).float()

        features = features.to(self.device)

        # Compute embeddings
        return_emb, vol_emb = self.model(features)

        return (
            return_emb.cpu().numpy(),
            vol_emb.cpu().numpy(),
        )

    @torch.no_grad()
    def predict(
        self,
        query_features: np.ndarray,
        use_return_head: bool = True,
        use_volatility_head: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions for query samples using K-NN.

        Args:
            query_features: Query features (N, T, C)
            use_return_head: Whether to use return head
            use_volatility_head: Whether to use volatility head

        Returns:
            Tuple of (predictions, details_dict)
            predictions: (N, 2) array with [predicted_return, predicted_volatility]
            details_dict: Dictionary with detailed info
        """
        self.model.eval()

        # Convert to tensor
        if not isinstance(query_features, torch.Tensor):
            query_features = torch.from_numpy(query_features).float()

        query_features = query_features.to(self.device)

        # Compute query embeddings
        query_return_emb, query_vol_emb = self.model(query_features)

        # Initialize predictions
        n_queries = len(query_features)
        return_predictions = np.zeros(n_queries)
        vol_predictions = np.zeros(n_queries)

        if use_return_head:
            # Compute cosine similarity with return embeddings
            return_sim = torch.sum(
                query_return_emb.unsqueeze(1) * self.return_bank.unsqueeze(0),
                dim=2
            )  # (n_queries, bank_size)

            # Get top-k indices
            top_k = min(self.config.top_k, len(self.memory_bank))
            top_return_sim, top_return_idx = torch.topk(return_sim, k=top_k, dim=1)

            # Apply softmax to get weights
            return_weights = torch.softmax(
                top_return_sim / self.config.temperature,
                dim=1
            )  # (n_queries, top_k)

            # Weighted average
            return_predictions = (
                return_weights * self.return_targets[top_return_idx]
            ).sum(dim=1).cpu().numpy()

        if use_volatility_head:
            # Compute cosine similarity with volatility embeddings
            vol_sim = torch.sum(
                query_vol_emb.unsqueeze(1) * self.vol_bank.unsqueeze(0),
                dim=2
            )  # (n_queries, bank_size)

            # Get top-k indices
            top_k = min(self.config.top_k, len(self.memory_bank))
            top_vol_sim, top_vol_idx = torch.topk(vol_sim, k=top_k, dim=1)

            # Apply softmax to get weights
            vol_weights = torch.softmax(
                top_vol_sim / self.config.temperature,
                dim=1
            )  # (n_queries, top_k)

            # Weighted average
            vol_predictions = (
                vol_weights * self.vol_targets[top_vol_idx]
            ).sum(dim=1).cpu().numpy()

        # Combine predictions
        predictions = np.stack([return_predictions, vol_predictions], axis=1)

        # Details
        details = {
            "return_predictions": return_predictions,
            "volatility_predictions": vol_predictions,
        }

        return predictions, details

    @torch.no_grad()
    def predict_single(
        self,
        query_features: np.ndarray,
    ) -> Tuple[float, float, Dict]:
        """
        Make prediction for a single query sample.

        Args:
            query_features: Query features (T, C) or (1, T, C)

        Returns:
            Tuple of (predicted_return, predicted_volatility, details)
        """
        # Ensure batch dimension
        if query_features.ndim == 2:
            query_features = query_features[np.newaxis, ...]

        predictions, details = self.predict(query_features)

        return (
            predictions[0, 0],
            predictions[0, 1],
            details,
        )

    def evaluate(
        self,
        test_features: np.ndarray,
        test_return_targets: np.ndarray,
        test_vol_targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate predictions on test set.

        Args:
            test_features: Test features (N, T, C)
            test_return_targets: True return targets (N,)
            test_vol_targets: True volatility targets (N,)

        Returns:
            Dictionary of evaluation metrics
        """
        predictions, _ = self.predict(test_features)

        pred_returns = predictions[:, 0]
        pred_vols = predictions[:, 1]

        # Compute metrics
        return_mae = np.mean(np.abs(pred_returns - test_return_targets))
        return_mse = np.mean((pred_returns - test_return_targets) ** 2)

        vol_mae = np.mean(np.abs(pred_vols - test_vol_targets))
        vol_mse = np.mean((pred_vols - test_vol_targets) ** 2)

        metrics = {
            "return_mae": return_mae,
            "return_mse": return_mse,
            "return_rmse": np.sqrt(return_mse),
            "volatility_mae": vol_mae,
            "volatility_mse": vol_mse,
            "volatility_rmse": np.sqrt(vol_mse),
        }

        logger.info("Evaluation Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return metrics


def create_memory_bank(
    model: DualHeadSiameseTCN,
    features: np.ndarray,
    return_targets: np.ndarray,
    volatility_targets: np.ndarray,
    asset_ids: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    device: str = "cuda",
    batch_size: int = 1024,
) -> MemoryBank:
    """
    Create a memory bank by computing embeddings for all samples.

    Args:
        model: Trained model
        features: Input features (N, T, C)
        return_targets: Return targets (N,)
        volatility_targets: Volatility targets (N,)
        asset_ids: Asset IDs (N,)
        timestamps: Optional timestamps (N,)
        device: Device to run on
        batch_size: Batch size for embedding computation

    Returns:
        MemoryBank instance
    """
    model = model.to(device)
    model.eval()

    n_samples = len(features)

    all_return_embeddings = []
    all_vol_embeddings = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_features = features[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch_features).float().to(device)

            return_emb, vol_emb = model(batch_tensor)

            all_return_embeddings.append(return_emb.cpu().numpy())
            all_vol_embeddings.append(vol_emb.cpu().numpy())

    return_embeddings = np.concatenate(all_return_embeddings, axis=0)
    vol_embeddings = np.concatenate(all_vol_embeddings, axis=0)

    memory_bank = MemoryBank(
        return_embeddings=return_embeddings,
        volatility_embeddings=vol_embeddings,
        return_targets=return_targets,
        volatility_targets=volatility_targets,
        asset_ids=asset_ids,
        timestamps=timestamps,
    )

    logger.info(f"Created memory bank with {len(memory_bank)} samples")

    return memory_bank
