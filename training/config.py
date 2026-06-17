"""
Configuration module for Dual-Head Siamese Triplet TCN.

Defines all hyperparameters, paths, and model configurations.
"""

from dataclasses import dataclass, field
from typing import Tuple
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Data paths
    data_dir: str = "./alpha_prices_csv"
    data_pattern: str = "alpha_prices_netuid_{i}.csv"
    num_assets: int = 128

    # Feature engineering
    lookback_window: int = 12  # 12-hour lookback
    future_window: int = 12    # 12-hour future for targets

    # Data splits (per asset to prevent leakage)
    train_split: float = 0.60
    val_split: float = 0.20
    test_split: float = 0.20

    # Preprocessing options
    use_log_returns: bool = True
    normalize_method: str = "rolling_zscore"  # Options: "rolling_zscore", "none"
    rolling_window: int = 12

    # Minimum samples required (filter short assets)
    min_samples: int = 100

    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert self.lookback_window > 0, "Lookback window must be positive"
        assert self.future_window > 0, "Future window must be positive"


@dataclass
class ModelConfig:
    """TCN model architecture configuration."""

    # Input/Output dimensions
    input_dim: int = 2  # Two features: price and volume
    embedding_dim: int = 64

    # TCN architecture
    hidden_dims: Tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.2

    # Projection heads
    return_head_dim: int = 32
    volatility_head_dim: int = 32

    def __post_init__(self):
        """Validate configuration."""
        assert all(d > 0 for d in self.hidden_dims), "Hidden dims must be positive"
        assert self.kernel_size > 0, "Kernel size must be positive"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"


@dataclass
class TripletConfig:
    """Triplet sampling configuration."""

    # Sampling strategy
    strategy: str = "binning"  # Options: "binning", "random", "hard_negative"

    # Binning parameters
    num_bins: int = 20  # For binning strategy
    top_k_percent: float = 0.1  # For hard negative mining

    # Triplet margin parameters
    return_margin: float = 1.0
    volatility_margin: float = 1.0

    # Mining parameters
    samples_per_asset: int = 100  # For offline mining
    n_triplets: int = 5000  # Number of triplets per epoch (reduced for speed)

    def __post_init__(self):
        """Validate configuration."""
        assert self.strategy in ["binning", "random", "hard_negative"], \
            f"Unknown strategy: {self.strategy}"
        assert self.num_bins > 0, "Number of bins must be positive"
        assert self.return_margin > 0, "Return margin must be positive"
        assert self.volatility_margin > 0, "Volatility margin must be positive"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Training parameters
    num_epochs: int = 100
    batch_size: int = 256

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 5  # Save every N epochs

    # Logging
    log_interval: int = 10  # Log every N batches
    tensorboard_dir: str = "./runs"

    # Device
    device: str = "cuda"  # Will be set based on availability

    def __post_init__(self):
        """Validate and adjust configuration."""
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"


@dataclass
class InferenceConfig:
    """Inference and K-NN configuration."""

    # K-NN parameters
    top_k: int = 100
    temperature: float = 1.0  # For softmax

    # Memory bank
    memory_bank_dir: str = "./memory_bank"

    def __post_init__(self):
        """Validate configuration."""
        assert self.top_k > 0, "Top-K must be positive"
        assert self.temperature > 0, "Temperature must be positive"


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    triplet: TripletConfig = field(default_factory=TripletConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Create directories and set device."""
        try:
            import torch
            # Set device based on availability
            if self.training.device == "cuda" and not torch.cuda.is_available():
                self.training.device = "cpu"
        except ImportError:
            self.training.device = "cpu"

        # Create directories
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.inference.memory_bank_dir).mkdir(parents=True, exist_ok=True)


# Default configuration instance (lazy loaded)
default_config = None

def get_default_config() -> Config:
    """Get default configuration instance."""
    global default_config
    if default_config is None:
        default_config = Config()
    return default_config
