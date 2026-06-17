"""
Example custom configuration for Dual-Head Siamese Triplet TCN.

This file demonstrates how to create a custom configuration.
Run with: python main.py --config example_config.py
"""

from training.config import (
    Config,
    DataConfig,
    ModelConfig,
    TripletConfig,
    TrainingConfig,
    InferenceConfig,
)

# Data configuration
data_config = DataConfig(
    # Paths
    data_dir="./alpha_prices_csv",
    data_pattern="alpha_prices_netuid_{i}.csv",
    num_assets=128,

    # Window sizes
    lookback_window=24,  # 12-hour input window
    future_window=24,    # 12-hour future window for targets

    # Data splits (per asset, chronological)
    train_split=0.60,
    val_split=0.20,
    test_split=0.20,

    # Preprocessing
    use_log_returns=True,
    normalize_method="rolling_zscore",  # Options: "rolling_zscore", "none"
    rolling_window=24,

    # Minimum samples (skip assets with fewer samples)
    min_samples=100,
)

# Model architecture configuration
model_config = ModelConfig(
    # Input/Output
    input_dim=2,  # Single feature (close price derived)
    embedding_dim=64,

    # TCN backbone
    hidden_dims=(64, 128, 256),  # Channels for each TCN layer
    kernel_size=3,
    dilation_base=2,
    dropout=0.2,

    # Projection heads
    return_head_dim=32,
    volatility_head_dim=32,
)

# Triplet mining configuration
triplet_config = TripletConfig(
    # Mining strategy
    strategy="binning",  # Options: "binning", "random"

    # Binning parameters
    num_bins=20,
    top_k_percent=0.1,

    # Triplet margins
    return_margin=1.0,
    volatility_margin=1.0,

    # Number of triplets
    samples_per_asset=100,
    n_triplets=10000,
)

# Training configuration
training_config = TrainingConfig(
    # Training parameters
    num_epochs=100,
    batch_size=64,

    # Optimizer
    learning_rate=1e-3,
    weight_decay=1e-5,

    # Learning rate scheduler
    use_scheduler=True,
    scheduler_patience=5,
    scheduler_factor=0.5,

    # Early stopping
    early_stopping_patience=15,
    min_delta=1e-4,

    # Checkpointing
    checkpoint_dir="./checkpoints",
    checkpoint_every=5,

    # Logging
    log_interval=10,
    tensorboard_dir="./runs",

    # Device (set based on availability)
    device="cuda",
)

# Inference configuration
inference_config = InferenceConfig(
    # K-NN parameters
    top_k=100,
    temperature=1.0,

    # Memory bank storage
    memory_bank_dir="./memory_bank",
)

# Main configuration
config = Config(
    data=data_config,
    model=model_config,
    triplet=triplet_config,
    training=training_config,
    inference=inference_config,

    # Random seed
    seed=42,
)
