"""
Test script to validate the implementation with synthetic data.

This script creates dummy CSV files and runs a minimal test of the pipeline
to verify all components work correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import logging

from training.config import Config, DataConfig
from training.data import DataLoader
from training.dataset import create_triplet_datasets
from training.model import create_model
from training.train import Trainer
from training.inference import create_memory_bank, KNNInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(
    n_assets: int = 5,
    n_samples: int = 1000,
    output_dir: str = "./test_data",
):
    """
    Create dummy CSV files for testing.

    Args:
        n_assets: Number of assets to create
        n_samples: Number of samples per asset
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    logger.info(f"Creating {n_assets} dummy CSV files in {output_dir}...")

    for asset_id in range(1, n_assets + 1):
        # Generate synthetic price data
        np.random.seed(asset_id)

        # Random walk with drift
        returns = np.random.randn(n_samples) * 0.02 + 0.001
        prices = 100 * np.cumprod(1 + returns)

        # Generate synthetic volume data
        volumes = np.exp(np.random.randn(n_samples) * 0.5 + 5)  # Log-normal distribution

        # Create dates
        dates = pd.date_range(
            start="2024-01-01",
            periods=n_samples,
            freq="1h",
        )

        # Create block IDs (300 block intervals)
        blocks = np.arange(5250000, 5250000 + n_samples * 300, 300)

        # Create IDs
        ids = np.arange(4273447, 4273447 + n_samples * 4, 4)

        df = pd.DataFrame({
            "id": ids,
            "date": dates.strftime("%Y-%m-%d"),
            "block": blocks,
            "netuid": asset_id,
            "price": prices,
            "volume": volumes,
        })

        filepath = output_path / f"alpha_prices_netuid_{asset_id}.csv"
        df.to_csv(filepath, index=False)

    logger.info(f"Created dummy data successfully!")


def test_data_loading():
    """Test data loading pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Data Loading")
    logger.info("=" * 80)

    config = Config()
    config.data.data_dir = "./test_data"
    config.data.num_assets = 5
    config.data.min_samples = 50

    data_loader = DataLoader(config.data)
    data_loader.load_all_assets()

    assert len(data_loader.assets) > 0, "No assets loaded!"
    logger.info(f"✓ Loaded {len(data_loader.assets)} assets")

    # Check splits
    train_data = data_loader.get_split_data("train")
    val_data = data_loader.get_split_data("val")
    test_data = data_loader.get_split_data("test")

    logger.info(f"✓ Train samples: {len(train_data[0])}")
    logger.info(f"✓ Val samples: {len(val_data[0])}")
    logger.info(f"✓ Test samples: {len(test_data[0])}")

    return data_loader, train_data, val_data, test_data


def test_dataset_creation(train_data, val_data, test_data):
    """Test triplet dataset creation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Dataset Creation")
    logger.info("=" * 80)

    config = Config()
    config.triplet.n_triplets = 100

    train_dataset, val_dataset, test_dataset = create_triplet_datasets(
        train_data, val_data, test_data, config.triplet
    )

    logger.info(f"✓ Train triplets: {len(train_dataset)}")
    logger.info(f"✓ Val triplets: {len(val_dataset)}")
    logger.info(f"✓ Test triplets: {len(test_dataset)}")

    # Test sampling a batch
    batch = train_dataset[0]
    assert len(batch) == 12, "Batch should have 12 elements"
    logger.info(f"✓ Batch shape correct")

    return train_dataset, val_dataset


def test_model_creation():
    """Test model creation."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Model Creation")
    logger.info("=" * 80)

    config = Config()
    config.model.hidden_dims = (32, 64, 128)  # Smaller for testing

    model = create_model(config.model)

    # Test forward pass
    batch_size = 4
    seq_len = 12
    n_features = 2  # Price and volume

    dummy_input = torch.randn(batch_size, seq_len, n_features)
    return_emb, vol_emb = model(dummy_input)

    logger.info(f"✓ Model forward pass successful")
    logger.info(f"✓ Return embedding shape: {return_emb.shape}")
    logger.info(f"✓ Volatility embedding shape: {vol_emb.shape}")

    return model


def test_training_step(model, train_dataset, val_dataset):
    """Test a single training step."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Training Step")
    logger.info("=" * 80)

    config = Config()
    config.training.device = "cpu"
    config.training.num_epochs = 1
    config.training.batch_size = 4
    config.training.log_interval = 1

    config.model.hidden_dims = (32, 64, 128)

    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Run one epoch
    trainer.train_epoch()

    logger.info(f"✓ Training step completed")

    # Test validation
    val_metrics = trainer.validate()
    logger.info(f"✓ Validation metrics: {val_metrics}")

    return trainer


def test_inference(model, test_data):
    """Test inference pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Inference")
    logger.info("=" * 80)

    config = Config()
    config.inference.top_k = 10
    config.training.device = "cpu"

    # Create memory bank
    memory_bank = create_memory_bank(
        model,
        test_data[0],
        test_data[1],
        test_data[2],
        test_data[3],
        device="cpu",
        batch_size=32,
    )

    logger.info(f"✓ Memory bank created with {len(memory_bank)} samples")

    # Create inference engine
    inference_engine = KNNInference(
        model,
        memory_bank,
        config.inference,
        device="cpu",
    )

    # Test single prediction
    query_features = test_data[0][:5]
    predictions, details = inference_engine.predict(query_features)

    logger.info(f"✓ Predictions shape: {predictions.shape}")
    logger.info(f"✓ Sample predictions: {predictions[:2]}")

    # Test evaluation
    metrics = inference_engine.evaluate(
        test_data[0],
        test_data[1],
        test_data[2],
    )

    logger.info(f"✓ Evaluation metrics computed")

    return inference_engine


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Running Pipeline Tests with Synthetic Data")
    logger.info("=" * 80)

    # Create dummy data
    create_dummy_data(n_assets=5, n_samples=500)

    try:
        # Test data loading
        data_loader, train_data, val_data, test_data = test_data_loading()

        # Test dataset creation
        train_dataset, val_dataset = test_dataset_creation(
            train_data, val_data, test_data
        )

        # Test model creation
        model = test_model_creation()

        # Test training
        trainer = test_training_step(model, train_dataset, val_dataset)

        # Test inference
        inference_engine = test_inference(model, test_data)

        logger.info("\n" + "=" * 80)
        logger.info("All Tests Passed! ✓")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        import shutil
        if Path("./test_data").exists():
            shutil.rmtree("./test_data")
        if Path("./checkpoints").exists():
            shutil.rmtree("./checkpoints")
        if Path("./runs").exists():
            shutil.rmtree("./runs")
        logger.info("\nCleaned up test files")


if __name__ == "__main__":
    main()
