"""
Main entry point for Dual-Head Siamese Triplet TCN training and inference.

This module orchestrates the entire pipeline:
1. Data loading and preprocessing
2. Triplet dataset creation
3. Model training
4. Memory bank creation
5. Inference evaluation

Usage:
    python main.py --mode train --config config.py
    python main.py --mode inference --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
from pathlib import Path

import torch  # Import torch early for torch.load usage
from training.config import Config
from training.data import DataLoader
from training.dataset import create_triplet_datasets
from training.model import create_model
from training.train import train_model, Trainer
from training.inference import create_memory_bank, KNNInference, MemoryBank
from training.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dual-Head Siamese Triplet TCN for time-series representation learning"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "full"],
        default="full",
        help="Mode to run: train, inference, or full (both)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for inference or resuming training",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (Python file)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./alpha_prices_csv",
        help="Directory containing CSV files",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="training.log",
        help="Log file path",
    )

    parser.add_argument(
        "--memory-bank-path",
        type=str,
        default=None,
        help="Path to load/save memory bank",
    )

    return parser.parse_args()


def load_custom_config(config_path: str) -> Config:
    """
    Load custom configuration from Python file.

    Args:
        config_path: Path to config file

    Returns:
        Config instance
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if hasattr(config_module, "config"):
        return config_module.config
    elif hasattr(config_module, "Config"):
        return config_module.Config()
    else:
        logger.warning(f"No config found in {config_path}, using default")
        return Config()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    logger.info("=" * 80)
    logger.info("Dual-Head Siamese Triplet TCN")
    logger.info("=" * 80)

    # Load configuration
    if args.config:
        config = load_custom_config(args.config)
        logger.info(f"Loaded custom config from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    config.seed = args.seed

    # Override data directory if specified
    if args.data_dir:
        config.data.data_dir = args.data_dir

    # Set random seed
    set_seed(config.seed)

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config.data)
    data_loader.load_all_assets()

    if len(data_loader.assets) == 0:
        logger.error("No assets loaded! Check data directory and file paths.")
        return

    # Get split data
    train_data = data_loader.get_split_data("train")
    val_data = data_loader.get_split_data("val")
    test_data = data_loader.get_split_data("test")

    logger.info(f"Train samples: {len(train_data[0]):,}")
    logger.info(f"Val samples: {len(val_data[0]):,}")
    logger.info(f"Test samples: {len(test_data[0]):,}")

    # Create triplet datasets
    logger.info("Creating triplet datasets...")
    train_dataset, val_dataset, test_dataset = create_triplet_datasets(
        train_data, val_data, test_data, config.triplet
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config.model)

    # Training mode
    if args.mode in ["train", "full"]:
        logger.info("Starting training...")

        # Load checkpoint if specified
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                trainer = Trainer(model, train_dataset, val_dataset, config)
                trainer.load_checkpoint(checkpoint_path)
                logger.info(f"Resumed from checkpoint: {checkpoint_path}")

        # Train
        train_model(model, train_dataset, val_dataset, config)

        logger.info("Training completed!")

        # Load best model for inference
        best_checkpoint = Path(config.training.checkpoint_dir) / "best_model.pt"
        if best_checkpoint.exists():
            logger.info(f"Loading best model: {best_checkpoint}")
            model.load_state_dict(
                torch.load(
                    best_checkpoint,
                    map_location=config.training.device,
                )["model_state_dict"]
            )
        else:
            logger.warning(f"Best model not found: {best_checkpoint}")

    # Inference mode
    if args.mode in ["inference", "full"]:
        logger.info("Starting inference...")

        # Load checkpoint if not already loaded
        if args.mode == "inference" and args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=config.training.device,
                )["model_state_dict"]
            )
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        # Create or load memory bank
        memory_bank_path = args.memory_bank_path or Path(
            config.inference.memory_bank_dir
        ) / "memory_bank.pkl"

        if Path(memory_bank_path).exists():
            logger.info(f"Loading memory bank: {memory_bank_path}")
            memory_bank = MemoryBank.load(Path(memory_bank_path))
        else:
            logger.info("Creating memory bank from test set...")
            memory_bank = create_memory_bank(
                model,
                test_data[0],
                test_data[1],
                test_data[2],
                test_data[3],
                device=config.training.device,
            )
            memory_bank.save(Path(memory_bank_path))

        # Create inference engine
        inference_engine = KNNInference(
            model,
            memory_bank,
            config.inference,
            device=config.training.device,
        )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = inference_engine.evaluate(
            test_data[0],
            test_data[1],
            test_data[2],
        )

        logger.info("Evaluation completed!")

        # Log summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Assets loaded: {len(data_loader.assets)}")
        logger.info(f"Train samples: {len(train_data[0]):,}")
        logger.info(f"Val samples: {len(val_data[0]):,}")
        logger.info(f"Test samples: {len(test_data[0]):,}")
        logger.info(f"Memory bank size: {len(memory_bank):,}")

        if args.mode in ["inference", "full"] and "metrics" in locals():
            logger.info("\nTest Set Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        logger.info("=" * 80)


if __name__ == "__main__":
    main()
