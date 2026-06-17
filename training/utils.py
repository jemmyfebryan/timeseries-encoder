"""
Utility functions for the project.

Includes logging setup, random seed setting, and helper functions.
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format,
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get PyTorch device.

    Args:
        device_str: Device string ("cuda" or "cpu")

    Returns:
        torch.device
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)

    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def save_model(
    model: torch.nn.Module,
    filepath: Path,
) -> None:
    """
    Save model state dict.

    Args:
        model: PyTorch model
        filepath: Save path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(
    model: torch.nn.Module,
    filepath: Path,
    device: torch.device,
) -> None:
    """
    Load model state dict.

    Args:
        model: PyTorch model (initialized)
        filepath: Checkpoint path
        device: Device to load to
    """
    state_dict = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)


def format_time(seconds: float) -> str:
    """
    Format seconds into readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self):
        """Initialize meter."""
        self.reset()

    def reset(self) -> None:
        """Reset meter."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.

        Args:
            val: New value
            n: Number of samples for this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


class EarlyStopping:
    """
    Early stopping utility.

    Stops training when validation metric doesn't improve.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" for the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
