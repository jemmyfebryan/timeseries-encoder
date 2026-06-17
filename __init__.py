"""
Dual-Head Siamese Triplet TCN for Time-Series Representation Learning.

A production-grade PyTorch implementation for learning scale-invariant
representations from asset price time-series data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Re-export from training package
from training.config import Config
from training.model import DualHeadSiameseTCN, create_model
from training.data import DataLoader, AssetData
from training.dataset import TripletDataset
from training.train import Trainer, train_model
from training.inference import KNNInference, MemoryBank, create_memory_bank
from training.utils import setup_logging, set_seed

__all__ = [
    "Config",
    "DualHeadSiameseTCN",
    "create_model",
    "DataLoader",
    "AssetData",
    "TripletDataset",
    "Trainer",
    "train_model",
    "KNNInference",
    "MemoryBank",
    "create_memory_bank",
    "setup_logging",
    "set_seed",
]
