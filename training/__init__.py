"""
Training package for Dual-Head Siamese Triplet TCN.

This package contains all modules for:
- Configuration management
- Data loading and preprocessing
- Triplet dataset creation
- Model architecture
- Training pipeline
- Inference and K-NN memory bank
"""

__version__ = "1.0.0"

# Core imports
from .config import Config
from .data import DataLoader, AssetData
from .dataset import TripletDataset, create_triplet_datasets
from .model import DualHeadSiameseTCN, create_model
from .train import Trainer, train_model
from .inference import KNNInference, MemoryBank, create_memory_bank
from .utils import setup_logging, set_seed

__all__ = [
    # Config
    "Config",
    # Data
    "DataLoader",
    "AssetData",
    # Dataset
    "TripletDataset",
    "create_triplet_datasets",
    # Model
    "DualHeadSiameseTCN",
    "create_model",
    # Training
    "Trainer",
    "train_model",
    # Inference
    "KNNInference",
    "MemoryBank",
    "create_memory_bank",
    # Utils
    "setup_logging",
    "set_seed",
]
