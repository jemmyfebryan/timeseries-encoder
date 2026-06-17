"""
TCN model architecture with dual projection heads.

Implements a Temporal Convolutional Network (TCN) backbone
with separate embedding heads for return and volatility prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated convolution.

    Contains two dilated convolutions with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        """
        Initialize TemporalBlock.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout rate
        """
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal block.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, out_channels, T)
        """
        residual = x

        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class TCNBackbone(nn.Module):
    """
    Temporal Convolutional Network (TCN) backbone.

    Stacks multiple temporal blocks with increasing dilation.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize TCN backbone.

        Args:
            config: ModelConfig instance
        """
        super().__init__()

        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.kernel_size = config.kernel_size
        self.dilation_base = config.dilation_base
        self.dropout = config.dropout

        # Build TCN layers
        layers = []
        in_channels = config.input_dim

        for i, out_channels in enumerate(config.hidden_dims):
            dilation = self.dilation_base ** i

            layer = TemporalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.kernel_size,
                dilation=dilation,
                dropout=config.dropout,
            )

            layers.append(layer)
            in_channels = out_channels

        self.tcn_layers = nn.ModuleList(layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN backbone.

        Args:
            x: Input tensor (B, T, C) or (B, C, T)

        Returns:
            Encoded representation (B, final_hidden_dim)
        """
        # Ensure (B, C, T) format
        if x.dim() == 3 and x.size(2) < x.size(1):
            x = x.transpose(1, 2)

        # Pass through TCN layers
        for layer in self.tcn_layers:
            x = layer(x)

        # Global pooling
        x = self.global_pool(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)

        return x


class ProjectionHead(nn.Module):
    """
    Projection head for embedding generation.

    Maps TCN output to embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Initialize projection head.

        Args:
            input_dim: Input dimension
            output_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Embedding tensor (B, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)

        return x


class DualHeadSiameseTCN(nn.Module):
    """
    Dual-Head Siamese TCN for time-series representation learning.

    Uses a shared TCN backbone with two separate projection heads:
    - Return embedding head: Optimized for return similarity
    - Volatility embedding head: Optimized for volatility similarity
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Dual-Head Siamese TCN.

        Args:
            config: ModelConfig instance
        """
        super().__init__()

        self.config = config

        # Shared TCN backbone
        self.backbone = TCNBackbone(config)

        # Return projection head
        self.return_head = ProjectionHead(
            input_dim=config.hidden_dims[-1],
            output_dim=config.return_head_dim,
            hidden_dim=128,
            dropout=config.dropout,
        )

        # Volatility projection head
        self.volatility_head = ProjectionHead(
            input_dim=config.hidden_dims[-1],
            output_dim=config.volatility_head_dim,
            hidden_dim=128,
            dropout=config.dropout,
        )

        logger.info(
            f"Initialized DualHeadSiameseTCN with "
            f"{sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (B, T, C) or (B, C, T)

        Returns:
            Tuple of (return_embeddings, volatility_embeddings)
        """
        # Shared backbone encoding
        shared_repr = self.backbone(x)  # (B, hidden_dim)

        # Separate projection heads
        return_emb = self.return_head(shared_repr)  # (B, return_head_dim)
        vol_emb = self.volatility_head(shared_repr)  # (B, volatility_head_dim)

        return return_emb, vol_emb

    def get_return_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get return embedding only.

        Args:
            x: Input tensor

        Returns:
            Return embedding tensor
        """
        shared_repr = self.backbone(x)
        return self.return_head(shared_repr)

    def get_volatility_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get volatility embedding only.

        Args:
            x: Input tensor

        Returns:
            Volatility embedding tensor
        """
        shared_repr = self.backbone(x)
        return self.volatility_head(shared_repr)


def create_model(config: ModelConfig) -> DualHeadSiameseTCN:
    """
    Create a DualHeadSiameseTCN model.

    Args:
        config: ModelConfig instance

    Returns:
        Initialized model
    """
    model = DualHeadSiameseTCN(config)
    return model
