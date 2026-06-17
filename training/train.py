"""
Training module for Dual-Head Siamese TCN.

Implements training loop with validation, checkpointing,
and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from .config import Config, TrainingConfig, TripletConfig
from .model import DualHeadSiameseTCN
from .dataset import TripletDataset

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for Dual-Head Siamese TCN.

    Handles training loop, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: DualHeadSiameseTCN,
        train_dataset: TripletDataset,
        val_dataset: TripletDataset,
        config: Config,
    ):
        """
        Initialize trainer.

        Args:
            model: DualHeadSiameseTCN model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Config instance
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.train_config = config.training

        # Device
        self.device = torch.device(self.train_config.device)
        self.model = self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Loss function
        self.return_criterion = nn.TripletMarginLoss(
            margin=self.config.triplet.return_margin,
            p=2,
        )

        self.volatility_criterion = nn.TripletMarginLoss(
            margin=self.config.triplet.volatility_margin,
            p=2,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None
        if self.train_config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.train_config.scheduler_factor,
                patience=self.train_config.scheduler_patience,
            )

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(
            Path(self.train_config.tensorboard_dir) / f"run_{timestamp}"
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

        logger.info(f"Trainer initialized with device: {self.device}")

    def train_step(
        self,
        batch: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single training step with adaptive margins for soft labels.

        Args:
            batch: Tuple of (anchor_feat, pos_feat, neg_feat, targets...)

        Returns:
            Tuple of (return_loss, volatility_loss, total_loss, ...)
        """
        # Unpack batch (includes target values now)
        anchor_feat, pos_feat, neg_feat = batch[0], batch[1], batch[2]
        anchor_return_tgt = batch[3].to(self.device)
        pos_return_tgt = batch[4].to(self.device)
        neg_return_tgt = batch[5].to(self.device)
        anchor_vol_tgt = batch[6].to(self.device)
        pos_vol_tgt = batch[7].to(self.device)
        neg_vol_tgt = batch[8].to(self.device)

        # Move features to device
        anchor_feat = anchor_feat.to(self.device)
        pos_feat = pos_feat.to(self.device)
        neg_feat = neg_feat.to(self.device)

        # Forward pass
        anchor_return_emb, anchor_vol_emb = self.model(anchor_feat)
        pos_return_emb, pos_vol_emb = self.model(pos_feat)
        neg_return_emb, neg_vol_emb = self.model(neg_feat)

        # Compute embedding distances
        pos_return_dist = 1 - torch.sum(anchor_return_emb * pos_return_emb, dim=1)
        neg_return_dist = 1 - torch.sum(anchor_return_emb * neg_return_emb, dim=1)
        pos_vol_dist = 1 - torch.sum(anchor_vol_emb * pos_vol_emb, dim=1)
        neg_vol_dist = 1 - torch.sum(anchor_vol_emb * neg_vol_emb, dim=1)

        # Compute adaptive margins based on target distances
        return_margin = torch.clamp(
            1.0 + torch.tanh((torch.abs(anchor_return_tgt - pos_return_tgt) +
                               torch.abs(anchor_return_tgt - neg_return_tgt)) / 2.0),
            min=0.2, max=1.8
        )

        vol_margin = torch.clamp(
            1.0 + torch.tanh((torch.abs(anchor_vol_tgt - pos_vol_tgt) +
                               torch.abs(anchor_vol_tgt - neg_vol_tgt)) / 2.0),
            min=0.2, max=1.8
        )

        # Compute losses with adaptive margins
        return_loss = torch.clamp(
            pos_return_dist - neg_return_dist + return_margin,
            min=0.0
        ).mean()

        volatility_loss = torch.clamp(
            pos_vol_dist - neg_vol_dist + vol_margin,
            min=0.0
        ).mean()

        total_loss = return_loss + volatility_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return (
            return_loss,
            volatility_loss,
            total_loss,
            pos_return_dist.mean(),
            pos_vol_dist.mean(),
            neg_return_dist.mean(),
            neg_vol_dist.mean(),
        )

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        return_losses = []
        volatility_losses = []
        total_losses = []

        for batch in self.val_loader:
            anchor_feat, pos_feat, neg_feat = batch[0], batch[1], batch[2]

            anchor_feat = anchor_feat.to(self.device)
            pos_feat = pos_feat.to(self.device)
            neg_feat = neg_feat.to(self.device)

            # Forward pass
            anchor_return_emb, anchor_vol_emb = self.model(anchor_feat)
            pos_return_emb, pos_vol_emb = self.model(pos_feat)
            neg_return_emb, neg_vol_emb = self.model(neg_feat)

            # Compute losses
            return_loss = self.return_criterion(anchor_return_emb, pos_return_emb, neg_return_emb)
            volatility_loss = self.volatility_criterion(anchor_vol_emb, pos_vol_emb, neg_vol_emb)
            total_loss = return_loss + volatility_loss

            return_losses.append(return_loss.item())
            volatility_losses.append(volatility_loss.item())
            total_losses.append(total_loss.item())

        metrics = {
            "val_return_loss": np.mean(return_losses),
            "val_volatility_loss": np.mean(volatility_losses),
            "val_total_loss": np.mean(total_losses),
        }

        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with progress bar and per-batch metrics.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        return_losses = []
        volatility_losses = []
        total_losses = []
        pos_return_dists = []
        pos_vol_dists = []
        neg_return_dists = []
        neg_vol_dists = []

        start_time = time.time()

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.train_config.num_epochs}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            (
                return_loss,
                volatility_loss,
                total_loss,
                pos_return_dist,
                pos_vol_dist,
                neg_return_dist,
                neg_vol_dist,
            ) = self.train_step(batch)

            return_losses.append(return_loss.item())
            volatility_losses.append(volatility_loss.item())
            total_losses.append(total_loss.item())
            pos_return_dists.append(pos_return_dist.item())
            pos_vol_dists.append(pos_vol_dist.item())
            neg_return_dists.append(neg_return_dist.item())
            neg_vol_dists.append(neg_vol_dist.item())

            # Update progress bar with current metrics
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "pos_dist": f"{((pos_return_dist + pos_vol_dist) / 2).item():.3f}",
                "neg_dist": f"{((neg_return_dist + neg_vol_dist) / 2).item():.3f}",
            })

            # Log to TensorBoard
            if self.global_step % self.train_config.log_interval == 0:
                self.writer.add_scalar("train/return_loss", return_loss.item(), self.global_step)
                self.writer.add_scalar("train/volatility_loss", volatility_loss.item(), self.global_step)
                self.writer.add_scalar("train/total_loss", total_loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("train/pos_return_dist", pos_return_dist.item(), self.global_step)
                self.writer.add_scalar("train/pos_vol_dist", pos_vol_dist.item(), self.global_step)
                self.writer.add_scalar("train/neg_return_dist", neg_return_dist.item(), self.global_step)
                self.writer.add_scalar("train/neg_vol_dist", neg_vol_dist.item(), self.global_step)

            self.global_step += 1

        pbar.close()

        epoch_time = time.time() - start_time

        # Compute metrics
        avg_pos_return_dist = np.mean(pos_return_dists)
        avg_pos_vol_dist = np.mean(pos_vol_dists)
        avg_neg_return_dist = np.mean(neg_return_dists)
        avg_neg_vol_dist = np.mean(neg_vol_dists)

        metrics = {
            "train_return_loss": np.mean(return_losses),
            "train_volatility_loss": np.mean(volatility_losses),
            "train_total_loss": np.mean(total_losses),
            "train_pos_return_dist": avg_pos_return_dist,
            "train_pos_vol_dist": avg_pos_vol_dist,
            "train_neg_return_dist": avg_neg_return_dist,
            "train_neg_vol_dist": avg_neg_vol_dist,
            "epoch_time": epoch_time,
        }

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        filepath = Path(self.train_config.checkpoint_dir) / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded: {filepath}")

    def train(self) -> None:
        """Run full training loop."""
        logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.train_config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.train_config.num_epochs} - "
                f"Train Loss: {train_metrics['train_total_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_total_loss']:.4f} - "
                f"Pos Dist: {((train_metrics['train_pos_return_dist'] + train_metrics['train_pos_vol_dist']) / 2):.3f} - "
                f"Neg Dist: {((train_metrics['train_neg_return_dist'] + train_metrics['train_neg_vol_dist']) / 2):.3f} - "
                f"Time: {train_metrics['epoch_time']:.1f}s"
            )

            # Log to TensorBoard
            self.writer.add_scalar("epoch/train_total_loss", train_metrics["train_total_loss"], epoch)
            self.writer.add_scalar("epoch/val_total_loss", val_metrics["val_total_loss"], epoch)
            self.writer.add_scalar("epoch/train_return_loss", train_metrics["train_return_loss"], epoch)
            self.writer.add_scalar("epoch/train_volatility_loss", train_metrics["train_volatility_loss"], epoch)
            self.writer.add_scalar("epoch/val_return_loss", val_metrics["val_return_loss"], epoch)
            self.writer.add_scalar("epoch/val_volatility_loss", val_metrics["val_volatility_loss"], epoch)
            self.writer.add_scalar("epoch/train_pos_return_dist", train_metrics["train_pos_return_dist"], epoch)
            self.writer.add_scalar("epoch/train_pos_vol_dist", train_metrics["train_pos_vol_dist"], epoch)
            self.writer.add_scalar("epoch/train_neg_return_dist", train_metrics["train_neg_return_dist"], epoch)
            self.writer.add_scalar("epoch/train_neg_vol_dist", train_metrics["train_neg_vol_dist"], epoch)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["val_total_loss"])

            # Save checkpoint
            if (epoch + 1) % self.train_config.checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Check for improvement
            if val_metrics["val_total_loss"] < self.best_val_loss - self.train_config.min_delta:
                self.best_val_loss = val_metrics["val_total_loss"]
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                logger.info("New best model saved!")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.train_config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        self.writer.close()
        logger.info("Training completed!")


def train_model(
    model: DualHeadSiameseTCN,
    train_dataset: TripletDataset,
    val_dataset: TripletDataset,
    config: Config,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Train the Dual-Head Siamese TCN model.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration
        checkpoint_path: Optional path to checkpoint for resuming training
    """
    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        trainer.load_checkpoint(Path(checkpoint_path))
        logger.info(f"Resumed training from checkpoint: {checkpoint_path}")

    trainer.train()
