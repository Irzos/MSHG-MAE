"""
Training callbacks for monitoring and controlling the training process.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """
        Called at the end of an epoch.

        Returns:
            bool: Whether to stop training
        """
        return False

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when monitored metric stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min', baseline: Optional[float] = None,
                 min_epochs: int = 0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            baseline: Baseline value for the monitored metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.min_epochs = int(min_epochs) if min_epochs is not None else 0

        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset the state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == 'min' else -np.inf

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop."""
        # Respect minimal number of epochs before allowing early stop
        if (epoch + 1) < self.min_epochs:
            # Still update best value if available
            current = logs.get('val_loss') if logs else None
            if current is not None:
                if self.best is None:
                    self.best = current
                    self.best_weights = model.state_dict() if model else None
                elif self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    self.best_weights = model.state_dict() if model else None
                    self.wait = 0
                else:
                    self.wait = 0  # Do not accumulate patience before min_epochs
            return False
        current = logs.get('val_loss')
        if current is None:
            logger.warning("EarlyStopping: val_loss not found in logs")
            return False

        # Check if we have improvement
        if self.best is None:
            self.best = current
            self.best_weights = model.state_dict() if model else None
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_weights = model.state_dict() if model else None
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")

                # Restore best weights
                if model and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")

                return True

        return False


class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on monitored metric.
    """

    def __init__(self, directory: Path, filename: str = 'checkpoint.pth',
                 monitor: str = 'val_loss', mode: str = 'min',
                 save_best_only: bool = True, save_weights_only: bool = False,
                 config: Optional[Dict] = None):
        """
        Initialize model checkpoint callback.

        Args:
            directory: Directory to save checkpoints
            filename: Checkpoint filename
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_weights_only: Only save model weights
            config: Model configuration to save with checkpoint
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.config = config

        self.best = None

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Save checkpoint if needed."""
        if model is None:
            return False

        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"ModelCheckpoint: {self.monitor} not found in logs")
            return False

        # Check if we should save
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                save = True
            else:
                save = False
        else:
            save = True

        if save:
            filepath = self.directory / self.filename

            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    self.monitor: current,
                    'logs': logs,
                    'config': self.config
                }
                torch.save(checkpoint, filepath)

            logger.info(f"Model checkpoint saved to {filepath} ({self.monitor}: {current:.4f})")

        return False


class LearningRateMonitor(Callback):
    """
    Monitor and log learning rate during training.
    """

    def __init__(self, logging_interval: int = 1):
        """
        Initialize learning rate monitor.

        Args:
            logging_interval: Log every N epochs
        """
        self.logging_interval = logging_interval

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Log current learning rate."""
        if epoch % self.logging_interval == 0:
            lr = logs.get('learning_rate', 0.0)
            logger.info(f"Learning rate: {lr:.6f}")

        return False


class GradientMonitor(Callback):
    """
    Monitor gradient statistics during training.
    """

    def __init__(self, parameters: Optional[nn.Module] = None,
                 logging_interval: int = 100):
        """
        Initialize gradient monitor.

        Args:
            parameters: Model parameters to monitor
            logging_interval: Log every N batches
        """
        self.parameters = parameters
        self.logging_interval = logging_interval
        self.gradient_norms = deque(maxlen=100)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Monitor gradients after each batch."""
        if self.parameters is None:
            return

        if batch % self.logging_interval == 0:
            total_norm = 0.0
            param_count = 0

            for p in self.parameters.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            total_norm = total_norm ** 0.5
            self.gradient_norms.append(total_norm)

            # Log statistics
            if len(self.gradient_norms) > 0:
                mean_norm = np.mean(self.gradient_norms)
                max_norm = np.max(self.gradient_norms)
                logger.info(f"Gradient norm - Mean: {mean_norm:.4f}, Max: {max_norm:.4f}")


class ValidationCallback(Callback):
    """
    Run validation at specified intervals.
    """

    def __init__(self, validation_fn: callable, interval: int = 1):
        """
        Initialize validation callback.

        Args:
            validation_fn: Function to run validation
            interval: Validate every N epochs
        """
        self.validation_fn = validation_fn
        self.interval = interval

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Run validation if needed."""
        if epoch % self.interval == 0:
            val_results = self.validation_fn(model)
            if logs is not None:
                logs.update(val_results)

        return False


class TensorBoardCallback(Callback):
    """
    Log metrics to TensorBoard.
    """

    def __init__(self, log_dir: Path, comment: str = ''):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to add to run name
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir), comment=comment)
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Log metrics to TensorBoard."""
        if self.writer is None or logs is None:
            return False

        # Log scalars
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # Log model graph (first epoch only)
        if epoch == 0 and model is not None:
            try:
                # Create dummy input
                dummy_input = torch.randn(1, model.in_dim)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

        self.writer.flush()
        return False

    def on_train_end(self, logs: Optional[Dict] = None):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class CallbackList:
    """
    Container for managing multiple callbacks.
    """

    def __init__(self, callbacks: Optional[list] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None,
                     model: Optional[nn.Module] = None) -> bool:
        """Call on_epoch_end for all callbacks."""
        should_stop = False
        for callback in self.callbacks:
            if callback.on_epoch_end(epoch, logs, model):
                should_stop = True
        return should_stop

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
