"""
Optimizer configurations for HyperGraph-MAE training.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Get optimizer based on configuration.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        Configured optimizer
    """
    optimizer_type = config.get('type', 'adamw').lower()
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)

    # Separate parameters for different weight decay
    decay_params, no_decay_params = separate_weight_decay_params(model)

    if optimizer_type == 'adam':
        optimizer = optim.Adam([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, betas=config.get('betas', (0.9, 0.999)))

    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, betas=config.get('betas', (0.9, 0.999)))

    elif optimizer_type == 'sgd':
        optimizer = optim.SGD([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, momentum=config.get('momentum', 0.9))

    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, alpha=config.get('alpha', 0.99))

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict,
                  total_steps: Optional[int] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        total_steps: Total training steps (for some schedulers)

    Returns:
        Configured scheduler or None
    """
    scheduler_type = config.get('type', 'none').lower()

    if scheduler_type == 'none':
        return None

    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 0)
        )

    elif scheduler_type == 'cosine_warmup':
        # Custom cosine with warmup
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=config.get('warmup_steps', 100),
            total_steps=total_steps or config.get('total_steps', 1000),
            eta_min=config.get('eta_min', 0)
        )

    elif scheduler_type == 'onecycle':
        if total_steps is None:
            raise ValueError("total_steps required for OneCycleLR")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=config.get('pct_start', 0.1),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 10000.0)
        )

    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )

    elif scheduler_type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [30, 60, 90]),
            gamma=config.get('gamma', 0.1)
        )

    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )

    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            min_lr=config.get('min_lr', 1e-6)
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler


def separate_weight_decay_params(model: torch.nn.Module) -> tuple:
    """
    Separate model parameters into those that should have weight decay
    and those that shouldn't (biases and normalization parameters).

    Args:
        model: Model to process

    Returns:
        Tuple of (decay_params, no_decay_params)
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to biases and normalization parameters
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    logger.info(f"Parameters: {len(decay_params)} with decay, {len(no_decay_params)} without decay")
    return decay_params, no_decay_params


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with linear warmup.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 0, last_epoch: int = -1):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            eta_min: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + torch.cos(torch.tensor(torch.pi * progress))) / 2
                    for base_lr in self.base_lrs]


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup scheduler.
    """

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate."""
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def get_optimizer_config(model_size: str = 'base') -> Dict:
    """
    Get recommended optimizer configuration based on model size.

    Args:
        model_size: Model size ('small', 'base', 'large')

    Returns:
        Optimizer configuration dictionary
    """
    configs = {
        'small': {
            'type': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'scheduler': {
                'type': 'cosine',
                'T_max': 100
            }
        },
        'base': {
            'type': 'adamw',
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'scheduler': {
                'type': 'onecycle',
                'pct_start': 0.1,
                'div_factor': 25.0
            }
        },
        'large': {
            'type': 'adamw',
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'scheduler': {
                'type': 'cosine_warmup',
                'warmup_steps': 1000,
                'eta_min': 1e-6
            }
        }
    }

    return configs.get(model_size, configs['base'])
