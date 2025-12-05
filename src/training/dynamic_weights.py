"""
Dynamic loss weighting mechanisms for multi-objective training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting using uncertainty estimation.
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(self, num_tasks: int = 2, init_log_vars: float = 0.0):
        """
        Initialize adaptive loss weighting.
        
        Args:
            num_tasks: Number of loss components (recon, edge)
            init_log_vars: Initial log variance values
        """
        super().__init__()
        # Learnable log variance parameters for each task
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_vars))
        self.num_tasks = num_tasks
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate weighted total loss using learned uncertainties.
        
        Args:
            losses: Dictionary with 'recon_loss', 'edge_loss'
            
        Returns:
            Total weighted loss and individual weights
        """
        # Extract losses in consistent order
        loss_values = torch.stack([
            losses['recon_loss'],
            losses['edge_loss']
        ])
        
        # Calculate precision (inverse variance)
        precision = torch.exp(-self.log_vars)
        
        # Weighted loss: L_i / (2 * σ_i^2) + log(σ_i)
        weighted_losses = precision * loss_values + 0.5 * self.log_vars
        total_loss = weighted_losses.sum()
        
        # Return weights for logging
        weights = {
            'recon_weight': precision[0].item(),
            'edge_weight': precision[1].item()
        }
        
        return total_loss, weights


class GradNormWeighting(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.
    Balances training rates between different tasks.
    """
    
    def __init__(self, num_tasks: int = 2, alpha: float = 1.5, initial_task_loss: Optional[torch.Tensor] = None):
        """
        Initialize GradNorm weighting.
        
        Args:
            num_tasks: Number of loss components
            alpha: Restoring force strength
            initial_task_loss: Initial loss values for normalization
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        # Learnable task weights
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        # Store initial losses for relative loss rate calculation
        self.register_buffer('initial_losses', 
                           initial_task_loss if initial_task_loss is not None 
                           else torch.ones(num_tasks))
        
    def forward(self, losses: Dict[str, torch.Tensor], shared_params: nn.Parameter) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate GradNorm weighted loss.
        
        Args:
            losses: Dictionary with individual losses
            shared_params: Shared parameters for gradient calculation
            
        Returns:
            Total weighted loss and current weights
        """
        # Extract losses (keep reconstruction and edge losses only)
        loss_values = torch.stack([
            losses['recon_loss'],
            losses['edge_loss']
        ])
        
        # Normalize task weights
        normalized_weights = F.softmax(self.task_weights, dim=0) * self.num_tasks
        
        # Calculate weighted loss
        weighted_loss = (normalized_weights * loss_values).sum()
        
        # Calculate gradients for each task
        task_grads = []
        for i, (loss_val, weight) in enumerate(zip(loss_values, normalized_weights)):
            grad = torch.autograd.grad(weight * loss_val, shared_params, 
                                     retain_graph=True, create_graph=True)[0]
            task_grads.append(grad.norm())
            
        task_grads = torch.stack(task_grads)
        
        # Calculate relative loss rates
        current_losses = loss_values.detach()
        relative_rates = current_losses / self.initial_losses
        
        # Target gradient norm
        avg_grad_norm = task_grads.mean()
        target_grads = avg_grad_norm * (relative_rates ** self.alpha)
        
        # GradNorm loss for updating task weights
        gradnorm_loss = F.l1_loss(task_grads, target_grads.detach())
        
        weights = {
            'recon_weight': normalized_weights[0].item(),
            'edge_weight': normalized_weights[1].item()
        }
        
        return weighted_loss + gradnorm_loss, weights


class DynamicWeightAveraging(nn.Module):
    """
    Dynamic Weight Averaging (DWA) for multi-task learning.
    Adjusts weights based on relative loss change rates.
    """
    
    def __init__(self, num_tasks: int = 2, temperature: float = 2.0, window_size: int = 5):
        """
        Initialize DWA weighting.
        
        Args:
            num_tasks: Number of loss components
            temperature: Temperature parameter for softmax
            window_size: Window size for loss rate calculation
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.window_size = window_size
        
        # Store loss history
        self.loss_history = []
        
    def forward(self, losses: Dict[str, torch.Tensor], epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate DWA weighted loss.
        
        Args:
            losses: Dictionary with individual losses
            epoch: Current epoch number
            
        Returns:
            Total weighted loss and current weights
        """
        # Extract current losses
        current_losses = torch.stack([
            losses['recon_loss'],
            losses['edge_loss']
        ])
        
        # GPU-optimized: keep history on GPU; avoid CPU transfers
        self.loss_history.append(current_losses.detach())
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            
        # Calculate weights
        if len(self.loss_history) >= 2:
            # Calculate relative loss rates
            prev_losses = self.loss_history[-2]
            curr_losses = self.loss_history[-1]
            
            # Avoid division by zero
            loss_rates = curr_losses / (prev_losses + 1e-8)
            
            # Apply temperature scaling and softmax
            weights = F.softmax(loss_rates / self.temperature, dim=0) * self.num_tasks
        else:
            # Equal weights for initial epochs
            weights = torch.ones(self.num_tasks, device=current_losses.device)
            
        # Calculate weighted loss
        weighted_loss = (weights * current_losses).sum()
        
        weight_dict = {
            'recon_weight': weights[0].item(),
            'edge_weight': weights[1].item()
        }
        
        return weighted_loss, weight_dict


class RandomWeightPerturbation(nn.Module):
    """
    Random weight perturbation for robust multi-task learning.
    Adds controlled noise to loss weights during training.
    """
    
    def __init__(self, base_weights: Dict[str, float], noise_std: float = 0.1):
        """
        Initialize random weight perturbation.
        
        Args:
            base_weights: Base weights for each loss component
            noise_std: Standard deviation of noise
        """
        super().__init__()
        self.base_weights = base_weights
        self.noise_std = noise_std
        
    def forward(self, losses: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate loss with perturbed weights.
        
        Args:
            losses: Dictionary with individual losses
            training: Whether in training mode
            
        Returns:
            Total weighted loss and current weights
        """
        if training:
            # Add Gaussian noise to weights
            perturbed_weights = {
                key: max(0.01, val + torch.randn(1).item() * self.noise_std)
                for key, val in self.base_weights.items()
            }
        else:
            # Use base weights during evaluation
            perturbed_weights = self.base_weights.copy()
            
        # Calculate weighted loss
        weighted_loss = (
            perturbed_weights['recon_weight'] * losses['recon_loss'] +
            perturbed_weights['edge_weight'] * losses['edge_loss']
        )
        
        return weighted_loss, perturbed_weights


class LossBalancedWeighting(nn.Module):
    """
    Loss-balanced weighting that maintains equal contribution from each loss.
    """
    
    def __init__(self, momentum: float = 0.9):
        """
        Initialize loss-balanced weighting.
        
        Args:
            momentum: Momentum for exponential moving average
        """
        super().__init__()
        self.momentum = momentum
        self.register_buffer('loss_ema', torch.ones(2))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate loss-balanced weighted loss.
        
        Args:
            losses: Dictionary with individual losses
            
        Returns:
            Total weighted loss and current weights
        """
        # Extract current losses
        current_losses = torch.stack([
            losses['recon_loss'],
            losses['edge_loss']
        ])
        
        # Update EMA of losses
        self.loss_ema = self.momentum * self.loss_ema + (1 - self.momentum) * current_losses.detach()
        
        # Calculate inverse weights (higher loss gets lower weight)
        inv_weights = 1.0 / (self.loss_ema + 1e-8)
        weights = inv_weights / inv_weights.sum() * 2  # Normalize to sum to 2
        
        # Calculate weighted loss
        weighted_loss = (weights * current_losses).sum()
        
        weight_dict = {
            'recon_weight': weights[0].item(),
            'edge_weight': weights[1].item()
        }
        
        return weighted_loss, weight_dict
