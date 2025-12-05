"""
Target Contribution Control (TCC) for multi-objective loss weighting.

Maintains target contribution ranges for each loss component while avoiding
oscillation common in traditional adaptive weighting methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TargetContributionControl(nn.Module):
    """
    Target Contribution Control (TCC).

    Key ideas:
    1) Target contribution ranges per loss component
    2) Hysteresis control to reduce weight jitter
    3) Log-domain updates for numerical stability
    4) EMA-smoothing of measured contributions
    """
    
    def __init__(self, loss_components: List[str], config: Dict):
        """
        Initialize TCC controller.

        Args:
            loss_components: List like ['reconstruction', 'edge']
            config: TCC configuration
        """
        super().__init__()
        
        # Handle OmegaConf ListConfig and plain lists
        if isinstance(loss_components, str):
            logger.error(f"loss_components should be a list, got string: '{loss_components}'")
            raise ValueError(f"loss_components must be a list, not string: '{loss_components}'")
        
        # Convert OmegaConf ListConfig to a plain list if present
        try:
            from omegaconf import ListConfig
            if isinstance(loss_components, ListConfig):
                loss_components = list(loss_components)
        except ImportError:
            pass  # omegaconf might not be installed
            
        # Ensure iterable semantics
        if not hasattr(loss_components, '__iter__') or isinstance(loss_components, (str, bytes)):
            logger.error(f"loss_components should be iterable, got {type(loss_components)}: {loss_components}")
            raise ValueError(f"loss_components must be iterable, got {type(loss_components)}")
        
        # Ensure plain list for compatibility
        loss_components = list(loss_components)
        
        if len(loss_components) == 0:
            raise ValueError("loss_components cannot be empty")
            
        self.loss_components = loss_components
        self.num_components = len(loss_components)
        
        # Control parameters (tuned for stability)
        self.update_frequency = config.get('update_frequency', 100)  # lower update frequency
        self.tolerance = config.get('tolerance', 0.08)  # larger tolerance to avoid churn
        self.adjustment_rate = config.get('adjustment_rate', 0.04)  # smaller step for stability
        self.ema_alpha = config.get('ema', 0.9)  # smoother EMA
        self.weight_clip = config.get('weight_clip', [0.1, 20.0])
        self.renorm = config.get('renorm', 'mean_log')
        
        # Auto-set target ranges based on component count
        self.target_ratios = self._get_adaptive_ratios(loss_components, config)
        
        # Learnable log-weights (start uniform)
        initial_log_weights = torch.zeros(self.num_components)
        self.log_weights = nn.Parameter(initial_log_weights)
        
        # State variables
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('ema_contributions', torch.zeros(self.num_components))
        self.register_buffer('initialized', torch.tensor(False))
        
        # Loss scale tracking for multi-task balance
        self.register_buffer('ema_scales', torch.ones(self.num_components))
        self.scale_ema_alpha = config.get('scale_ema', 0.9)
        self.register_buffer('scale_initialized', torch.tensor(False))
        
        logger.info(f"TCC initialized for components: {loss_components}")
        logger.info(f"Target ratios: {self.target_ratios}")
        logger.info(f"Step-based config: update_freq={self.update_frequency}, tolerance={self.tolerance:.3f}")
        logger.info(f"Optimized for fast convergence: adj_rate={self.adjustment_rate:.3f}, ema={self.ema_alpha:.3f}")
        
    def _get_adaptive_ratios(self, components: List[str], config: Dict) -> Dict[str, Tuple[float, float]]:
        """Set target ratio ranges adaptively by component count."""
        adaptive_config = config.get('adaptive_ratios', {})
        
        if self.num_components == 1:
            # Single component: ~100% contribution
            return {components[0]: (0.95, 1.0)}
            
        elif self.num_components == 2:
            # Two components: primary + auxiliary
            ratios_config = adaptive_config.get('two_components', {
                'primary': [0.65, 0.75],
                'secondary': [0.25, 0.35]
            })
            primary_idx = 0 if 'reconstruction' in components else 0
            secondary_idx = 1 - primary_idx
            
            return {
                components[primary_idx]: tuple(ratios_config['primary']),
                components[secondary_idx]: tuple(ratios_config['secondary'])
            }
            
        elif self.num_components == 3:
            # Three components: primary + auxiliary + regularization
            ratios_config = adaptive_config.get('three_components', {
                'primary': [0.40, 0.50],
                'auxiliary': [0.30, 0.40],
                'regularization': [0.15, 0.25]
            })

            primary_idx = components.index('reconstruction') if 'reconstruction' in components else 0
            aux_idx = components.index('edge') if 'edge' in components else None
            # Prefer 'descriptor' for regularization, else 'contrastive', else remaining
            if 'descriptor' in components:
                reg_idx = components.index('descriptor')
            elif 'contrastive' in components:
                reg_idx = components.index('contrastive')
            else:
                reg_idx = None

            remaining = [i for i in range(3) if i != primary_idx and i not in [aux_idx, reg_idx] and i is not None]
            if aux_idx is None and remaining:
                aux_idx = remaining.pop(0)
            if reg_idx is None and remaining:
                reg_idx = remaining.pop(0)

            return {
                components[primary_idx]: tuple(ratios_config['primary']),
                components[aux_idx]: tuple(ratios_config['auxiliary']),
                components[reg_idx]: tuple(ratios_config['regularization'])
            }
            
        elif self.num_components == 4:
            # Four components: primary + auxiliary + regularization + contrastive
            ratios_config = adaptive_config.get('four_components', {
                'primary': [0.45, 0.55],
                'auxiliary': [0.20, 0.30],
                'regularization': [0.10, 0.15],
                'contrastive': [0.10, 0.20]
            })
            
            # Assign roles heuristically
            role_mapping = {}
            for i, comp in enumerate(components):
                if comp == 'reconstruction':
                    role_mapping[i] = 'primary'
                elif comp == 'edge':
                    role_mapping[i] = 'auxiliary'
                elif comp == 'regularization':
                    role_mapping[i] = 'regularization'
                elif comp == 'contrastive':
                    role_mapping[i] = 'contrastive'
                else:
                    # Default role assignment
                    if len([r for r in role_mapping.values() if r == 'auxiliary']) == 0:
                        role_mapping[i] = 'auxiliary'
                    else:
                        role_mapping[i] = 'regularization'
            
            return {
                components[i]: tuple(ratios_config[role])
                for i, role in role_mapping.items()
            }
        
        else:
            # More than two components: distribute uniformly with a small margin
            uniform_ratio = 1.0 / self.num_components
            margin = 0.1 * uniform_ratio
            return {
                comp: (uniform_ratio - margin, uniform_ratio + margin)
                for comp in components
            }
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass: compute weighted total loss with scale normalization.

        Args:
            losses: Dict of individual loss components

        Returns:
            total_loss: Weighted total loss
            info: Weights and contribution ratios (with scale diagnostics)
        """
        # Collect raw losses (robust: missing components -> 0)
        if len(losses) > 0:
            some_tensor = next(iter(losses.values()))
            device = some_tensor.device if hasattr(some_tensor, 'device') else (
                torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
            dtype = some_tensor.dtype if hasattr(some_tensor, 'dtype') else torch.float32
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            dtype = torch.float32

        loss_list = []
        for comp in self.loss_components:
            v = losses.get(comp, None)
            if v is None:
                v = torch.tensor(0.0, device=device, dtype=dtype)
            loss_list.append(v)
        raw_loss_values = torch.stack(loss_list)
        
        # Update EMA of loss scales
        with torch.no_grad():
            raw_loss_items = torch.stack([(losses.get(comp, torch.tensor(0.0, device=device, dtype=dtype))).detach()
                                          for comp in self.loss_components])
            
            if self.scale_initialized:
                self.ema_scales = (
                    self.scale_ema_alpha * self.ema_scales + 
                    (1 - self.scale_ema_alpha) * raw_loss_items
                )
            else:
                self.ema_scales = raw_loss_items.clone()
                self.scale_initialized = torch.tensor(True)
        
        # Normalize by EMA scale (core to multi-task balance)
        normalized_loss_values = raw_loss_values / (self.ema_scales + 1e-8)
        
        # Use normalized losses to compute weights
        current_weights = torch.exp(self.log_weights)
        
        # Weighted loss via normalized losses
        weighted_normalized_losses = current_weights * normalized_loss_values
        normalized_total_loss = weighted_normalized_losses.sum()
        
        # Current contribution ratios (normalized basis)
        current_contributions = (weighted_normalized_losses / normalized_total_loss).detach()
        
        # EMA-smooth contribution ratios
        if self.initialized:
            self.ema_contributions = (
                self.ema_alpha * self.ema_contributions + 
                (1 - self.ema_alpha) * current_contributions
            )
        else:
            self.ema_contributions = current_contributions
            self.initialized = torch.tensor(True)
        
        # Update weights with hysteresis based on contributions
        if self.step_count % self.update_frequency == 0:
            self._update_weights()
        
        self.step_count += 1
        
        # Calculate final loss (use normalized losses for backprop to avoid scale dominance)
        weighted_raw_losses = current_weights * raw_loss_values
        weighted_normalized_losses = current_weights * normalized_loss_values
        raw_total_loss = weighted_raw_losses.sum()     # diagnostics only
        total_loss = weighted_normalized_losses.sum()  # backprop
        
        # Return diagnostics (including scale tracking)
        info = {
            'weights': {comp: current_weights[i].item() for i, comp in enumerate(self.loss_components)},
            'contributions': {comp: current_contributions[i].item() for i, comp in enumerate(self.loss_components)},
            'ema_contributions': {comp: self.ema_contributions[i].item() for i, comp in enumerate(self.loss_components)},
            'targets': self.target_ratios,
            'total_loss': total_loss.item(),
            'raw_total_loss': raw_total_loss.item(),
            # Additional scale tracking
            'raw_losses': {comp: raw_loss_values[i].item() for i, comp in enumerate(self.loss_components)},
            'normalized_losses': {comp: normalized_loss_values[i].item() for i, comp in enumerate(self.loss_components)},
            'ema_scales': {comp: self.ema_scales[i].item() for i, comp in enumerate(self.loss_components)}
        }
        
        return total_loss, info
    
    def _update_weights(self):
        """Update weights with hysteresis to avoid jitter."""
        with torch.no_grad():
            for i, comp in enumerate(self.loss_components):
                target_low, target_high = self.target_ratios[comp]
                current_contrib = self.ema_contributions[i]
                
                # Deviation from target band
                gap_low = target_low - current_contrib
                gap_high = current_contrib - target_high
                
                # Hysteresis: adjust only outside tolerance
                if gap_low > self.tolerance:
                    # Below lower bound -> increase weight
                    adjustment = self.adjustment_rate * gap_low
                    self.log_weights[i] += adjustment
                elif gap_high > self.tolerance:
                    # Above upper bound -> decrease weight
                    adjustment = self.adjustment_rate * gap_high
                    self.log_weights[i] -= adjustment
                # Otherwise inside dead zone: no adjustment
            
            # Hard weight clipping
            self.log_weights.clamp_(
                min=np.log(self.weight_clip[0]),
                max=np.log(self.weight_clip[1])
            )
            
            # Renormalize to remove overall drift
            if self.renorm == 'mean_log':
                mean_log_weight = self.log_weights.mean()
                self.log_weights -= mean_log_weight
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights."""
        with torch.no_grad():
            current_weights = torch.exp(self.log_weights)
            return {comp: current_weights[i].item() for i, comp in enumerate(self.loss_components)}
    
    def get_target_ratios(self) -> Dict[str, Tuple[float, float]]:
        """Get target ratio ranges."""
        return self.target_ratios.copy()


def create_tcc_controller(loss_components: List[str], config: Dict) -> Optional[TargetContributionControl]:
    """
    Factory for TCC controller.

    Args:
        loss_components: Loss component names
        config: TCC config

    Returns:
        TCC instance or None
    """
    if not config.get('enabled', False):
        return None
    
    if len(loss_components) < 1:
        logger.warning("TCC requires at least 1 loss component")
        return None
    
    return TargetContributionControl(loss_components, config)
