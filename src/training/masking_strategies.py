"""
Minimal masking strategy module.

This module implements the essential baseline strategy:
- Random masking (baseline)

Note: Semantic masking is implemented in `semantic_masking.py`.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from rdkit import Chem
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class MaskingStrategy(ABC):
    """Abstract base class for masking strategies."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.mask_ratio = self.config.get('mask_ratio', 0.4)
        self.device = None
        
    @abstractmethod
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                       hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate node and hyperedge masks.

        Args:
            x: Node features [num_nodes, feature_dim]
            hyperedge_index: Hyperedge incidence [2, num_connections]
            hyperedge_attr: Hyperedge attributes [num_edges, edge_dim]

        Returns:
            Tuple of (node_mask, edge_mask)
        """
        pass
    
    def set_device(self, device: torch.device):
        """Set target device for internal buffers (if any)."""
        self.device = device


class RandomMasking(MaskingStrategy):
    """Basic random masking strategy."""
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                       hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random node and hyperedge masks."""
        num_nodes = x.size(0)
        num_edges = hyperedge_attr.size(0) if hyperedge_attr is not None else 0

        device = x.device

        node_mask = torch.rand(num_nodes, device=device) < self.mask_ratio

        if num_edges > 0:
            edge_mask = torch.rand(num_edges, device=device) < self.mask_ratio
        else:
            edge_mask = torch.empty(0, dtype=torch.bool, device=device)
            
        return node_mask, edge_mask





def create_masking_strategy(strategy_type: str, config: Dict = None) -> MaskingStrategy:
    """Factory function for masking strategies (minimal: random only)."""
    strategy_map = {
        'random': RandomMasking,
    }
    
    if strategy_type not in strategy_map:
        # Unknown strategy; fall back to random
        logger.warning(f"Unknown masking strategy: {strategy_type}. Falling back to 'random'.")
        strategy_type = 'random'
    
    return strategy_map[strategy_type](config)
