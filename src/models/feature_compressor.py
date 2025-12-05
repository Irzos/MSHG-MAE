"""
Feature compressor module for hypergraph features.
Provides dimension compression for full feature vectors from preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureCompressor(nn.Module):
    """
    Compresses full-dimensional features to target dimensions for model training.
    """
    
    def __init__(self, input_dims: Dict[str, int], target_dim: int = 64, dropout: float = 0.1):
        """
        Initialize feature compressor.
        
        Args:
            input_dims: Dictionary mapping feature type to input dimension
                      e.g., {'atom': 25, 'bond': 20, 'ring': 7, 'functional_group': 30, 'hydrogen_bond': 15}
            target_dim: Target dimension for compressed features
            dropout: Dropout probability
        """
        super(FeatureCompressor, self).__init__()
        
        self.input_dims = input_dims
        self.target_dim = target_dim
        self.dropout = dropout
        
        # Create separate linear layers for each feature type
        self.compressors = nn.ModuleDict()
        
        for feature_type, input_dim in input_dims.items():
            if input_dim > 0:
                self.compressors[feature_type] = nn.Sequential(
                    nn.Linear(input_dim, target_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(target_dim * 2, target_dim),
                    nn.LayerNorm(target_dim)
                )
                logger.info(f"Created compressor for {feature_type}: {input_dim} -> {target_dim}")
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor, feature_type: str) -> torch.Tensor:
        """
        Compress features to target dimension.
        
        Args:
            features: Input features of shape [batch_size, input_dim] or [num_items, input_dim]
            feature_type: Type of features (must be in input_dims)
            
        Returns:
            Compressed features of shape [batch_size, target_dim] or [num_items, target_dim]
        """
        if feature_type not in self.compressors:
            raise ValueError(f"Unknown feature type: {feature_type}. Available types: {list(self.compressors.keys())}")
        
        # Handle empty features
        if features.numel() == 0:
            return torch.empty(0, self.target_dim, device=features.device, dtype=features.dtype)
        
        # Ensure input dimension matches expected
        expected_dim = self.input_dims[feature_type]
        if features.shape[-1] != expected_dim:
            logger.warning(f"Feature dimension mismatch for {feature_type}: expected {expected_dim}, got {features.shape[-1]}")
            # Pad or truncate to match expected dimension
            if features.shape[-1] < expected_dim:
                padding = torch.zeros(*features.shape[:-1], expected_dim - features.shape[-1], 
                                    device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :expected_dim]
        
        return self.compressors[feature_type](features)
    
    def compress_atom_features(self, atom_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for compressing atom features."""
        return self.forward(atom_features, 'atom')
    
    def compress_bond_features(self, bond_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for compressing bond features."""
        return self.forward(bond_features, 'bond')
    
    def compress_ring_features(self, ring_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for compressing ring features."""
        return self.forward(ring_features, 'ring')
    
    def compress_functional_group_features(self, fg_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for compressing functional group features."""
        return self.forward(fg_features, 'functional_group')
    
    def compress_hydrogen_bond_features(self, hb_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for compressing hydrogen bond features."""
        return self.forward(hb_features, 'hydrogen_bond')
    
    def get_output_dim(self) -> int:
        """Get the output dimension of compressed features."""
        return self.target_dim


class AdaptiveFeatureCompressor(nn.Module):
    """
    Adaptive feature compressor that can handle variable input dimensions.
    """
    
    def __init__(self, max_input_dim: int = 100, target_dim: int = 64, 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize adaptive feature compressor.
        
        Args:
            max_input_dim: Maximum expected input dimension
            target_dim: Target output dimension
            num_heads: Number of attention heads for adaptive compression
            dropout: Dropout probability
        """
        super(AdaptiveFeatureCompressor, self).__init__()
        
        self.max_input_dim = max_input_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        
        # Dimension projection layer
        self.input_projection = nn.Linear(max_input_dim, target_dim * 2)
        
        # Multi-head attention for adaptive compression
        self.attention = nn.MultiheadAttention(
            embed_dim=target_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(target_dim * 2, target_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(target_dim)
        )
        
        # Learnable query for compression
        self.compression_query = nn.Parameter(torch.randn(1, target_dim * 2))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compress features adaptively.
        
        Args:
            features: Input features of shape [batch_size, input_dim]
            
        Returns:
            Compressed features of shape [batch_size, target_dim]
        """
        batch_size, input_dim = features.shape
        
        # Pad or truncate to max_input_dim
        if input_dim < self.max_input_dim:
            padding = torch.zeros(batch_size, self.max_input_dim - input_dim,
                                device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=-1)
        elif input_dim > self.max_input_dim:
            features = features[:, :self.max_input_dim]
        
        # Project to intermediate dimension
        projected = self.input_projection(features)  # [batch_size, target_dim * 2]
        
        # Expand query for batch
        query = self.compression_query.expand(batch_size, -1, -1)  # [batch_size, 1, target_dim * 2]
        
        # Use projected features as both key and value
        kv = projected.unsqueeze(1)  # [batch_size, 1, target_dim * 2]
        
        # Apply attention-based compression
        compressed, _ = self.attention(query, kv, kv)  # [batch_size, 1, target_dim * 2]
        compressed = compressed.squeeze(1)  # [batch_size, target_dim * 2]
        
        # Final projection to target dimension
        output = self.output_projection(compressed)  # [batch_size, target_dim]
        
        return output


def create_feature_compressor(config: Dict, feature_dims: Optional[Dict[str, int]] = None) -> FeatureCompressor:
    """
    Create feature compressor based on configuration.
    
    Args:
        config: Configuration dictionary
        feature_dims: Optional feature dimensions dictionary
        
    Returns:
        Configured feature compressor
    """
    compressor_config = config.get('feature_compressor', {})
    
    # Default feature dimensions (can be overridden)
    default_dims = {
        'atom': 25,  # Estimated from atom features
        'bond': 20,  # Estimated from bond features  
        'ring': 7,   # From ring features
        'functional_group': 30,  # Estimated from functional group features
        'hydrogen_bond': 15,  # From hydrogen bond features
        'conjugated_system': 6   # From conjugated system features
    }
    
    if feature_dims:
        default_dims.update(feature_dims)
    
    target_dim = compressor_config.get('target_dim', 64)
    dropout = compressor_config.get('dropout', 0.1)
    
    return FeatureCompressor(
        input_dims=default_dims,
        target_dim=target_dim,
        dropout=dropout
    )