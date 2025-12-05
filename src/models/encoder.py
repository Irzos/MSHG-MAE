"""
Encoder modules for HyperGraph-MAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from typing import Optional, List, Dict

from .layers.multi_scale import MultiScaleHypergraphConv
from .layers.attention import HypergraphAttentionAggregation


class HyperGATEncoder(nn.Module):
    """
    Hypergraph Attention Encoder.
    Encodes molecular hypergraphs into latent representations.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256,
                 proj_dim: int = 512, heads: int = 8, dropout: float = 0.3,
                 num_layers: int = 5, config: Optional[Dict] = None):
        """
        Initialize HyperGAT encoder.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            proj_dim: Projection head output dimension
            heads: Number of attention heads
            dropout: Dropout rate
            num_layers: Number of encoder layers
            config: Optional configuration dictionary
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.config = config or {}

        # Multi-scale convolution settings
        multi_scale_config = self.config.get('model', {}).get('multi_scale', {})
        scales = multi_scale_config.get('scales', [1])  # Default to single scale
        max_scale = multi_scale_config.get('max_scale', 1)


        # Build encoder layers
        self.conv_layers = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()

        for i in range(num_layers):
            # Use input dimension for first layer, hidden dimension for others
            in_channels = in_dim if i == 0 else hidden_dim

            # Multi-scale convolution layer
            self.conv_layers.append(
                MultiScaleHypergraphConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    scales=scales
                )
            )

            # Residual connection
            if in_channels != hidden_dim:
                self.res_linears.append(nn.Linear(in_channels, hidden_dim))
            else:
                self.res_linears.append(nn.Identity())

            # Normalization layers
            # Use LayerNorm instead of BatchNorm to avoid dimension issues
            self.batch_norms.append(nn.LayerNorm(hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

            # Attention aggregation layer
            edge_dim = self.config.get('features', {}).get('hyperedge_dim', 17)
            self.attn_layers.append(
                HypergraphAttentionAggregation(hidden_dim, heads=heads, max_dim=edge_dim)
            )

        # Projection head
        self.projection_head = self._build_projection_head(hidden_dim, proj_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_projection_head(self, in_dim: int, out_dim: int) -> nn.Module:
        """Build projection head network."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_attr: Optional[torch.Tensor] = None,
                hyperedge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Node features [num_nodes, in_dim]
            hyperedge_index: Hyperedge connectivity [2, num_connections]
            hyperedge_attr: Hyperedge attributes [num_edges, edge_dim]
            hyperedge_weights: Dynamic edge weights [num_edges, 1]

        Returns:
            Encoded features [num_nodes, proj_dim]
        """
        # Ensure consistent data types
        x = x.float()
        if hyperedge_index.numel() > 0:
            hyperedge_index = hyperedge_index.long()
        if hyperedge_attr is not None:
            hyperedge_attr = hyperedge_attr.float()
        if hyperedge_weights is not None:
            hyperedge_weights = hyperedge_weights.float()

        # Apply encoder layers
        for i in range(self.num_layers):
            # Store residual
            residual = x

            # Multi-scale convolution
            x_conv = self.conv_layers[i](x, hyperedge_index)
            # Inference: force FP32 to avoid Half/Float normalization dtype conflicts
            if (not self.training) and x_conv.dtype != torch.float32:
                x_conv = x_conv.float()
            # Ensure normalization layer dtype matches activation to avoid Half/Float conflicts
            bn = self.batch_norms[i]
            try:
                if hasattr(bn, 'weight') and bn.weight is not None and bn.weight.dtype != x_conv.dtype:
                    bn = bn.float()
                    # Also replace stored module to keep dtype consistent in later calls
                    self.batch_norms[i] = bn
            except Exception:
                pass
            x_conv = bn(x_conv)

            # Attention aggregation
            if hyperedge_attr is not None and hyperedge_weights is not None:
                x_attn = self.attn_layers[i](x_conv, hyperedge_index, hyperedge_attr, hyperedge_weights)
                x_conv = x_conv + self.dropout * x_attn

            # Residual connection and normalization
            # Match layer norm dtype with current activation
            ln = self.layer_norms[i]
            try:
                if hasattr(ln, 'weight') and ln.weight is not None and ln.weight.dtype != x.dtype:
                    ln = ln.float()
                    self.layer_norms[i] = ln
            except Exception:
                pass
            # Inference: also align residual to FP32 for consistency
            if (not self.training) and residual.dtype != torch.float32:
                residual = residual.float()
            x = ln(self.res_linears[i](residual) + x_conv)

            # Activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Projection head
        return self.projection_head(x)

    def get_embedding(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: Optional[torch.Tensor] = None,
                      hyperedge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings (for inference).

        Args:
            x: Node features
            hyperedge_index: Hyperedge connectivity
            hyperedge_attr: Hyperedge attributes
            hyperedge_weights: Optional dynamic edge weights; if provided, they will be used
                in the same way as in forward() to keep inference consistent with training.

        Returns:
            Normalized embeddings
        """
        self.eval()
        with torch.no_grad():
            # Ensure hyperedge_index is on the same device as x
            hyperedge_index = hyperedge_index.to(x.device)

            # Get embeddings (use weights if provided for consistency with training)
            embedding = self.forward(x, hyperedge_index, hyperedge_attr, hyperedge_weights)

        return embedding


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder with multiple resolution levels.
    """

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 heads: int = 8, dropout: float = 0.3):
        """
        Initialize hierarchical encoder.

        Args:
            in_dim: Input dimension
            hidden_dims: List of hidden dimensions for each level
            out_dim: Output dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_levels = len(hidden_dims)

        # Build hierarchical layers
        self.encoders = nn.ModuleList()
        in_channels = in_dim

        for hidden_dim in hidden_dims:
            self.encoders.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout
                )
            )
            in_channels = hidden_dim

        # Global pooling layer
        self.global_pool = GlobalHypergraphPooling()

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(sum(hidden_dims), out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with hierarchical encoding.

        Args:
            x: Node features
            hyperedge_index: Hyperedge connectivity
            batch: Batch assignment vector

        Returns:
            Hierarchical embeddings
        """
        level_embeddings = []
        current_x = x

        # Process through each level
        for encoder in self.encoders:
            current_x = encoder(current_x, hyperedge_index)

            # Pool at this level
            pooled = self.global_pool(current_x, batch)
            level_embeddings.append(pooled)

        # Concatenate all level embeddings
        combined = torch.cat(level_embeddings, dim=-1)

        # Final projection
        return self.final_proj(combined)


class EncoderBlock(nn.Module):
    """
    Single encoder block with convolution, normalization, and attention.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 dropout: float = 0.3):
        """
        Initialize encoder block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Convolution layer
        self.conv = MultiScaleHypergraphConv(
            in_channels, out_channels, heads, dropout
        )

        # Normalization and activation
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of encoder block.

        Args:
            x: Input features
            hyperedge_index: Hyperedge connectivity

        Returns:
            Encoded features
        """
        # Apply convolution
        out = self.conv(x, hyperedge_index)

        # Add residual connection
        out = out + self.residual(x)

        # Normalize and activate
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class GlobalHypergraphPooling(nn.Module):
    """
    Global pooling operation for hypergraphs.
    """

    def __init__(self, pooling_type: str = 'mean'):
        """
        Initialize global pooling.

        Args:
            pooling_type: Type of pooling ('mean', 'max', 'sum')
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply global pooling.

        Args:
            x: Node features [num_nodes, channels]
            batch: Batch assignment [num_nodes]

        Returns:
            Pooled features [num_graphs, channels]
        """
        if batch is None:
            # Single graph case
            if self.pooling_type == 'mean':
                return x.mean(dim=0, keepdim=True)
            elif self.pooling_type == 'max':
                return x.max(dim=0, keepdim=True)[0]
            elif self.pooling_type == 'sum':
                return x.sum(dim=0, keepdim=True)
        else:
            # Batched graphs
            num_graphs = batch.max().item() + 1
            pooled = []

            for i in range(num_graphs):
                mask = (batch == i)
                graph_x = x[mask]

                if self.pooling_type == 'mean':
                    pooled.append(graph_x.mean(dim=0))
                elif self.pooling_type == 'max':
                    pooled.append(graph_x.max(dim=0)[0])
                elif self.pooling_type == 'sum':
                    pooled.append(graph_x.sum(dim=0))

            return torch.stack(pooled)
