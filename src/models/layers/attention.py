"""
Attention mechanisms for hypergraph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HypergraphAttentionAggregation(nn.Module):
    """
    Enhanced hypergraph attention aggregation based on oldcode implementation.
    Includes multi-head attention, cross attention, and edge feature integration.
    """

    def __init__(self, in_dim: int, heads: int = 4, max_dim: int = 17):
        """
        Initialize enhanced hypergraph attention aggregation.

        Args:
            in_dim: Input feature dimension
            heads: Number of attention heads
            max_dim: Maximum dimension for edge features
        """
        super().__init__()
        
        # Validate dimensions
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if in_dim % heads != 0:
            raise ValueError(f"in_dim must be divisible by heads")

        self.heads = heads
        self.max_dim = max_dim
        
        # Multi-head attention components (enhanced from oldcode)
        self.node_attn = nn.MultiheadAttention(in_dim, heads)
        self.edge_attn = nn.MultiheadAttention(in_dim, heads)
        self.cross_attn = nn.MultiheadAttention(in_dim, heads)  # Cross attention like oldcode
        
        # Projection layers
        self.edge_proj = nn.Linear(max_dim, in_dim)
        self.node_proj = nn.Linear(in_dim, in_dim)
        
        # Learnable mixing parameter (like oldcode)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

        # Force use of stable math SDPA kernel; disable Flash/MemEff to avoid mixed-precision instability
        try:
            import torch.backends.cuda as cuda_backends
            if hasattr(cuda_backends, "enable_flash_sdp"):
                cuda_backends.enable_flash_sdp(False)
            if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
                cuda_backends.enable_mem_efficient_sdp(False)
            if hasattr(cuda_backends, "enable_math_sdp"):
                cuda_backends.enable_math_sdp(True)
        except Exception:
            # Be permissive on CPU / older Torch versions
            pass

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_attr: Optional[torch.Tensor] = None,
                hyperedge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass based on oldcode implementation.
        """
        if hyperedge_index.numel() == 0:
            return x

        # Use sigmoid to constrain alpha in [0,1] range (like oldcode)
        alpha = torch.sigmoid(self.alpha_raw)

        # Node self-attention
        x_proj = self.node_proj(x)
        node_attn_out, _ = self.node_attn(
            x_proj.unsqueeze(1),
            x_proj.unsqueeze(1), 
            x_proj.unsqueeze(1)
        )
        node_attn_out = node_attn_out.squeeze(1)

        # Edge-aware processing (enhanced from oldcode)
        edge_messages = torch.zeros_like(x)
        if hyperedge_attr is not None and hyperedge_attr.shape[0] > 0:
            # Project edge features to match node dimension
            edge_features = self.edge_proj(hyperedge_attr)  # [E, in_dim]

            # Edge self-attention
            edge_attn_out, _ = self.edge_attn(
                edge_features.unsqueeze(1),
                edge_features.unsqueeze(1),
                edge_features.unsqueeze(1)
            )
            edge_attn_out = edge_attn_out.squeeze(1)  # [E, in_dim]

            # Node-edge cross attention (like oldcode)
            cross_attn, _ = self.cross_attn(
                query=x_proj.unsqueeze(1),  # [N, 1, in_dim]
                key=edge_attn_out.unsqueeze(1),  # [E, 1, in_dim] 
                value=edge_attn_out.unsqueeze(1)
            )
            cross_attn = cross_attn.squeeze(1)  # [N, in_dim]

            # Message passing based on hyperedge structure (like oldcode)
            nodes, edges = hyperedge_index
            if len(nodes) > 0:
                # Use attention-weighted edge features
                weighted_edges = edge_attn_out[edges]  # [M, in_dim]

                # Apply edge weights if available
                if hyperedge_weights is not None:
                    weights = hyperedge_weights[edges].squeeze(-1)  # [M]
                    weighted_edges = weighted_edges * weights.unsqueeze(-1)

                # Efficient aggregation using scatter_add
                edge_messages = torch.zeros_like(x)
                edge_messages.scatter_add_(
                    dim=0,
                    index=nodes.unsqueeze(1).expand(-1, x.size(1)),
                    src=weighted_edges.to(edge_messages.dtype)
                )

            # Combine node self-attention and cross-attention (like oldcode)
            combined = alpha * node_attn_out + (1 - alpha) * cross_attn
        else:
            combined = node_attn_out

        # Final combination with edge messages
        final_out = combined + edge_messages

        # L2 normalization with explicit eps to avoid tiny-norm blowups
        return F.normalize(final_out, p=2, dim=-1, eps=1e-6)


class DynamicHyperedgeWeighting(nn.Module):
    """
    Dynamic weighting module for hyperedges.
    Learns importance weights for different hyperedges.
    """

    def __init__(self, in_channels: int):
        """
        Initialize dynamic hyperedge weighting.

        Args:
            in_channels: Input feature channels
        """
        super().__init__()

        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hyperedge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights for hyperedges.

        Args:
            hyperedge_attr: Hyperedge attributes [num_edges, in_channels]

        Returns:
            Edge weights [num_edges, 1]
        """
        return self.weight_net(hyperedge_attr)


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for standard graph structures.
    Can be used as a component in hypergraph models.
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 8,
                 dropout: float = 0.1, concat: bool = True):
        """
        Initialize graph attention layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate head outputs
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations for each head
        self.W = nn.Parameter(torch.empty(size=(in_features, heads * out_features)))
        self.a = nn.Parameter(torch.empty(size=(1, heads, 2 * out_features)))

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph attention layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Output features [num_nodes, heads * out_features] or [num_nodes, out_features]
        """
        # Linear transformation
        h = torch.mm(x, self.W).view(-1, self.heads, self.out_features)

        # Attention mechanism
        edge_h = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=-1)
        edge_e = torch.sum(self.a * edge_h, dim=-1)

        # Compute attention coefficients
        attention = self._compute_attention_coefficients(edge_e, edge_index, x.size(0))
        attention = self.dropout_layer(attention)

        # Apply attention to features
        h_prime = self._apply_attention(h, attention, edge_index)

        if self.concat:
            return h_prime.view(-1, self.heads * self.out_features)
        else:
            return h_prime.mean(dim=1)

    def _compute_attention_coefficients(self, edge_e: torch.Tensor,
                                        edge_index: torch.Tensor,
                                        num_nodes: int) -> torch.Tensor:
        """Compute normalized attention coefficients."""
        # Create sparse attention matrix
        row, col = edge_index
        attention = torch.zeros((num_nodes, num_nodes, self.heads),
                                device=edge_e.device)
        attention[row, col] = edge_e

        # Apply softmax
        attention = F.softmax(attention, dim=1)

        return attention

    def _apply_attention(self, h: torch.Tensor, attention: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
        """Apply attention weights to features."""
        # Message passing with attention
        row, col = edge_index
        out = torch.zeros_like(h)

        for i in range(h.size(0)):
            # Get neighbors
            mask = (row == i)
            if mask.any():
                neighbors = col[mask]
                attn_weights = attention[i, neighbors].unsqueeze(-1)
                neighbor_features = h[neighbors]
                out[i] = torch.sum(attn_weights * neighbor_features, dim=0)

        return F.relu(out)
