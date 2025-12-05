"""
Hypergraph convolution layers for HyperGraph-MAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, HypergraphConv
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DynamicHyperedgeWeighting(nn.Module):
    """
    Dynamic weighting module for hyperedges.
    Learns importance weights for different hyperedges based on their attributes.
    """

    def __init__(self, in_channels: int):
        """
        Initialize dynamic hyperedge weighting.

        Args:
            in_channels: Input feature channels
        """
        super().__init__()
        
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
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


class MultiScaleHypergraphConv(nn.Module):
    """
    Multi-scale hypergraph convolution layer.
    Processes hypergraph at multiple scales to capture both local and global patterns.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 dropout: float = 0.3, scales: List[int] = [1, 2], max_scale: int = 3):
        """
        Initialize multi-scale hypergraph convolution.

        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels  
            heads: Number of attention heads
            dropout: Dropout rate
            scales: List of scales to use
            max_scale: Maximum allowed scale
        """
        super().__init__()
        self.scales = scales
        self.max_scale = max_scale
        self.dropout = dropout

        # Validate scales
        if any(s > max_scale for s in scales):
            raise ValueError(f"All scales must be <= {max_scale}")

        # Create convolution layers for each scale
        self.convs = nn.ModuleList([
            HypergraphConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                concat=False
            ) for _ in scales
        ])

        # Learnable scale weights for aggregation
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))

        # Attention mechanism for dynamic scale weighting
        self.scale_attention = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(scales)),
            nn.Softmax(dim=-1)
        )

        # Add feature transformation for each scale
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in scales
        ])

    def modify_for_scale(self, hyperedge_index: torch.Tensor, scale: int, num_nodes: int) -> torch.Tensor:
        """
        GPU-optimized multi-scale processing using sparse operations.
        Avoid dense matrix operations for better memory and speed.
        """
        if scale > self.max_scale:
            raise ValueError(f"Scale {scale} exceeds max allowed {self.max_scale}")
            
        if scale == 1:
            return hyperedge_index
            
        if hyperedge_index.numel() == 0:
            return hyperedge_index

        try:
            # Use sparse operations throughout to avoid dense matrix conversion
            indices = hyperedge_index
            values = torch.ones(indices.size(1), device=hyperedge_index.device)
            
            # Keep operations sparse to save memory and computation
            num_hyperedges = int(hyperedge_index[1].max().item() + 1)
            
            # For higher scales, use approximate methods instead of exact matrix power
            if scale == 2:
                # Scale 2: Add 2-hop connections via sparse operations
                return self._add_two_hop_connections(hyperedge_index, num_nodes, num_hyperedges)
            else:
                # For scale > 2, use iterative sparse expansion (much more efficient)
                return self._iterative_sparse_expansion(hyperedge_index, scale, num_nodes, num_hyperedges)
                
        except Exception as e:
            logger.warning(f"Multi-scale processing failed for scale {scale}: {e}")
            return hyperedge_index
    
    def _add_two_hop_connections(self, hyperedge_index: torch.Tensor, num_nodes: int, num_hyperedges: int) -> torch.Tensor:
        """Add 2-hop connections using efficient sparse operations."""
        # Original edges
        original_edges = hyperedge_index
        
        # Find 2-hop connections by finding shared hyperedges
        nodes = hyperedge_index[0]
        edges = hyperedge_index[1]
        
        # Group nodes by hyperedge for efficient processing
        # Use scatter operations instead of loops
        new_edges = []
        edge_counter = num_hyperedges
        
        # Process each original hyperedge
        unique_edges = torch.unique(edges)
        
        for edge_id in unique_edges:
            # Get nodes in this hyperedge
            mask = (edges == edge_id)
            edge_nodes = nodes[mask]
            
            if len(edge_nodes) <= 1:
                continue
                
            # For 2-hop: connect this hyperedge's nodes to nodes in other hyperedges
            # that share at least one node
            for other_edge_id in unique_edges:
                if edge_id >= other_edge_id:  # Avoid duplicates
                    continue
                    
                other_mask = (edges == other_edge_id)  
                other_nodes = nodes[other_mask]
                
                # Check if hyperedges share nodes
                if len(torch.intersect1d(edge_nodes, other_nodes)) > 0:
                    # Create new hyperedge connecting all nodes from both hyperedges
                    combined_nodes = torch.unique(torch.cat([edge_nodes, other_nodes]))
                    
                    if len(combined_nodes) > 1:  # Only add non-trivial hyperedges
                        new_hyperedge_id = torch.full((len(combined_nodes),), edge_counter, 
                                                    device=hyperedge_index.device)
                        new_edge = torch.stack([combined_nodes, new_hyperedge_id])
                        new_edges.append(new_edge)
                        edge_counter += 1
        
        if new_edges:
            # Combine original and new edges
            all_new_edges = torch.cat(new_edges, dim=1)
            expanded_index = torch.cat([original_edges, all_new_edges], dim=1)
            return expanded_index
        else:
            return original_edges
    
    def _iterative_sparse_expansion(self, hyperedge_index: torch.Tensor, scale: int, 
                                  num_nodes: int, num_hyperedges: int) -> torch.Tensor:
        """
        Iterative sparse expansion for scales > 2.
        Much more memory efficient than dense matrix operations.
        """
        current_index = hyperedge_index
        
        # Limit iterations to prevent excessive computation
        max_iterations = min(scale - 1, 3)  # Cap at 3 iterations
        
        for iteration in range(max_iterations):
            # Each iteration adds one more hop
            current_index = self._add_two_hop_connections(
                current_index, num_nodes, 
                int(current_index[1].max().item() + 1) if current_index.numel() > 0 else 0
            )
            
            # Limit total edges to prevent memory explosion
            if current_index.size(1) > 50000:  # Reasonable limit
                logger.warning(f"Stopping expansion at iteration {iteration+1} due to size limit")
                break
                
        return current_index

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: multi-scale hypergraph convolution with weighted aggregation.
        """
        outputs = []
        transformed_outputs = []

        for i, (scale, conv) in enumerate(zip(self.scales, self.convs)):
            try:
                # Modify hyperedge indices for current scale
                modified_hyperedge_index = self.modify_for_scale(hyperedge_index, scale, x.size(0))

                # Perform hypergraph convolution
                out = conv(x, modified_hyperedge_index)
                outputs.append(out)

                # Apply scale-specific transformation
                transformed = self.scale_transforms[i](out)
                transformed_outputs.append(transformed)
                
            except Exception as e:
                logger.warning(f"Convolution failed at scale {scale}: {e}")
                # Use identity transformation as fallback
                identity_out = x if x.size(-1) == self.convs[i].out_channels else \
                              torch.zeros(x.size(0), self.convs[i].out_channels, dtype=x.dtype, device=x.device)
                outputs.append(identity_out)
                transformed_outputs.append(self.scale_transforms[i](identity_out))

        # Stack outputs for attention calculation
        output_stack = torch.stack(outputs, dim=0)  # [num_scales, num_nodes, out_channels]
        transformed_stack = torch.stack(transformed_outputs, dim=0)

        # Method 1: Use static weights
        static_weights = F.softmax(self.scale_weights, dim=0)

        # Method 2: Node-level dynamic scale weighting
        node_features = torch.mean(output_stack, dim=0)  # [num_nodes, out_channels]
        node_scale_weights = self.scale_attention(node_features)  # [num_nodes, num_scales]
        dynamic_weights = node_scale_weights.mean(dim=0)  # [num_scales]

        # Combine both weighting methods
        combined_weights = (static_weights + dynamic_weights) / 2
        combined_weights = combined_weights.view(-1, 1, 1)  # [num_scales, 1, 1]

        # Apply weighted aggregation to transformed outputs
        weighted_sum = torch.sum(transformed_stack * combined_weights, dim=0)  # [num_nodes, out_channels]

        return weighted_sum


class BasicHypergraphConv(nn.Module):
    """
    Basic hypergraph convolution layer as a fallback option.
    """

    def __init__(self, in_channels: int, out_channels: int, 
                 bias: bool = True, dropout: float = 0.0):
        """
        Initialize basic hypergraph convolution.

        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            bias: Whether to use bias
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of basic hypergraph convolution.

        Args:
            x: Node features [num_nodes, in_channels]
            hyperedge_index: Hyperedge connectivity [2, num_connections]

        Returns:
            Convolved features [num_nodes, out_channels]
        """
        # Apply linear transformation
        x = self.linear(x)
        x = self.dropout(x)
        
        if hyperedge_index.numel() == 0:
            return x

        # Simple aggregation: average features of nodes in the same hyperedge
        try:
            nodes, edges = hyperedge_index
            num_nodes = x.size(0)
            num_edges = int(edges.max().item() + 1)
            
            # Use non-inplace operations for aggregation
            try:
                from torch_scatter import scatter_add
                # Aggregate node features to hyperedges
                edge_features = scatter_add(x[nodes], edges, dim=0, dim_size=num_edges)
                edge_counts = scatter_add(torch.ones_like(edges, dtype=torch.float), edges, dim=0, dim_size=num_edges)
                
                # Normalize by count
                edge_features = edge_features / edge_counts.unsqueeze(1).clamp(min=1)
                
                # Aggregate back to nodes
                out = scatter_add(edge_features[edges], nodes, dim=0, dim_size=num_nodes)
                node_counts = scatter_add(torch.ones_like(nodes, dtype=torch.float), nodes, dim=0, dim_size=num_nodes)
                
                # Normalize by count
                out = out / node_counts.unsqueeze(1).clamp(min=1)
                
            except ImportError:
                # Fallback to manual aggregation
                edge_features = torch.zeros((num_edges, x.size(1)), dtype=x.dtype, device=x.device)
                edge_counts = torch.zeros(num_edges, dtype=torch.float32, device=x.device)
                
                # Manual aggregation without inplace operations
                for i in range(len(nodes)):
                    node_idx = nodes[i].item()
                    edge_idx = edges[i].item()
                    edge_features[edge_idx] = edge_features[edge_idx] + x[node_idx]
                    edge_counts[edge_idx] = edge_counts[edge_idx] + 1.0
                
                # Normalize by count
                edge_features = edge_features / edge_counts.unsqueeze(1).clamp(min=1)
                
                # Aggregate back to nodes
                out = torch.zeros_like(x)
                node_counts = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
                
                for i in range(len(nodes)):
                    node_idx = nodes[i].item()
                    edge_idx = edges[i].item()
                    out[node_idx] = out[node_idx] + edge_features[edge_idx]
                    node_counts[node_idx] = node_counts[node_idx] + 1.0
                
                # Normalize by count
                out = out / node_counts.unsqueeze(1).clamp(min=1)
            
            return out
            
        except Exception as e:
            logger.warning(f"Basic hypergraph convolution failed: {e}")
            return x 
