"""
Multi-scale hypergraph processing layers - simplified based on oldcode approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.amp import autocast
import torch_scatter


class VectorizedHypergraphConv(nn.Module):
    """
    GPU-optimized hypergraph convolution with vectorized message passing.
    Replaces Python loops with scatter/gather operations for better GPU utilization.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 dropout: float = 0.0, concat: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations
        self.lin_node = nn.Linear(in_channels, heads * out_channels)
        
        # Output projection
        if concat:
            self.out_proj = nn.Linear(heads * out_channels, out_channels)
        else:
            self.out_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_node.weight)
        if self.out_proj is not None:
            nn.init.xavier_uniform_(self.out_proj.weight)

    @autocast('cuda', enabled=True)
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized forward pass with vectorized operations.
        """
        if hyperedge_index.numel() == 0:
            return self.lin_node(x)

        # Transform node features
        x_transformed = self.lin_node(x)  # [num_nodes, heads * out_channels]
        
        # Get edge information
        node_idx, edge_idx = hyperedge_index
        
        # GPU-optimized: avoid .item() sync
        if edge_idx.numel() == 0:
            return x_transformed if not self.concat or self.out_proj is None else self.out_proj(x_transformed)

        # Vectorized message passing using scatter operations
        num_edges = int(edge_idx.max().item() + 1) if edge_idx.numel() > 0 else 0
        num_nodes = x.size(0)
        
        # Step 1: Aggregate node features to hyperedges
        edge_features = torch_scatter.scatter_mean(
            x_transformed[node_idx], edge_idx, dim=0, dim_size=num_edges
        )  # [num_edges, heads * out_channels]
        
        # Step 2: Broadcast edge features back to nodes
        output = torch_scatter.scatter_add(
            edge_features[edge_idx], node_idx, dim=0, dim_size=num_nodes
        )  # [num_nodes, heads * out_channels]

        # Apply dropout
        if self.training:
            output = F.dropout(output, p=self.dropout)

        # Final projection if needed
        if self.concat and self.out_proj is not None:
            output = self.out_proj(output)

        return output


class ParallelMultiScaleHypergraphConv(nn.Module):
    """
    GPU-optimized parallel multi-scale hypergraph convolution.
    All scales are processed simultaneously for maximum GPU utilization.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 dropout: float = 0.3, scales: Optional[List[int]] = None):
        super().__init__()
        
        if scales is None:
            scales = [1, 2]
        
        # Store as regular attributes, not parameters
        self.scales = list(scales)  # Ensure it's a list, not tensor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # Create convolution layers for each scale
        self.convs = nn.ModuleList([
            VectorizedHypergraphConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                concat=False
            ) for _ in scales
        ])

        # Output projection - should match the actual output dimension from BasicHypergraphConv
        # BasicHypergraphConv outputs heads * out_channels, so input should be that
        self.output_proj = nn.Linear(heads * out_channels, out_channels)

    @autocast('cuda', enabled=True)
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized parallel multi-scale processing.
        """
        if hyperedge_index.numel() == 0:
            return torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
            
        # Prepare all scale transformations in parallel
        modified_indices = []
        for scale in self.scales:
            modified_idx = self._modify_scale_vectorized(hyperedge_index, scale, x.size(0))
            modified_indices.append(modified_idx)
        
        # Process all scales in parallel using vmap
        if len(self.scales) == 1:
            # Single scale - direct processing
            output = self.convs[0](x, modified_indices[0])
            return self.output_proj(output)
        
        # Multiple scales - parallel processing
        outputs = []
        for i, (conv, modified_idx) in enumerate(zip(self.convs, modified_indices)):
            with autocast('cuda', enabled=True):
                out = conv(x, modified_idx)
                outputs.append(out)
        
        # Efficient tensor stacking and aggregation
        if len(outputs) > 1:
            output_stack = torch.stack(outputs, dim=0)  # [num_scales, num_nodes, features]
            # Use more efficient aggregation
            weighted_sum = torch.mean(output_stack, dim=0)
        else:
            weighted_sum = outputs[0]

        return self.output_proj(weighted_sum)

    def _modify_scale_vectorized(self, hyperedge_index: torch.Tensor, scale: int, num_nodes: int) -> torch.Tensor:
        """
        GPU-optimized scale transformation using vectorized operations.
        """
        if scale == 1 or hyperedge_index.numel() == 0:
            return hyperedge_index
            
        # Limit maximum scale
        max_scale = 3
        if scale > max_scale:
            scale = max_scale

        # GPU-optimized: avoid .item() sync
        if hyperedge_index[1].numel() == 0:
            return hyperedge_index

        # Calculate num_hyperedges from hyperedge_index with additional safety checks
        if hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            num_hyperedges = 0
        else:
            # Calculate num_hyperedges safely with .item() sync
            num_hyperedges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index[1].numel() > 0 else 0

        # Use sparse operations to avoid dense conversions when possible
        with autocast('cuda', enabled=False):  # Keep sparse ops in FP32 for stability
            indices = hyperedge_index
            values = torch.ones(indices.size(1), dtype=torch.float32, device=hyperedge_index.device)
            H_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_hyperedges))
            
            # Efficient scale transformation using sparse operations
            current_H = H_sparse
            for _ in range(scale - 1):
                # H @ H^T @ H pattern using sparse operations
                HT_H = torch.sparse.mm(current_H.t(), current_H)  # [num_hyperedges, num_hyperedges]
                new_H = torch.sparse.mm(current_H, HT_H)  # [num_nodes, num_hyperedges]
                current_H = new_H.coalesce()
            
            # Convert back to edge index format
            indices = current_H.indices()
            # Filter out zero values
            values = current_H.values()
            mask = values > 0
            new_hyperedge_index = indices[:, mask]
            
            return new_hyperedge_index.contiguous()


# Backward compatibility aliases
BasicHypergraphConv = VectorizedHypergraphConv
MultiScaleHypergraphConv = ParallelMultiScaleHypergraphConv