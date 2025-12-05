"""
Decoder modules for HyperGraph-MAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class LightweightDecoder(nn.Module):
    """
    Lightweight decoder for efficient reconstruction.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize lightweight decoder.

        Args:
            in_dim: Input dimension (from encoder)
            out_dim: Output dimension (original feature size)
            hidden_dim: Hidden layer dimension
            num_layers: Number of decoder layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers

        # Build decoder layers
        layers = []
        current_dim = in_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(current_dim, out_dim))

        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations.

        Args:
            x: Latent features [num_nodes, in_dim]

        Returns:
            Reconstructed features [num_nodes, out_dim]
        """
        return self.decoder(x)


class HypergraphDecoder(nn.Module):
    """
    Hypergraph-aware decoder that considers structure during reconstruction.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 heads: int = 4, dropout: float = 0.1):
        """
        Initialize hypergraph decoder.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Hidden dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Initial projection
        self.init_proj = nn.Linear(in_dim, hidden_dim)

        # Self-attention for structure-aware decoding
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor,
                hyperedge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode with optional structure awareness.

        Args:
            x: Latent features
            hyperedge_index: Optional hyperedge connectivity

        Returns:
            Reconstructed features
        """
        # Initial projection
        x = self.init_proj(x)

        # Self-attention
        attn_out, _ = self.self_attn(
            x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1)
        )
        x = self.norm1(x + attn_out.squeeze(1))

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Output projection
        return self.output_proj(x)


class EdgeDecoder(nn.Module):
    """
    Decoder specifically for hyperedge attribute reconstruction.
    """

    def __init__(self, in_dim: int, edge_attr_dim: int = 17,
                 hidden_dim: int = 128):
        """
        Initialize edge decoder.

        Args:
            in_dim: Input dimension (aggregated node features)
            edge_attr_dim: Dimension of edge attributes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_attr_dim)
        )

    def forward(self, node_features: torch.Tensor,
                hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode edge attributes from node features.

        Args:
            node_features: Node features [num_nodes, in_dim]
            hyperedge_index: Hyperedge connectivity [2, num_connections]

        Returns:
            Predicted edge attributes [num_edges, edge_attr_dim]
        """
        if hyperedge_index.numel() == 0:
            return torch.zeros((0, self.decoder[-1].out_features),
                               dtype=node_features.dtype, device=node_features.device)

        # Safe edge count calculation with boundary checking
        if hyperedge_index.size(1) == 0:
            return torch.zeros((0, self.decoder[-1].out_features),
                               dtype=node_features.dtype, device=node_features.device)
        
        # GPU-optimized: check validity without .item() sync
        if hyperedge_index[1].numel() == 0 or (hyperedge_index[1] < 0).any():
            return torch.zeros((0, self.decoder[-1].out_features),
                               dtype=node_features.dtype, device=node_features.device)
        
        # Calculate number of edges properly (avoiding .item() for GPU efficiency)
        num_edges = int(hyperedge_index[1].max() + 1) if hyperedge_index[1].numel() > 0 else 0
        edge_features = self._aggregate_to_edges(
            node_features, hyperedge_index, num_edges
        )

        # Decode edge attributes
        return self.decoder(edge_features)

    def _aggregate_to_edges(self, node_features: torch.Tensor,
                            hyperedge_index: torch.Tensor,
                            num_edges: int) -> torch.Tensor:
        """
        Aggregate node features to hyperedges.

        Args:
            node_features: Node features
            hyperedge_index: Connectivity
            num_edges: Number of edges

        Returns:
            Aggregated edge features
        """
        if num_edges == 0:
            return torch.zeros((0, node_features.size(1)), dtype=node_features.dtype, device=node_features.device)
        
        edge_features = torch.zeros(
            (num_edges, node_features.size(1)),
            device=node_features.device,
            dtype=node_features.dtype
        )
        counts = torch.zeros(num_edges, dtype=torch.float32, device=node_features.device)

        # Aggregate features with boundary checking (non-inplace operations)
        nodes, edges = hyperedge_index
        
        # Filter out invalid indices
        valid_mask = (nodes >= 0) & (nodes < node_features.size(0)) & (edges >= 0) & (edges < num_edges)
        if not valid_mask.any():
            return edge_features
        
        valid_nodes = nodes[valid_mask]
        valid_edges = edges[valid_mask]
        
        # Use non-inplace operations with torch_scatter (required)
        from torch_scatter import scatter_add
        edge_features = scatter_add(node_features[valid_nodes], valid_edges, dim=0, dim_size=num_edges)
        counts = scatter_add(torch.ones_like(valid_edges, dtype=torch.float), valid_edges, dim=0, dim_size=num_edges)

        # Normalize by count
        edge_features = edge_features / counts.unsqueeze(1).clamp(min=1)

        return edge_features


class VariationalDecoder(nn.Module):
    """
    Variational decoder with stochastic reconstruction.
    """

    def __init__(self, in_dim: int, out_dim: int, latent_dim: int = 128):
        """
        Initialize variational decoder.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            latent_dim: Latent variable dimension
        """
        super().__init__()

        # Encoder to latent distribution
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent distribution parameters.

        Args:
            x: Input features

        Returns:
            Mean and log-variance of latent distribution
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling.

        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution

        Returns:
            Sampled latent variable
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with variational decoding.

        Args:
            x: Input features

        Returns:
            Reconstructed features, mean, and log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute variational loss.

        Args:
            recon: Reconstructed features
            target: Target features
            mu: Latent mean
            logvar: Latent log-variance
            beta: KL divergence weight

        Returns:
            Total loss and component losses
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, target, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class AdaptiveDecoder(nn.Module):
    """
    Adaptive decoder that adjusts complexity based on input.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 min_layers: int = 1, max_layers: int = 3):
        """
        Initialize adaptive decoder.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            min_layers: Minimum number of layers
            max_layers: Maximum number of layers
        """
        super().__init__()
        self.min_layers = min_layers
        self.max_layers = max_layers

        # Build decoder blocks
        self.blocks = nn.ModuleList()
        current_dim = in_dim

        for i in range(max_layers):
            if i < max_layers - 1:
                block = nn.Sequential(
                    nn.Linear(current_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                )
                current_dim = 256
            else:
                block = nn.Linear(current_dim, out_dim)

            self.blocks.append(block)

        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_layers - min_layers + 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptively decode based on input complexity.

        Args:
            x: Input features

        Returns:
            Decoded features
        """
        # Predict complexity
        complexity_scores = self.complexity_predictor(x.mean(dim=0, keepdim=True))
        num_layers = self.min_layers + complexity_scores.argmax().item()

        # Apply selected number of layers
        out = x
        for i in range(num_layers):
            out = self.blocks[i](out)

        # Apply final layer if not already applied
        if num_layers < self.max_layers:
            out = self.blocks[-1](out)

        return out
