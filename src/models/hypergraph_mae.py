"""
Main HyperGraph-MAE model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

from .encoder import HyperGATEncoder
from .decoder import LightweightDecoder, EdgeDecoder
from .layers.hypergraph_conv import DynamicHyperedgeWeighting
from .layers.attention import HypergraphAttentionAggregation
from ..training.dynamic_weights import (
    AdaptiveLossWeighting, 
    DynamicWeightAveraging, 
    LossBalancedWeighting,
    RandomWeightPerturbation
)
from ..training.simple_scheduler import SimpleMaskingScheduler, create_simple_masking_scheduler
from ..training.tcc_weights import create_tcc_controller


class EnhancedHyperGraphMAE(nn.Module):
    """
    Enhanced Hypergraph Masked Autoencoder for molecular representation learning.

    This model implements a self-supervised learning framework that:
    1. Masks portions of the molecular hypergraph
    2. Encodes the masked graph to learn representations
    3. Reconstructs the original features
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, latent_dim: int = 256,
                 proj_dim: int = 512, heads: int = 8, num_layers: int = 5,
                 mask_ratio: float = 0.7, config: Optional[Dict] = None, 
                 masking_scheduler: Optional[SimpleMaskingScheduler] = None):
        """
        Initialize the HyperGraph-MAE model.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent representation dimension
            proj_dim: Projection dimension
            heads: Number of attention heads
            num_layers: Number of encoder layers
            mask_ratio: Ratio of nodes/edges to mask (fallback for compatibility)
            config: Optional configuration dictionary
            masking_scheduler: Optional intelligent masking scheduler
        """
        super().__init__()
        
        # Validate input dimensions
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if proj_dim % heads != 0:
            raise ValueError(f"proj_dim ({proj_dim}) must be divisible by heads ({heads})")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be between 0 and 1, got {mask_ratio}")
        
        self.mask_ratio = mask_ratio  # Keep for backward compatibility
        self.config = config or {}
        # KISS: simple gated global attention (can be disabled via config['model']['attention']['disabled'])
        try:
            self.attention_enabled = not bool(self.config.get('model', {}).get('attention', {}).get('disabled', False))
        except Exception:
            self.attention_enabled = True

        # KISS: configure how hyperedge weights α_e are computed and used
        attn_cfg = self.config.get('model', {}).get('attention', {})
        # Whether to enable attribute-driven hyperedge weights during training (default True; matches max_full)
        self.use_edge_weights = bool(attn_cfg.get('use_edge_weights', True))
        # Whether to detach weights and use them only as gating coefficients (default True to avoid leakage paths)
        self.edge_weights_detach = bool(attn_cfg.get('weights_detach', True))

        # Whether to weight edge reconstruction loss by hyperedge weights (default True; configurable in loss section)
        loss_cfg_local = self.config.get('loss', {})
        self.edge_weights_in_loss = bool(loss_cfg_local.get('use_edge_weights_for_edge_loss', True))
        
        # Initialize simple masking scheduler
        if masking_scheduler is not None:
            self.masking_scheduler = masking_scheduler
        elif self.config.get('masking', {}):
            # Create simple scheduler from config
            self.masking_scheduler = create_simple_masking_scheduler(self.config)
        else:
            # Fallback: no advanced masking
            self.masking_scheduler = None
            logger.info("Using simple masking (no scheduler)")

        # Extract configuration
        loss_config = self.config.get('loss', {})
        training_config = self.config.get('training', {})
        
        # Initialize TCC controller (smoothing loss removed)
        self.loss_components = training_config.get('loss_components', ['reconstruction', 'edge'])
        tcc_config = training_config.get('tcc', {'enabled': True})
        
        self.tcc_controller = create_tcc_controller(self.loss_components, tcc_config)
        
        # Static edge loss weight as fallback when TCC is disabled
        self.edge_loss_weight = loss_config.get('edge_weight', 0.5)
        
        # Set default edge dimension for handling missing hyperedge_attr
        self.default_edge_dim = self.config.get('default_edge_dim', hidden_dim)
        
        # Dynamic weighting configuration
        self.use_dynamic_weights = loss_config.get('use_dynamic_weights', False)
        self.dynamic_weight_method = loss_config.get('dynamic_weight_method', 'adaptive')
        
        # Initialize dynamic weighting if enabled
        if self.use_dynamic_weights:
            if self.dynamic_weight_method == 'adaptive':
                self.dynamic_weighting = AdaptiveLossWeighting(num_tasks=2)
            elif self.dynamic_weight_method == 'dwa':
                self.dynamic_weighting = DynamicWeightAveraging(num_tasks=2)
            elif self.dynamic_weight_method == 'balanced':
                self.dynamic_weighting = LossBalancedWeighting()
            elif self.dynamic_weight_method == 'random':
                base_weights = {
                    'recon_weight': 1.0,
                    'edge_weight': self.edge_loss_weight
                }
                self.dynamic_weighting = RandomWeightPerturbation(base_weights)
            else:
                logger.warning(f"Unknown dynamic weight method: {self.dynamic_weight_method}")
                self.use_dynamic_weights = False

        # Initialize components
        # Note: adding lightweight global PE (degree + centrality) increases input dim by 2
        self.encoder = HyperGATEncoder(
            in_dim=in_dim + 2,  # KISS: +2 for degree and centrality features
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            proj_dim=proj_dim,
            heads=heads,
            num_layers=num_layers,
            config=config
        )

        self.decoder = LightweightDecoder(
            in_dim=proj_dim,
            out_dim=in_dim
        )

        # Initialize with config dimension, will be adapted dynamically if needed
        self.edge_decoder = EdgeDecoder(
            in_dim=proj_dim,
            edge_attr_dim=self.config.get('features', {}).get('hyperedge_dim', 17)
        )

        # Additional components
        self.hyperedge_weighting = DynamicHyperedgeWeighting(
            in_channels=self.config.get('features', {}).get('hyperedge_dim', 17)
        )

        edge_dim = self.config.get('features', {}).get('hyperedge_dim', 17)
        self.hypergraph_attention = HypergraphAttentionAggregation(
            in_dim=proj_dim,
            heads=heads,
            max_dim=edge_dim
        )

        # Optional descriptor head (graph-level attribute prediction)
        self.descriptor_cfg = self.config.get('descriptor_head', {})
        self.descriptor_enabled = bool(self.descriptor_cfg.get('enabled', False))
        self.descriptor_names = list(self.descriptor_cfg.get('names', [
            'TPSA', 'MolLogP', 'HBD', 'HBA', 'MolWt', 'RotatableBonds'
        ]))
        if self.descriptor_enabled and len(self.descriptor_names) > 0:
            desc_hidden = int(self.descriptor_cfg.get('hidden_dim', 128))
            self.descriptor_head = nn.Sequential(
                nn.Linear(proj_dim, desc_hidden),
                nn.ELU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(desc_hidden, len(self.descriptor_names))
            )
        else:
            self.descriptor_head = None

        # Descriptor normalization & warmup settings
        self.descriptor_normalize = bool(self.descriptor_cfg.get('normalize', True))
        self.descriptor_momentum = float(self.descriptor_cfg.get('momentum', 0.9))
        self.descriptor_freeze_steps = int(self.descriptor_cfg.get('warmup_freeze_steps', 0))
        if self.descriptor_head is not None:
            # Running stats for target standardization
            self.register_buffer('desc_running_mean', torch.zeros(len(self.descriptor_names)))
            self.register_buffer('desc_running_var', torch.ones(len(self.descriptor_names)))
            self.register_buffer('desc_stats_initialized', torch.tensor(False))


        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_attr: torch.Tensor, node_mask: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None, epoch: int = 0,
                global_step: int = None, max_steps: int = None,
                smiles: Optional[str] = None, recent_loss: Optional[float] = None,
                eval_mode: bool = False, return_z: bool = False, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.

        Args:
            x: Node features [num_nodes, in_dim]
            hyperedge_index: Hyperedge connectivity [2, num_connections]
            hyperedge_attr: Hyperedge attributes [num_edges, edge_dim]
            node_mask: Binary mask for nodes [num_nodes] (optional)
            edge_mask: Binary mask for edges [num_edges] (optional)
            epoch: Current training epoch (legacy, for compatibility)
            global_step: Current training step (preferred for step-based training)
            max_steps: Total training steps (for curriculum progress)
            smiles: SMILES string for intelligent masking (optional)
            recent_loss: Recent training loss for strategy optimization
            eval_mode: If True, return reconstruction results; if False, return losses
            **kwargs: Additional context for masking strategies

        Returns:
            If eval_mode=False: Tuple of (total_loss, reconstruction_loss, edge_loss)
            If eval_mode=True: Tuple of (recon_x, edge_pred, z_enhanced)
        """
        # Ensure consistent data types and validate inputs
        x = x.float()
        
        # Handle missing hyperedge attributes gracefully
        if hyperedge_attr is None:
            logger.warning("hyperedge_attr is None, creating zero tensor placeholder")
            # Create zero tensor placeholder with dimensions aligned to config.features.hyperedge_dim
            num_edges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() > 0 else 0
            edge_dim_cfg = int(self.config.get('features', {}).get('hyperedge_dim', getattr(self, 'default_edge_dim', 64)))
            if num_edges > 0:
                hyperedge_attr = torch.zeros(num_edges, edge_dim_cfg, dtype=x.dtype, device=x.device)
            else:
                hyperedge_attr = torch.zeros(0, edge_dim_cfg, dtype=x.dtype, device=x.device)
        else:
            hyperedge_attr = hyperedge_attr.float()

        # Unify normalization to configured dimension even for zero tensors
        expected_dim = int(self.config.get('features', {}).get('hyperedge_dim', hyperedge_attr.shape[-1]))
        actual_dim = int(hyperedge_attr.shape[-1]) if hyperedge_attr.dim() == 2 else expected_dim
        if actual_dim != expected_dim:
            if hyperedge_attr.size(0) > 0:
                if actual_dim < expected_dim:
                    pad_cols = expected_dim - actual_dim
                    pad_tensor = torch.zeros(hyperedge_attr.size(0), pad_cols, dtype=hyperedge_attr.dtype, device=hyperedge_attr.device)
                    hyperedge_attr = torch.cat([hyperedge_attr, pad_tensor], dim=-1)
                else:
                    hyperedge_attr = hyperedge_attr[:, :expected_dim]
            else:
                # Ensure empty tensors have the right trailing dim
                hyperedge_attr = hyperedge_attr.new_zeros((0, expected_dim))
        
        # Ensure hyperedge_index is long type for proper indexing
        if hyperedge_index.numel() > 0:
            hyperedge_index = hyperedge_index.long()
            
        # Validate tensor shapes and values
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN/inf detected in input features, cleaning...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(hyperedge_attr).any() or torch.isinf(hyperedge_attr).any():
            logger.warning("NaN/inf detected in edge attributes, cleaning...")
            hyperedge_attr = torch.nan_to_num(hyperedge_attr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Generate masks using intelligent scheduler or fallback
        if node_mask is None or edge_mask is None:
            generated_node_mask, generated_edge_mask = self._generate_intelligent_masks(
                x, hyperedge_index, hyperedge_attr, epoch, global_step, max_steps, smiles, recent_loss, **kwargs
            )
            
            if node_mask is None:
                node_mask = generated_node_mask
            if edge_mask is None:
                edge_mask = generated_edge_mask

        # Apply masks
        x_masked = self._apply_node_mask(x, node_mask)
        hyperedge_attr_masked = self._apply_edge_mask(hyperedge_attr, edge_mask)

        # Compute α_e from masked hyperedge attributes to avoid side-channel leakage (config-controlled)
        hyperedge_weights = None
        if self.use_edge_weights and (hyperedge_attr_masked is not None) and (hyperedge_attr_masked.size(0) > 0):
            try:
                hyperedge_weights = self.hyperedge_weighting(hyperedge_attr_masked)
                if self.edge_weights_detach and hyperedge_weights is not None:
                    # Use only as gating coefficients, avoiding gradients through attribute weights
                    hyperedge_weights = hyperedge_weights.detach()
            except Exception as _e:
                # If weight computation fails, fall back to no weights so training can continue
                logger.warning(f"Edge weight computation failed, fallback to None: {_e}")
                hyperedge_weights = None
        
        # Add lightweight global positional encoding (KISS: improves ESOL-style regression)
        x_masked = self._add_light_positional_encoding(x_masked, hyperedge_index)
        
        # Encode with proper error handling
        try:
            z = self.encoder(x_masked, hyperedge_index, hyperedge_attr_masked, hyperedge_weights)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during encoding: {e}")
                # Try to recover by cleaning memory and reducing precision
                torch.cuda.empty_cache()
                # Convert to float32 if needed
                x_masked = x_masked.float()
                hyperedge_attr_masked = hyperedge_attr_masked.float()
                # no hyperedge_weights in this variant
                z = self.encoder(x_masked, hyperedge_index, hyperedge_attr_masked, hyperedge_weights)
            else:
                raise e

        # Apply (or bypass) hypergraph attention based on simple config gate
        if getattr(self, 'attention_enabled', True):
            z_enhanced = self.hypergraph_attention(z, hyperedge_index, hyperedge_attr_masked, hyperedge_weights)
        else:
            z_enhanced = z

        # Decode node features
        recon_x = self.decoder(z_enhanced)

        # Edge decoder dimension is fixed at init; hyperedge_attr already normalized above
        edge_pred = self.edge_decoder(z_enhanced, hyperedge_index)

        # Build extra losses (e.g., descriptor head)
        extra_losses = {}
        if (not eval_mode) and self.descriptor_head is not None and ('descriptor' in self.loss_components):
            try:
                batch_vec = kwargs.get('batch', None)
                smiles_list = smiles if isinstance(smiles, (list, tuple)) else ([smiles] if isinstance(smiles, str) else None)
                # Optional warmup: skip descriptor loss during initial steps
                current_step = int(kwargs.get('global_step', 0) or 0)
                if current_step < int(self.descriptor_freeze_steps):
                    pass  # intentionally skip descriptor loss early on
                elif batch_vec is not None and smiles_list is not None and len(smiles_list) > 0:
                    g_emb = self._graph_pool_mean(z_enhanced, batch_vec)
                    desc_pred = self.descriptor_head(g_emb)
                    desc_target, valid_mask = self._compute_rdkit_descriptors(smiles_list, desc_pred.device, desc_pred.dtype)
                    # Align sizes conservatively
                    max_n = min(desc_pred.size(0), desc_target.size(0), valid_mask.size(0))
                    desc_pred = desc_pred[:max_n]
                    desc_target = desc_target[:max_n]
                    valid_mask = valid_mask[:max_n]
                    if valid_mask.any():
                        # Normalize targets to stabilize scales
                        if self.descriptor_normalize:
                            tgt = desc_target[valid_mask]
                            # Compute batch stats
                            b_mean = tgt.mean(dim=0)
                            # var with unbiased=False for numerical stability
                            b_var = tgt.var(dim=0, unbiased=False)
                            # Update running stats
                            if bool(self.desc_stats_initialized.item()):
                                self.desc_running_mean = self.descriptor_momentum * self.desc_running_mean + (1 - self.descriptor_momentum) * b_mean.detach()
                                self.desc_running_var = self.descriptor_momentum * self.desc_running_var + (1 - self.descriptor_momentum) * b_var.detach()
                            else:
                                self.desc_running_mean = b_mean.detach()
                                self.desc_running_var = b_var.detach() + 1e-6
                                self.desc_stats_initialized = torch.tensor(True, device=self.desc_running_mean.device)

                            # Standardize
                            eps = 1e-6
                            std = torch.sqrt(self.desc_running_var + eps)
                            y_norm = (tgt - self.desc_running_mean) / std
                            extra_losses['descriptor'] = F.mse_loss(desc_pred[valid_mask], y_norm)
                        else:
                            extra_losses['descriptor'] = F.mse_loss(desc_pred[valid_mask], desc_target[valid_mask])
            except Exception as e:
                logger.debug(f"Descriptor head skipped due to error: {e}")

        # Return based on mode
        if eval_mode:
            # Evaluation mode: return reconstruction results for metric calculation
            return recon_x, edge_pred, z_enhanced
        else:
            # Training mode: calculate and return losses
            losses = self._calculate_losses(
                recon_x, x, edge_pred, hyperedge_attr,
                z_enhanced, hyperedge_index, hyperedge_weights,
                node_mask, edge_mask, epoch, extra_losses=extra_losses
            )
            if return_z:
                # Append z_enhanced to the standard losses tuple for callers that need it
                # losses is a tuple; extend it by one element
                return tuple(list(losses) + [z_enhanced])
            return losses

    def _generate_intelligent_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                                   hyperedge_attr: torch.Tensor, epoch: int = 0,
                                   global_step: int = None, max_steps: int = None,
                                   smiles: str = None, recent_loss: float = None,
                                   **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate masks using simple scheduler or fallback to random masking."""
        device = x.device
        
        logger.debug(f"Generating masks: nodes={x.size(0)}, edges={hyperedge_attr.size(0) if hyperedge_attr is not None else 0}")
        
        if self.masking_scheduler is not None:
            try:
                # Use simple masking scheduler
                node_mask, edge_mask = self.masking_scheduler.generate_masks(
                    x, hyperedge_index, hyperedge_attr,
                    global_step=global_step, max_steps=max_steps,
                    epoch=epoch, smiles=smiles,
                    recent_loss=recent_loss, **kwargs
                )
                
                # Validate masks
                if node_mask is None or node_mask.sum().item() == 0:
                    logger.warning("Scheduler returned empty node mask! Using fallback.")
                    raise ValueError("Empty node mask")
                
                node_count = node_mask.sum().item()
                edge_count = edge_mask.sum().item() if edge_mask.numel() > 0 else 0
                logger.debug(f"Generated masks: {node_count}/{node_mask.size(0)} nodes, {edge_count}/{edge_mask.size(0) if edge_mask.numel() > 0 else 0} edges")
                
                return node_mask, edge_mask
                
            except Exception as e:
                logger.warning(f"Masking scheduler failed: {e}. Using random fallback.")
        
        # Fallback to simple random masking
        logger.debug("Using random masking fallback")
        # Extract dynamic mask_ratio from kwargs if provided
        dynamic_mask_ratio = kwargs.get('mask_ratio', None)
        node_mask = self._generate_simple_node_mask(x.size(0), device, dynamic_mask_ratio)
        edge_mask = self._generate_simple_edge_mask(
            hyperedge_attr.size(0) if hyperedge_attr is not None else 0, device, dynamic_mask_ratio
        )
        
        return node_mask, edge_mask
    
    def _generate_simple_node_mask(self, num_nodes: int, device: torch.device, mask_ratio: Optional[float] = None) -> torch.Tensor:
        """Generate simple random node mask (fallback)."""
        ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        return torch.rand(num_nodes, device=device) < ratio

    def _generate_simple_edge_mask(self, num_edges: int, device: torch.device, mask_ratio: Optional[float] = None) -> torch.Tensor:
        """Generate simple random edge mask (fallback)."""
        if num_edges == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        return torch.rand(num_edges, device=device) < ratio

    def _apply_node_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to node features efficiently."""
        if not mask.any():  # Early exit if no masking needed
            return x
        # Use in-place operations when safe
        x_masked = x.clone()  # Only clone when actually needed
        x_masked[mask] = 0  # More direct indexing
        return x_masked

    def _apply_edge_mask(self, edge_attr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to edge attributes efficiently."""
        if edge_attr.size(0) == 0 or mask.size(0) == 0:
            return edge_attr
        
        # Early exit if no masking needed
        if not mask.any():
            return edge_attr
            
        # Use direct indexing for efficiency - fix chained indexing assignment
        edge_attr_masked = edge_attr.clone()
        effective_size = min(mask.size(0), edge_attr.size(0))
        if effective_size > 0:
            # Fix: Use proper indexing to avoid assignment to temporary view
            masked_indices = mask[:effective_size].nonzero(as_tuple=True)[0]
            if len(masked_indices) > 0:
                edge_attr_masked[masked_indices] = 0
        
        return edge_attr_masked

    def _add_light_positional_encoding(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Add lightweight global positional encoding (degree + centrality approximation).
        
        Goal: provide global coordinate information for ESOL-style global property tasks
        and improve regression performance at low cost.
        Cost: only adds 2 channels; no need for large precomputed graphs.
        
        Args:
            x: Node features [num_nodes, in_dim]
            hyperedge_index: Hyperedge connectivity [2, num_connections]
            
        Returns:
            Enhanced node features [num_nodes, in_dim + 2]
        """
        if hyperedge_index.numel() == 0 or x.size(0) == 0:
            # Empty graph: append zero global features
            zero_pe = torch.zeros(x.size(0), 2, dtype=x.dtype, device=x.device)
            return torch.cat([x, zero_pe], dim=-1)
        
        try:
            # 1. Degree feature (normalized)
            # Compute node degree from hyperedge connectivity (number of incident hyperedges)
            node_degrees = degree(hyperedge_index[0], num_nodes=x.size(0), dtype=torch.float)
            max_degree = node_degrees.max().clamp_min(1.0)
            normalized_degree = (node_degrees / max_degree).unsqueeze(-1)  # [N, 1]
            
            # 2. Approximate centrality (BFS-inspired)
            # Simplified: use a normalized degree ratio as a proxy for centrality
            # High-degree nodes tend to be closer to the graph center
            approx_centrality = (node_degrees / (node_degrees.sum().clamp_min(1.0))).unsqueeze(-1)  # [N, 1]
            
            # Concatenate original features and global PE
            enhanced_x = torch.cat([x, normalized_degree, approx_centrality], dim=-1)
            
            return enhanced_x
            
        except Exception as e:
            logger.warning(f"Failed to compute light PE, using original features: {e}")
            # If computation fails, append zero PE to keep dimensions consistent
            zero_pe = torch.zeros(x.size(0), 2, dtype=x.dtype, device=x.device)
            return torch.cat([x, zero_pe], dim=-1)

    def _calculate_losses(self, recon_x: torch.Tensor, x: torch.Tensor,
                          edge_pred: torch.Tensor, hyperedge_attr: torch.Tensor,
                          z: torch.Tensor, hyperedge_index: torch.Tensor,
                          hyperedge_weights: Optional[torch.Tensor],
                          node_mask: torch.Tensor, edge_mask: torch.Tensor,
                          epoch: int = 0, extra_losses: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        """Calculate all loss components with TCC adaptive weighting."""
        # Individual loss components with boundary checks
        if node_mask.sum() == 0:
            logger.warning("No nodes masked, using random 20% of nodes for reconstruction loss")
            num_nodes = node_mask.size(0)
            random_indices = torch.randperm(num_nodes, device=node_mask.device)[:max(1, num_nodes // 5)]
            node_mask = torch.zeros_like(node_mask, dtype=torch.bool)
            node_mask[random_indices] = True
        
        # Build loss dict containing only the selected components
        losses = {}
        # Track which components are actually computed at this step (for correct logging)
        computed_components = set()
        
        # Reconstruction loss (always used)
        if 'reconstruction' in self.loss_components:
            losses['reconstruction'] = F.mse_loss(recon_x[node_mask], x[node_mask])
            computed_components.add('reconstruction')
        
        # Edge loss
        if 'edge' in self.loss_components:
            losses['edge'] = self._calculate_edge_loss(
                edge_pred, hyperedge_attr, edge_mask, hyperedge_weights
            )
            computed_components.add('edge')
        
        # Additional losses (e.g., graph-level descriptor head)
        if extra_losses:
            for k, v in extra_losses.items():
                if k in self.loss_components and hasattr(v, 'dtype'):
                    losses[k] = v
                    computed_components.add(k)

        # Safety fallback: ensure every enabled component has a tensor value
        # (e.g. descriptor during warmup or when SMILES are invalid)
        for comp_name in self.loss_components:
            if comp_name not in losses:
                losses[comp_name] = torch.tensor(0.0, device=recon_x.device)

        # Contrastive loss (if implemented)
        if 'contrastive' in self.loss_components:
            # TODO: implement contrastive loss
            losses['contrastive'] = torch.tensor(0.0, device=recon_x.device, requires_grad=True)
            logger.debug("Contrastive loss not implemented yet, using zero")
        
        # Use TCC controller or fallback weighting
        if self.tcc_controller is not None:
            # Use TCC adaptive weights
            total_loss, tcc_info = self.tcc_controller(losses)
            
            # Extract diagnostic information
            weights = tcc_info['weights']
            contributions = tcc_info['contributions']
            ema_contributions = tcc_info['ema_contributions']
            targets = tcc_info['targets']
            
            # Build extended info dict
            extended_info = {
                'total_loss': total_loss.item(),
                'weights': weights,
                'contributions': contributions,
                'ema_contributions': ema_contributions,
                'target_ratios': targets,
                'loss_components': list(losses.keys()),
                # Only report losses for components actually computed this step
                'component_losses': {k: float(losses[k].detach().item()) for k in computed_components if k in losses},
                # Mark which components were computed for trainer-side logging
                'computed_components': {k: (k in computed_components) for k in self.loss_components},
                # Also keep TCC internal raw/normalized loss info
                'raw_losses': tcc_info.get('raw_losses', {}),
                'normalized_losses': tcc_info.get('normalized_losses', {}),
                'ema_scales': tcc_info.get('ema_scales', {}),
            }
            
            return (
                total_loss,
                losses.get('reconstruction', torch.tensor(0.0)),
                losses.get('edge', torch.tensor(0.0)),
                torch.tensor(0.0),  # Reserved for backward compatibility
                losses.get('contrastive', torch.tensor(0.0)),
                extended_info
            )
        
        elif self.use_dynamic_weights:
            # Fallback dynamic weighting scheme
            loss_dict = {
                'recon_loss': losses.get('reconstruction', torch.tensor(0.0)),
                'edge_loss': losses.get('edge', torch.tensor(0.0))
            }
            
            if self.dynamic_weight_method == 'dwa':
                total_loss, weights = self.dynamic_weighting(loss_dict, epoch)
            else:
                total_loss, weights = self.dynamic_weighting(loss_dict)
                
            return (
                total_loss,
                loss_dict['recon_loss'],
                loss_dict['edge_loss'], 
                torch.tensor(0.0),  # Reserved for backward compatibility
                torch.tensor(0.0),  # Contrastive loss placeholder
                weights
            )
        
        else:
            # Static weighting fallback
            recon_loss = losses.get('reconstruction', torch.tensor(0.0, device=recon_x.device))
            edge_loss = losses.get('edge', torch.tensor(0.0, device=recon_x.device))
            
            total_loss = recon_loss + self.edge_loss_weight * edge_loss
            
            weights = {
                'recon_weight': 1.0,
                'edge_weight': self.edge_loss_weight
            }
            
            return (
                total_loss,
                recon_loss,
                edge_loss, 
                torch.tensor(0.0),  # Reserved for backward compatibility
                torch.tensor(0.0),  # Contrastive loss placeholder
                weights
            )

    def _calculate_edge_loss(self, edge_pred: torch.Tensor, hyperedge_attr: torch.Tensor,
                             edge_mask: torch.Tensor, hyperedge_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Calculate edge reconstruction loss."""
        if edge_pred.size(0) == 0 or hyperedge_attr.size(0) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=edge_pred.device)

        # If edge loss is not configured to use weights, ignore provided weights
        if not getattr(self, 'edge_weights_in_loss', True):
            hyperedge_weights = None

        # Ensure mask compatibility
        valid_mask_size = min(edge_mask.size(0), edge_pred.size(0), hyperedge_attr.size(0))
        valid_mask = edge_mask[:valid_mask_size] if valid_mask_size > 0 else torch.zeros(0, dtype=torch.bool,
                                                                                         device=edge_mask.device)

        if valid_mask.sum() == 0:
            logger.warning("No edges masked for edge loss, using random 20% of edges")
            # Create a random edge mask with 20% of edges
            num_edges = min(edge_pred.size(0), hyperedge_attr.size(0))
            if num_edges > 0:
                random_indices = torch.randperm(num_edges, device=edge_pred.device)[:max(1, num_edges // 5)]
                valid_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_pred.device)
                valid_mask[random_indices] = True
                valid_mask_size = num_edges
            else:
                return torch.tensor(0.0, dtype=torch.float32, device=edge_pred.device)

        # Apply weights if available with proper dimension handling
        if hyperedge_weights is not None and hyperedge_weights.size(0) > 0:
            weights = hyperedge_weights[:valid_mask_size][valid_mask]
            
            # Fix dimension handling to prevent broadcasting bugs
            # First squeeze all trailing dimensions of size 1, then ensure proper shape
            weights = weights.squeeze()
            if weights.dim() == 0:  # Handle scalar case
                weights = weights.unsqueeze(0)
            
            edge_diff = (edge_pred[:valid_mask_size][valid_mask] - hyperedge_attr[:valid_mask_size][valid_mask]) ** 2
            weights = weights.to(edge_diff.dtype)
            
            # Handle dimension compatibility for broadcasting
            if edge_diff.dim() > 1 and weights.dim() == 1:
                # Ensure weights can broadcast properly with edge_diff
                weights = weights.view(-1, 1)  # [M, 1] for proper broadcasting with [M, D]
            
            edge_loss = (edge_diff * weights).mean()
        else:
            edge_loss = F.mse_loss(
                edge_pred[:valid_mask_size][valid_mask],
                hyperedge_attr[:valid_mask_size][valid_mask]
            )

        return edge_loss

    def _graph_pool_mean(self, z: torch.Tensor, batch_vec: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean-pool node embeddings to graph-level embeddings."""
        if batch_vec is None:
            return z.mean(dim=0, keepdim=True)
        try:
            import torch_scatter
            num_graphs = int(batch_vec.max().item() + 1) if batch_vec.numel() > 0 else 1
            return torch_scatter.scatter_mean(z, batch_vec, dim=0, dim_size=num_graphs)
        except Exception:
            num_graphs = int(batch_vec.max().item() + 1) if batch_vec.numel() > 0 else 1
            pooled = []
            for g in range(num_graphs):
                m = (batch_vec == g)
                pooled.append(z[m].mean(dim=0) if m.any() else z.new_zeros(z.size(-1)))
            return torch.stack(pooled, dim=0)

    def _compute_rdkit_descriptors(self, smiles_list: list, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RDKit descriptors for a list of SMILES.

        Supports common names in rdkit.Chem.Descriptors and a few aliases:
        - TPSA (alias to rdMolDescriptors.CalcTPSA)
        - FracCSP3 (alias to Descriptors.FractionCSP3)
        - LabuteASA (alias to rdMolDescriptors.CalcLabuteASA; returns first element if tuple)
        - qed (alias to rdMolDescriptors.CalcQED)
        - Any name available under rdkit.Chem.Descriptors, e.g.,
          SlogP_VSA1..12, PEOE_VSA1..14, EState_VSA1..11, SMR_VSA1..10,
          NumAromaticRings, NumAliphaticRings, etc.

        Returns:
            targets: [num_graphs, num_desc], valid_mask: [num_graphs]
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit.Chem import rdMolDescriptors as rdm
        except Exception as e:
            # Strict mode: fail loudly if RDKit is missing, avoiding silent fallbacks
            raise ImportError(f"RDKit is required for descriptor supervision but not available: {e}")

        vals = []
        mask = []
        # Strictly validate descriptor names before computation
        supported = set(dir(Descriptors))
        for name in self.descriptor_names:
            if name in ("TPSA", "FracCSP3", "LabuteASA", "qed"):
                continue
            # Common but unsupported aliases: instruct user to rename
            if name in ("HBD", "HBA"):
                raise ValueError("Unsupported descriptor alias detected: 'HBD'/'HBA'. Use 'NumHDonors'/'NumHAcceptors' instead.")
            if name not in supported:
                raise ValueError(f"Unknown descriptor name: '{name}'. Provide a valid RDKit descriptor name or supported alias (TPSA, FracCSP3, LabuteASA, qed).")

        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi) if smi is not None else None
                if mol is None:
                    vals.append([0.0] * len(self.descriptor_names))
                    mask.append(False)
                    continue
                row = []
                for name in self.descriptor_names:
                    try:
                        if name == 'TPSA':
                            v = float(rdm.CalcTPSA(mol))
                        elif name in ('FracCSP3', 'FractionCSP3'):
                            v = float(Descriptors.FractionCSP3(mol))
                        elif name == 'LabuteASA':
                            out = rdm.CalcLabuteASA(mol)
                            v = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
                        elif name == 'qed':
                            v = float(rdm.CalcQED(mol))
                        else:
                            # All other names must be callable in rdkit.Chem.Descriptors (validated above)
                            fn = getattr(Descriptors, name)
                            v = float(fn(mol))
                    except Exception as ex:
                        # Strict mode: descriptor computation failure should surface as an error
                        raise RuntimeError(f"Failed to compute descriptor '{name}' for SMILES '{smi}': {ex}")
                    row.append(v)
                vals.append(row)
                mask.append(True)
            except Exception:
                vals.append([0.0] * len(self.descriptor_names))
                mask.append(False)
        targets = torch.tensor(vals, device=device, dtype=dtype)
        valid_mask = torch.tensor(mask, device=device, dtype=torch.bool)
        return targets, valid_mask

    def get_embedding(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings for inference (without masking).

        Args:
            x: Node features
            hyperedge_index: Hyperedge connectivity
            hyperedge_attr: Hyperedge attributes

        Returns:
            Node embeddings
        """
        # === Normalize hyperedge_attr dimension to match training forward() ===
        x = x.float()
        if hyperedge_attr is None:
            logger.warning("hyperedge_attr is None, creating zero tensor placeholder")
            num_edges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() > 0 else 0
            edge_dim_cfg = int(self.config.get('features', {}).get('hyperedge_dim', getattr(self, 'default_edge_dim', 64)))
            if num_edges > 0:
                hyperedge_attr = torch.zeros(num_edges, edge_dim_cfg, dtype=x.dtype, device=x.device)
            else:
                hyperedge_attr = torch.zeros(0, edge_dim_cfg, dtype=x.dtype, device=x.device)
        else:
            hyperedge_attr = hyperedge_attr.float()

        # Unify normalization to configured dimension even for zero tensors
        expected_dim = int(self.config.get('features', {}).get('hyperedge_dim', hyperedge_attr.shape[-1]))
        actual_dim = int(hyperedge_attr.shape[-1]) if hyperedge_attr.dim() == 2 else expected_dim
        if actual_dim != expected_dim:
            if hyperedge_attr.size(0) > 0:
                if actual_dim < expected_dim:
                    pad_cols = expected_dim - actual_dim
                    pad_tensor = torch.zeros(hyperedge_attr.size(0), pad_cols, dtype=hyperedge_attr.dtype, device=hyperedge_attr.device)
                    hyperedge_attr = torch.cat([hyperedge_attr, pad_tensor], dim=-1)
                else:
                    hyperedge_attr = hyperedge_attr[:, :expected_dim]
            else:
                hyperedge_attr = hyperedge_attr.new_zeros((0, expected_dim))

        if hyperedge_index.numel() > 0:
            hyperedge_index = hyperedge_index.long()

        # Calculate hyperedge weights after normalization (no masking in inference)
        hyperedge_weights = None
        if hyperedge_attr is not None and hyperedge_attr.size(0) > 0:
            hyperedge_weights = self.hyperedge_weighting(hyperedge_attr)

        # Add lightweight global positional encoding (KISS: also needed at inference)
        x = self._add_light_positional_encoding(x, hyperedge_index)

        # Get embeddings (pass hyperedge_weights to keep parity with training path)
        z = self.encoder.get_embedding(x, hyperedge_index, hyperedge_attr, hyperedge_weights)

        # Apply hypergraph attention
        z_enhanced = self.hypergraph_attention(z, hyperedge_index, hyperedge_attr, hyperedge_weights)

        return z_enhanced

    def configure_optimizers(self, lr: float = 5e-4, weight_decay: float = 1e-4) -> Dict:
        """
        Configure optimizer for training.

        Args:
            lr: Learning rate
            weight_decay: Weight decay

        Returns:
            Dictionary with optimizer only (scheduler is managed by Trainer)
        """
        # Separate parameters for different weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():            
            # Only include tensor parameters that require gradients
            if not isinstance(param, torch.Tensor):
                continue
            if not param.requires_grad:
                continue
            if not param.dtype.is_floating_point:
                continue
            
            # KISS: group parameters so bias/norm/embedding do not use weight decay
            if (param.dim() == 1 or                          # 1D tensors: bias, norm parameters
                'bias' in name.lower() or 
                'norm' in name.lower() or 
                'embed' in name.lower() or 
                'positional' in name.lower()):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        if not decay_params and not no_decay_params:
            raise ValueError("No valid parameters found for optimizer!")

        # Use proper parameter grouping for different weight decay settings
        param_groups = []
        
        if decay_params:
            param_groups.append({
                'params': decay_params, 
                'weight_decay': float(weight_decay),
                'name': 'decay_params'
            })
            
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params, 
                'weight_decay': 0.0,
                'name': 'no_decay_params'
            })
        
        logger.info(f"Parameter groups: decay_params={len(decay_params)}, no_decay_params={len(no_decay_params)}")
        
        # Use CUDA fused AdamW when available for faster optimizer steps
        try:
            optimizer = torch.optim.AdamW(param_groups, lr=float(lr), fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups, lr=float(lr))

        # Scheduler is owned by the Trainer (single source of truth)
        return {
            'optimizer': optimizer
        }


class PretrainedHyperGraphMAE(EnhancedHyperGraphMAE):
    """
    HyperGraph-MAE with pretrained weight loading capabilities.
    """

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[Dict] = None,
                        strict: bool = True) -> 'PretrainedHyperGraphMAE':
        """
        Load pretrained model from checkpoint, inferring required in_dim.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Optional model configuration to override checkpoint config
            strict: Whether to strictly enforce parameter matching
        
        Returns:
            Loaded model instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Prefer provided config; otherwise pull from checkpoint
        if config is None:
            config = checkpoint.get('config')
        if config is None:
            raise ValueError("Missing config in checkpoint; please provide config.")

        # Copy model cfg to avoid mutating input config
        model_cfg = dict(config.get('model', {}))

        # Infer in_dim: prefer config.features.atom.dim, fallback to weights
        in_dim = (
            config.get('features', {})
                  .get('atom', {})
                  .get('dim')
        )
        if in_dim is None:
            state = checkpoint.get('state_dict', checkpoint)
            for key_name in ('encoder.init_proj.weight', 'encoder.input_proj.weight'):
                if key_name in state and getattr(state[key_name], 'shape', None):
                    in_dim = int(state[key_name].shape[1])
                    break
        if in_dim is None:
            raise ValueError("Failed to infer in_dim from config/weights.")

        # Avoid duplicate kwargs if present in model_cfg
        if 'in_dim' in model_cfg:
            model_cfg.pop('in_dim')

        # Initialize masking scheduler if config contains masking settings
        masking_scheduler = None
        if config.get('masking'):
            try:
                masking_scheduler = create_masking_scheduler(config)
                # Load scheduler state if available
                scheduler_state = checkpoint.get('masking_scheduler_state')
                if scheduler_state:
                    masking_scheduler.load_state_dict(scheduler_state)
                    logger.info("Loaded masking scheduler state from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load masking scheduler: {e}")

        # Initialize model
        model = cls(in_dim=in_dim, config=config, masking_scheduler=masking_scheduler, **model_cfg)

        # Load weights
        state = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state, strict=strict)
        
        return model

    def save_pretrained(self, save_path: str, config: Optional[Dict] = None):
        """
        Save model with configuration and masking scheduler state.

        Args:
            save_path: Path to save checkpoint
            config: Configuration to save with model
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': config or self.config
        }
        
        # Save masking scheduler state if available
        if hasattr(self, 'masking_scheduler') and self.masking_scheduler is not None:
            try:
                checkpoint['masking_scheduler_state'] = self.masking_scheduler.get_state_dict()
                logger.info("Saved masking scheduler state to checkpoint")
            except Exception as e:
                logger.warning(f"Failed to save masking scheduler state: {e}")
        
        torch.save(checkpoint, save_path)
