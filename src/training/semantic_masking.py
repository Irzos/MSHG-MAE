"""
Semantic-aware masking strategy driven by molecular structure.

Fixed semantic masking that can:
1) Identify functional groups, rings, and chains
2) Mask by semantic blocks instead of isolated atoms
3) Mix semantic masking with random masking for robustness
4) Preserve important functional groups when configured
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from rdkit import Chem

from .masking_strategies import MaskingStrategy
from ..data.molecular_semantics import MolecularSemanticAnalyzer

logger = logging.getLogger(__name__)


class SemanticMasking(MaskingStrategy):
    """Fixed semantic masking with optional random blending and preservation rules."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Base ratios
        self.node_mask_ratio = self.config.get('node_mask_ratio', 0.7)
        self.edge_mask_ratio = self.config.get('edge_mask_ratio', 0.7)
        
        # Semantic masking options
        semantic_config = self.config.get('semantic', {})
        self.semantic_priority = semantic_config.get('semantic_priority', 0.7)
        self.block_types = semantic_config.get('block_types', 
                                             ['functional_group', 'ring', 'chain'])
        self.min_block_size = semantic_config.get('min_block_size', 2)
        self.max_block_size = semantic_config.get('max_block_size', 8)
        # Important-group handling (mutually exclusive options)
        self.preserve_important = semantic_config.get('preserve_important', True)
        self.enhance_important_masking = semantic_config.get('enhance_important_masking', False)
        
        # Enforce mutual exclusivity
        if self.preserve_important and self.enhance_important_masking:
            logger.warning("preserve_important and enhance_important_masking cannot be enabled simultaneously. Using preserve_important=True.")
            self.enhance_important_masking = False
        
        # Important groups
        self.important_groups = set(semantic_config.get('important_groups', [
            'carboxyl', 'amino', 'hydroxyl', 'nitro', 'cyano', 'phosphate'
        ]))
        
        # Light jittering to avoid overly rigid patterns
        jitter_config = semantic_config.get('jittering', {})
        self.enable_jittering = jitter_config.get('enable', True)
        self.prob_jitter_std = jitter_config.get('prob_jitter_std', 0.1)
        self.boundary_jitter_prob = jitter_config.get('boundary_jitter_prob', 0.2)
        self.ratio_jitter_std = jitter_config.get('ratio_jitter_std', 0.05)
        
        # Block-type aliases (singular/plural)
        self.block_type_aliases = {
            'functional_groups': 'functional_group',
            'rings': 'ring', 
            'chains': 'chain'
        }
        
        # Normalize block type names
        self.block_types = [self.block_type_aliases.get(bt, bt) for bt in self.block_types]
        
        # Semantic analyzer
        self.analyzer = MolecularSemanticAnalyzer()
        
        # Preprocessing cache options
        cache_config = self.config.get('cache', {})
        self.cache_file = cache_config.get('file', None)
        self.enable_cache = cache_config.get('enable', True)
        self.semantic_cache = {}
        
        # Load cache if provided
        if self.enable_cache and self.cache_file:
            self._load_semantic_cache()
        
        # Reproducibility
        random_config = self.config.get('random', {})
        self.base_seed = random_config.get('base_seed', 42)
        self.use_deterministic = random_config.get('deterministic', False)
        self.mask_rng_state = None
        
        # Basic statistics (defaultdict avoids missing keys)
        from collections import defaultdict
        self.stats = {
            'semantic_attempts': 0,
            'semantic_successes': 0,
            'random_fallbacks': 0,
            'total_masks_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'block_type_usage': defaultdict(int)
        }
        # Pre-seed block type counters
        for bt in self.block_types:
            self.stats['block_type_usage'][bt] = 0
        
        # Log basic strategy configuration
        strategy_info = "preserve_important" if self.preserve_important else ("enhance_important" if self.enhance_important_masking else "neutral")
        logger.info(f"SemanticMasking initialized: semantic_priority={self.semantic_priority}, "
                   f"block_types={self.block_types}, strategy={strategy_info}")
        if self.semantic_cache:
            logger.info(f"Loaded semantic cache with {len(self.semantic_cache)} molecules")
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                      hyperedge_attr: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic-aware masks.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            hyperedge_index: Hyperedge incidence [2, num_connections]
            hyperedge_attr: Hyperedge attributes [num_edges, edge_dim]
            **kwargs: Extra context (smiles, mol_info, batch, etc.)
            
        Returns:
            Tuple of (node_mask, edge_mask)
        """
        num_nodes = x.size(0)
        # One-shot ratio override from Trainer via kwargs
        self._ratio_override = None
        if 'mask_ratio' in kwargs and kwargs['mask_ratio'] is not None:
            try:
                self._ratio_override = float(kwargs['mask_ratio'])
            except Exception:
                self._ratio_override = None
        # Infer edge count: prefer hyperedge_attr, else from hyperedge_index
        if hyperedge_attr is not None:
            num_edges = hyperedge_attr.size(0)
        elif hyperedge_index is not None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max().item()) + 1
        else:
            num_edges = 0
        device = x.device
        
        self.stats['total_masks_generated'] += 1
        
        # Deterministic seeding (optional)
        step = kwargs.get('global_step', 0)
        epoch = kwargs.get('epoch', 0)
        if self.use_deterministic:
            self.set_mask_seed(step, epoch)
        
        # Read SMILES and batch info
        smiles = kwargs.get('smiles', None)
        batch_tensor = kwargs.get('batch', None)  # PyG batch tensor
        
        # Batched case
        if batch_tensor is not None and smiles is not None and isinstance(smiles, (list, tuple)):
            return self._generate_batch_semantic_masks(
                x, hyperedge_index, hyperedge_attr, smiles, batch_tensor, device
            )
        
        # Single-molecule fallback
        single_smiles = None
        if smiles is not None:
            if isinstance(smiles, (list, tuple)) and len(smiles) > 0:
                single_smiles = smiles[0]
            else:
                single_smiles = str(smiles)
        
        # Always attempt semantic masking
        self.stats['semantic_attempts'] += 1
        
        # Molecular info
        mol_info = kwargs.get('mol_info')
        
        # Generate semantic masks
        semantic_masks = self._generate_semantic_masks(
            num_nodes, num_edges, device, single_smiles, mol_info, hyperedge_index
        )
        
        if semantic_masks is not None:
            self.stats['semantic_successes'] += 1
            node_mask, edge_mask = semantic_masks
            
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    nm = int(node_mask.sum())
                    em = int(edge_mask.sum())
                    logger.debug(f"Semantic masking applied: {nm}/{num_nodes} nodes, {em}/{num_edges} edges")
                except Exception:
                    pass
            # Clear override flag
            self._ratio_override = None
            return node_mask, edge_mask
        
        # Random fallback; log when missing SMILES/mol_info
        if single_smiles is None and kwargs.get('mol_info') is None:
            logger.info("Semantic masking fallback: no SMILES/mol_info provided; using random masking")
        self.stats['random_fallbacks'] += 1
        out_masks = self._generate_random_masks(num_nodes, num_edges, device)
        # Clear override flag
        self._ratio_override = None
        return out_masks
    
    def _generate_semantic_masks(self, num_nodes: int, num_edges: int, device: torch.device,
                               smiles: str = None, mol_info: Dict = None, 
                               hyperedge_index: Optional[torch.Tensor] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate masks based on molecular semantics and avoid structural leakage.
        
        Args:
            num_nodes: Number of nodes
            num_edges: Number of hyperedges
            device: Target device
            smiles: SMILES string
            mol_info: Precomputed molecular info
            hyperedge_index: Hyperedge index for structure-aware masking
            
        Returns:
            Tuple (node_mask, hyperedge_mask) or None if unavailable
        """
        # Fetch or compute molecular semantics (prefer cache)
        if mol_info is None and smiles:
            mol_info = self._get_cached_mol_info(smiles)
            if mol_info is None:
                return None
        
        if not mol_info:
            logger.debug("No molecular information available for semantic masking")
            return None
        
        # Retrieve semantic blocks
        semantic_blocks = mol_info.get('semantic_blocks', {})
        if not semantic_blocks or not any(semantic_blocks.values()):
            logger.debug("No semantic blocks found in molecular analysis")
            return None
        
        # Generate semantic node mask
        node_mask = self._create_semantic_node_mask(
            num_nodes, semantic_blocks, mol_info, device
        )
        
        # Create structure-aware hyperedge mask with correct length
        if hyperedge_index is not None:
            hyperedge_mask = self._generate_structure_aware_hyperedge_mask_fixed(
                node_mask, hyperedge_index, num_edges, device
            )
        else:
            # Fallback to simple edge mask
            hyperedge_mask = self._generate_edge_mask(num_edges, device)
        
        return node_mask, hyperedge_mask
    
    def _generate_batch_semantic_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                                     hyperedge_attr: torch.Tensor, smiles_list: List[str], 
                                     batch_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic masks per-molecule for a batch.

        Args:
            x: Node features [total_nodes, feature_dim]
            hyperedge_index: Hyperedge incidence [2, num_connections]
            hyperedge_attr: Hyperedge attributes [num_edges, edge_dim]
            smiles_list: List of SMILES strings
            batch_tensor: PyG batch tensor mapping nodes to molecules
            device: Target device

        Returns:
            Tuple (node_mask, edge_mask)
        """
        total_nodes = x.size(0)
        # Infer edge count for batch: prefer hyperedge_attr, else hyperedge_index
        if hyperedge_attr is not None:
            total_edges = hyperedge_attr.size(0)
        elif hyperedge_index is not None and hyperedge_index.numel() > 0:
            total_edges = int(hyperedge_index[1].max().item()) + 1
        else:
            total_edges = 0
        
        # Initialize mask tensors
        node_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
        edge_mask = torch.zeros(total_edges, dtype=torch.bool, device=device) if total_edges > 0 else torch.empty(0, dtype=torch.bool, device=device)
        
        # GPU-optimized: stay on device
        unique_graphs = batch_tensor.unique()
        
        # Track actual number of semantic successes
        actual_semantic_successes = 0
        
        # Loop on GPU to avoid CPU transfers
        for i in range(unique_graphs.size(0)):
            graph_idx_tensor = unique_graphs[i]
            graph_idx = graph_idx_tensor.item()
            try:
                # Nodes of current molecule
                graph_node_indices = torch.where(batch_tensor == graph_idx)[0]
                num_graph_nodes = len(graph_node_indices)
                
                # Fetch SMILES for this graph (bounds check)
                if graph_idx < len(smiles_list):
                    current_smiles = smiles_list[graph_idx]
                    
                    # Extract hyperedge index for current graph (handle None/empty)
                    if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                        graph_hyperedge_mask = (batch_tensor[hyperedge_index[0]] == graph_idx)
                        graph_hyperedge_index = hyperedge_index[:, graph_hyperedge_mask] if graph_hyperedge_mask.any() else None
                    else:
                        graph_hyperedge_mask = torch.zeros(0, dtype=torch.bool, device=device)
                        graph_hyperedge_index = None
                    
                    # Generate semantic mask for current molecule (nodes only)
                    semantic_masks = self._generate_semantic_masks(
                        num_graph_nodes, 0, device, current_smiles, None, None
                    )
                    
                    if semantic_masks is not None:
                        graph_node_mask, _ = semantic_masks
                        # Update global node mask
                        node_mask[graph_node_indices] = graph_node_mask
                        # Track success
                        actual_semantic_successes += 1
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Applied semantic mask to molecule {graph_idx}: "
                                f"{int(graph_node_mask.sum().item())}/{num_graph_nodes} nodes"
                            )
                    else:
                        # Fallback when SMILES missing
                        if current_smiles is None:
                            logger.info(f"Semantic masking fallback: molecule {graph_idx} missing SMILES; using random masking")
                        # Random fallback with jittered ratio
                        jittered_node_ratio, _ = self._get_jittered_ratios()
                        random_mask = torch.rand(num_graph_nodes, device=device) < jittered_node_ratio
                        node_mask[graph_node_indices] = random_mask
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Applied random fallback mask to molecule {graph_idx}: "
                                f"{int(random_mask.sum().item())}/{num_graph_nodes} nodes"
                            )
                        self.stats['random_fallbacks'] += 1
                else:
                    # If SMILES missing, use random masking
                    logger.warning(f"No SMILES available for molecule {graph_idx}, using random mask")
                    jittered_node_ratio, _ = self._get_jittered_ratios()
                    random_mask = torch.rand(num_graph_nodes, device=device) < jittered_node_ratio
                    node_mask[graph_node_indices] = random_mask
                    
                    self.stats['random_fallbacks'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing molecule {graph_idx} in batch: {e}")
                # On error, use random masking
                graph_node_indices = torch.where(batch_tensor == graph_idx)[0]
                num_graph_nodes = len(graph_node_indices)
                random_mask = torch.rand(num_graph_nodes, device=device) < self.node_mask_ratio
                node_mask[graph_node_indices] = random_mask
                
                # Build structure-aware edge mask for errors
                try:
                    if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                        graph_hyperedge_mask = (batch_tensor[hyperedge_index[0]] == graph_idx)
                        graph_hyperedge_index = hyperedge_index[:, graph_hyperedge_mask]
                    else:
                        graph_hyperedge_index = None
                except Exception as edge_error:
                    logger.warning(f"Failed to generate structure-aware edge mask for error case: {edge_error}")
                
                self.stats['random_fallbacks'] += 1
        
        # Complement structure-aware edge mask to target ratio (with jitter)
        if total_edges > 0:
            # Global structure-aware mask from final node_mask and full hyperedge_index
            if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                struct_mask_global = self._generate_structure_aware_hyperedge_mask_fixed(
                    node_mask, hyperedge_index, total_edges, device
                )
                if struct_mask_global.numel() == edge_mask.numel():
                    edge_mask |= struct_mask_global
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(
                            f"Global structure-aware edges masked: {int(edge_mask.sum().item())}/{total_edges}"
                        )
                    except Exception:
                        pass

            # Prefer edges connected to masked nodes for consistency
            candidate_edges = torch.empty(0, dtype=torch.long, device=device)
            if hyperedge_index is not None and hyperedge_index.numel() > 0 and hyperedge_index.size(1) > 0:
                try:
                    conn_nodes = hyperedge_index[0]
                    conn_edges = hyperedge_index[1]
                    involved_edges = conn_edges[node_mask[conn_nodes]].unique()
                    # Filter out-of-range ids
                    candidate_edges = involved_edges[involved_edges < total_edges]
                except Exception:
                    candidate_edges = torch.empty(0, dtype=torch.long, device=device)

            _, jittered_edge_ratio = self._get_jittered_ratios()
            current_masked_edges = edge_mask.sum().item()
            target_masked_edges = int(total_edges * jittered_edge_ratio)
            
            if current_masked_edges < target_masked_edges:
                # Randomly add edges among unmasked ones to reach target
                remaining_edges = target_masked_edges - current_masked_edges
                unmasked_indices = torch.where(~edge_mask)[0]
                # Prefer candidate set (edges connected to masked nodes)
                if candidate_edges.numel() > 0 and unmasked_indices.numel() > 0:
                    in_pool = torch.zeros(total_edges, dtype=torch.bool, device=device)
                    in_pool[candidate_edges] = True
                    primary_pool = unmasked_indices[in_pool[unmasked_indices]]
                else:
                    primary_pool = torch.empty(0, dtype=torch.long, device=device)
                
                if len(unmasked_indices) > 0:
                    # Randomly choose edges to add
                    pool = primary_pool if primary_pool.numel() > 0 else unmasked_indices
                    if remaining_edges >= len(pool):
                        edge_mask[pool] = True
                    else:
                        perm = torch.randperm(len(pool), device=pool.device)
                        selected_indices = pool[perm[:remaining_edges]]
                        edge_mask[selected_indices] = True
                        
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(f"Edge mask fill: {current_masked_edges} -> {int(edge_mask.sum())}/{total_edges} "
                                     f"(target: {target_masked_edges})")
                    except Exception:
                        pass
            elif current_masked_edges > target_masked_edges:
                # Too many masked edges: dilute
                excess_edges = current_masked_edges - target_masked_edges
                masked_indices = torch.where(edge_mask)[0]
                if excess_edges > 0 and len(masked_indices) > excess_edges:
                    # Randomly deselect some edges
                    perm = torch.randperm(len(masked_indices), device=masked_indices.device)
                    deselected_indices = masked_indices[perm[:excess_edges]]
                    edge_mask[deselected_indices] = False
                    
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug(f"Edge mask dilute: {current_masked_edges} -> {int(edge_mask.sum())}/{total_edges} "
                                     f"(target: {target_masked_edges})")
                    except Exception:
                        pass
        
        # Update stats for actual semantic successes
        self.stats['semantic_attempts'] += len(unique_graphs)
        self.stats['semantic_successes'] += actual_semantic_successes
        
        logger.debug(f"Batch semantic masking: {actual_semantic_successes}/{len(unique_graphs)} molecules processed successfully with semantic masks")
        
        return node_mask, edge_mask
    
    
    def _create_semantic_node_mask(self, num_nodes: int, semantic_blocks: Dict[str, List[List[int]]],
                                 mol_info: Dict, device: torch.device) -> torch.Tensor:
        """
        Create node mask from semantic blocks with semantic/random mix.
        
        Args:
            num_nodes: total nodes
            semantic_blocks: semantic block dict
            mol_info: full molecular info
            device: device
        
        Returns:
            node mask tensor
        """
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        masked_atoms = set()
        # Compute target count with jitter; ensure nonzero for tiny molecules
        import math
        jittered_node_ratio, _ = self._get_jittered_ratios()
        if jittered_node_ratio <= 0 or num_nodes == 0:
            target_mask_count = 0
        else:
            target_mask_count = max(1, min(num_nodes, int(math.ceil(num_nodes * float(jittered_node_ratio)))))
        
        # Pre-compute valid semantic blocks
        available_blocks = []
        for block_type in self.block_types:
            if block_type in semantic_blocks:
                for block in semantic_blocks[block_type]:
                    if self._is_valid_block(block, num_nodes):
                        available_blocks.append((block_type, block))
        
        # Target split: semantic vs. random by semantic_priority
        semantic_target = int(max(1 if target_mask_count > 0 else 0,
                                  target_mask_count * self.semantic_priority))
        
        # Stage 1: select blocks via greedy knapsack over value density
        if available_blocks:
            selected_blocks = self._knapsack_block_selection(
                available_blocks, semantic_target, num_nodes, mol_info, masked_atoms
            )
            
            # Apply selected blocks; optionally jitter boundaries
            for block_type, block in selected_blocks:
                block_atoms = [atom for atom in block if atom < num_nodes and atom not in masked_atoms]
                
                # Boundary jittering to avoid perfect block borders
                if self.enable_jittering and len(block_atoms) > 2:
                    jittered_atoms = self._apply_boundary_jittering(block_atoms)
                    for atom_idx in jittered_atoms:
                        node_mask[atom_idx] = True
                        masked_atoms.add(atom_idx)
                else:
                    # Apply whole block
                    for atom_idx in block_atoms:
                        node_mask[atom_idx] = True
                        masked_atoms.add(atom_idx)
                
                if block_atoms:
                    self.stats['block_type_usage'][block_type] += 1
                    logger.debug(f"Masked {block_type} block: {len(block_atoms)} atoms")
            
            # Log achieved ratio
            actual_ratio = len(masked_atoms) / semantic_target if semantic_target > 0 else 0
            logger.debug(f"Knapsack semantic masking: {len(masked_atoms)}/{semantic_target} target "
                        f"({actual_ratio:.1%} accuracy)")
        else:
            logger.debug("No valid semantic blocks found")
        
        # Stage 2: random fill to target
        remaining_atoms = [i for i in range(num_nodes) if i not in masked_atoms]
        if remaining_atoms and len(masked_atoms) < target_mask_count:
            additional_needed = min(target_mask_count - len(masked_atoms), len(remaining_atoms))
            
            if additional_needed > 0:
                try:
                    additional_atoms = random.sample(remaining_atoms, additional_needed)
                    for atom_idx in additional_atoms:
                        node_mask[atom_idx] = True
                    
                    logger.debug(f"Added {additional_needed} random atoms to reach target ratio")
                except ValueError as e:
                    # Recovery path (rare)
                    logger.warning(f"Random sampling failed: {e}, using all remaining atoms")
                    for atom_idx in remaining_atoms:
                        node_mask[atom_idx] = True
        
        # Calculate final masked count without torch.compile graph break
        final_masked = node_mask.sum()
        
        # Ensure at least one atom when ratio>0 on tiny molecules
        if final_masked == 0 and target_mask_count > 0 and num_nodes > 0:
            logger.warning(f"No atoms masked for {num_nodes}-atom molecule (target={target_mask_count}), masking one atom by rule")
            random_atom = random.randint(0, num_nodes - 1)
            node_mask[random_atom] = True
            final_masked = 1
        
        # Convert tensor to Python scalar for logging
        final_masked_int = int(final_masked.item()) if isinstance(final_masked, torch.Tensor) else final_masked
        logger.debug(f"Final masking: {final_masked_int}/{num_nodes} = {final_masked_int/num_nodes:.1%}")
        
        return node_mask
    
    def _is_valid_block(self, block: List[int], num_nodes: int) -> bool:
        """Validate block size and index bounds."""
        if not block or len(block) < self.min_block_size or len(block) > self.max_block_size:
            return False
        
        # Check atom indices are within bounds
        return all(0 <= atom_idx < num_nodes for atom_idx in block)
    
    def _get_block_priority(self, block_type: str, block: List[int], mol_info: Dict) -> int:
        """Return priority score (0-100; lower = higher priority)."""
        # Domain-informed priorities
        base_priority = {
            'functional_group': 15,
            'ring': 25,
            'aromatic': 20,
            'chain': 35,
            'complex': 30
        }
        
        priority = base_priority.get(block_type, 50)
        
        # Important-group adjustment (preserve/enhance)
        if block_type == 'functional_group':
            functional_groups = mol_info.get('functional_group', [])
            for fg_info in functional_groups:
                if isinstance(fg_info, dict):
                    fg_type = fg_info.get('type', '')
                    fg_atoms = fg_info.get('atoms', [])
                    
                    # Check overlap with important groups
                    if fg_type in self.important_groups and set(block) & set(fg_atoms):
                        if self.preserve_important:
                            # Preserve: lower priority
                            priority = min(95, priority + 20)
                            logger.debug(f"Protected important functional group (reduced masking): {fg_type}")
                        elif self.enhance_important_masking:
                            # Learn: increase priority
                            priority = max(5, priority - 10)
                            logger.debug(f"Enhanced masking for important functional group: {fg_type}")
                        break
        
        return priority
    
    def _should_mask_block(self, block_type: str, block: List[int], mol_info: Dict) -> bool:
        """
        Decide whether to mask a semantic block.

        Args:
            block_type: block type
            block: atom indices
            mol_info: molecular info

        Returns:
            bool
        """
        # Base probabilities by type
        base_probabilities = {
            'functional_group': 0.85,
            'aromatic': 0.75,
            'ring': 0.70,
            'complex': 0.65,
            'chain': 0.55
        }
        
        mask_prob = base_probabilities.get(block_type, 0.60)
        
        # Important-group adjustments
        if block_type == 'functional_group':
            functional_groups = mol_info.get('functional_group', [])
            for fg_info in functional_groups:
                if isinstance(fg_info, dict):
                    fg_type = fg_info.get('type', '')
                    fg_atoms = fg_info.get('atoms', [])
                    
                    # Adjustments for important groups
                    if fg_type in self.important_groups and set(block) & set(fg_atoms):
                        if self.preserve_important:
                            # Preserve: reduce probability
                            mask_prob = max(0.05, mask_prob - 0.30)
                            logger.debug(f"Protected important functional group (reduced masking): {fg_type}")
                        elif self.enhance_important_masking:
                            # Learn: increase probability
                            mask_prob = min(mask_prob + 0.10, 0.95)
                            logger.debug(f"Enhanced masking for important functional group: {fg_type}")
                        break
        
        # Probability jittering to avoid memorizing patterns
        if self.enable_jittering:
            prob_noise = np.random.normal(0, self.prob_jitter_std)
            jittered_prob = mask_prob + prob_noise
            final_prob = np.clip(jittered_prob, 0.05, 0.95)
            logger.debug(f"Applied probability jittering: {mask_prob:.3f} -> {final_prob:.3f}")
        else:
            random_factor = 0.9 + 0.2 * random.random()
            final_prob = min(mask_prob * random_factor, 0.95)
        
        return random.random() < final_prob
    
    def _apply_boundary_jittering(self, block_atoms: List[int]) -> List[int]:
        """
        Randomly jitter block boundaries to avoid memorization.

        Args:
            block_atoms: atom indices

        Returns:
            list of jittered atom indices
        """
        if not self.enable_jittering or len(block_atoms) <= 2:
            return block_atoms
            
        jittered_atoms = block_atoms.copy()
        
        # Randomly drop a few boundary atoms
        if len(jittered_atoms) > 3:
            num_to_remove = max(1, int(len(jittered_atoms) * self.boundary_jitter_prob))
            indices_to_remove = random.sample(range(len(jittered_atoms)), num_to_remove)
            jittered_atoms = [atom for i, atom in enumerate(jittered_atoms) if i not in indices_to_remove]
            
        logger.debug(f"Boundary jittering: {len(block_atoms)} -> {len(jittered_atoms)} atoms")
        return jittered_atoms
    
    def _get_jittered_ratios(self) -> Tuple[float, float]:
        """
        Get jittered (node_ratio, edge_ratio); supports one-shot override.
        """
        base_node = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.node_mask_ratio
        base_edge = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.edge_mask_ratio

        if not self.enable_jittering:
            return base_node, base_edge
            
        # Add Gaussian noise to ratios
        node_noise = np.random.normal(0, self.ratio_jitter_std)
        edge_noise = np.random.normal(0, self.ratio_jitter_std)
        
        jittered_node_ratio = np.clip(base_node + node_noise, 0.1, 0.9)
        jittered_edge_ratio = np.clip(base_edge + edge_noise, 0.1, 0.9)
        
        logger.debug(f"Ratio jittering: node {self.node_mask_ratio:.3f}->{jittered_node_ratio:.3f}, "
                    f"edge {base_edge:.3f}->{jittered_edge_ratio:.3f}")
        
        return jittered_node_ratio, jittered_edge_ratio
    
    def _generate_structure_aware_hyperedge_mask(self, node_mask: torch.Tensor, 
                                               hyperedge_index: torch.Tensor, 
                                               device: torch.device) -> torch.Tensor:
        """
        Generate structure-aware edge mask.
        Mask edges that are incident to masked nodes to prevent leakage.
        
        Args:
            node_mask: [num_nodes]
            hyperedge_index: [2, num_connections] (node_idx, hyperedge_idx)
            device: device
        
        Returns:
            edge mask [num_hyperedges]
        """
        # Skip if empty
        if hyperedge_index is None or hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        
        # hyperedge_index: [node_indices, hyperedge_indices]
        node_indices, hyperedge_indices = hyperedge_index[0], hyperedge_index[1]
        
        # Edges incident to masked nodes
        masked_nodes = node_mask.nonzero(as_tuple=True)[0]
        
        # Build mask by marking participating hyperedges
        if hyperedge_indices.numel() > 0:
            # Compute number of hyperedges (max index + 1)
            max_hyperedge_idx = hyperedge_indices.max().item() if hyperedge_indices.numel() > 0 else -1
            num_hyperedges = max_hyperedge_idx + 1
            
            # Create mask tensor of correct size
            hyperedge_mask = torch.zeros(num_hyperedges, dtype=torch.bool, device=device)
            
            # For each masked node, mark its incident edges
            for masked_node in masked_nodes:
                # Indices of hyperedges incident to the node
                participating_mask = (node_indices == masked_node)
                if participating_mask.any():
                    participating_hyperedges = hyperedge_indices[participating_mask]
                    # Mark them as masked
                    if participating_hyperedges.numel() > 0:
                        hyperedge_mask.scatter_(0, participating_hyperedges, True)
        else:
            hyperedge_mask = torch.empty(0, dtype=torch.bool, device=device)
        
        # Skip debug logging to avoid torch.compile graph breaks
        # logger.debug(f"Structure-aware masking: {hyperedge_mask.sum().item()}/{hyperedge_indices.max().item() + 1 if hyperedge_indices.numel() > 0 else 0} hyperedges masked")
        return hyperedge_mask
    
    def _generate_structure_aware_hyperedge_mask_fixed(self, node_mask: torch.Tensor, 
                                                     hyperedge_index: torch.Tensor, 
                                                     num_edges: int, 
                                                     device: torch.device) -> torch.Tensor:
        """
        Generate structure-aware edge mask with fixed length = num_edges.
        
        Args:
            node_mask: [num_nodes]
            hyperedge_index: [2, num_connections] (node_idx, hyperedge_idx)
            num_edges: expected mask length
            device: device
        
        Returns:
            edge mask [num_edges]
        """
        # Skip if empty
        if hyperedge_index is None or hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            return torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        # hyperedge_index: [node_indices, hyperedge_indices]
        node_indices, hyperedge_indices = hyperedge_index[0], hyperedge_index[1]
        
        # Edges incident to masked nodes
        masked_nodes = node_mask.nonzero(as_tuple=True)[0]
        
        # Create fixed-length mask
        hyperedge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        
        # Mark edges incident to masked nodes
        if hyperedge_indices.numel() > 0 and masked_nodes.numel() > 0:
            # For each masked node, mark its incident edges
            for masked_node in masked_nodes:
                # Indices of incident hyperedges
                participating_mask = (node_indices == masked_node)
                if participating_mask.any():
                    participating_hyperedges = hyperedge_indices[participating_mask]
                    # Mark valid indices only
                    if participating_hyperedges.numel() > 0:
                        valid_hyperedges = participating_hyperedges[participating_hyperedges < num_edges]
                        if valid_hyperedges.numel() > 0:
                            hyperedge_mask.scatter_(0, valid_hyperedges, True)
        
        return hyperedge_mask
    
    def _knapsack_block_selection(self, available_blocks: List[Tuple[str, List[int]]], 
                                 target_size: int, num_nodes: int, mol_info: Dict, 
                                 masked_atoms: set) -> List[Tuple[str, List[int]]]:
        """
        Greedy block selection to approximate target size (± tolerance).

        Args:
            available_blocks: [(block_type, block_atoms), ...]
            target_size: desired number of masked atoms
            num_nodes: total nodes
            mol_info: molecular info
            masked_atoms: already masked atoms

        Returns:
            list of (block_type, block_atoms)
        """
        if not available_blocks or target_size <= 0:
            return []
        
        # Compute effective block sizes (exclude masked/out-of-bounds)
        block_items = []
        for i, (block_type, block) in enumerate(available_blocks):
            valid_atoms = [atom for atom in block if atom < num_nodes and atom not in masked_atoms]
            if not valid_atoms:
                continue
                
            size = len(valid_atoms)
            # Value by inverted priority and mask preference
            priority_score = self._get_block_priority(block_type, block, mol_info)
            mask_prob = 1.0 if self._should_mask_block(block_type, block, mol_info) else 0.3
            
            inverted_priority = max(1, 101 - priority_score)
            value = inverted_priority * mask_prob * size
            
            block_items.append({
                'index': i,
                'block_type': block_type,
                'block': valid_atoms,
                'size': size,
                'value': value,
                'priority_score': priority_score,
                'inverted_priority': inverted_priority
            })
        
        if not block_items:
            return []
        
        # Adaptive tolerance (smaller targets -> larger tolerance)
        if target_size <= 3:
            tolerance_pct = 0.15
        elif target_size <= 6:
            tolerance_pct = 0.10
        else:
            tolerance_pct = 0.05
        
        tolerance = max(1, int(target_size * tolerance_pct))
        min_target = max(1, target_size - tolerance)
        max_target = target_size + tolerance
        
        # Greedy by value density
        block_items.sort(key=lambda x: x['value'] / x['size'], reverse=True)
        
        selected_blocks = []
        current_size = 0
        
        # Pass 1: density-based greedy selection
        for item in block_items:
            if current_size + item['size'] <= max_target:
                selected_blocks.append((item['block_type'], item['block']))
                current_size += item['size']
                if current_size >= min_target:
                    break
        
        # Pass 2: fill with small blocks to reach minimum
        if current_size < min_target:
            remaining_items = [item for item in block_items 
                             if (item['block_type'], item['block']) not in 
                             [(bt, b) for bt, b in selected_blocks]]
            
            # Prefer sizes close to remaining gap
            remaining_items.sort(key=lambda x: abs((min_target - current_size) - x['size']))
            
            for item in remaining_items:
                if current_size + item['size'] <= max_target:
                    selected_blocks.append((item['block_type'], item['block']))
                    current_size += item['size']
                    if current_size >= min_target:
                        break
        
        # Pass 3: fine adjustment for semantic integrity
        if current_size < min_target and target_size >= 3:
            remaining_gap = min_target - current_size
            
            # Prefer complete small blocks first
            small_remaining_items = [item for item in block_items 
                                   if (item['block_type'], item['block']) not in 
                                   [(bt, b) for bt, b in selected_blocks] and 
                                   item['size'] <= remaining_gap]
            
            if small_remaining_items:
                # Take the largest complete small block
                best_small_item = max(small_remaining_items, key=lambda x: x['size'])
                selected_blocks.append((best_small_item['block_type'], best_small_item['block']))
                current_size += best_small_item['size']
            else:
                # Last resort: allow partial selection for chain blocks only
                chain_items = [item for item in block_items 
                             if (item['block_type'], item['block']) not in 
                             [(bt, b) for bt, b in selected_blocks] and 
                             item['block_type'] == 'chain' and
                             item['size'] > remaining_gap]
                
                if chain_items:
                    # Pick closest chain block and use a prefix
                    best_chain = min(chain_items, key=lambda x: x['size'] - remaining_gap)
                    partial_block = best_chain['block'][:remaining_gap]
                    selected_blocks.append((best_chain['block_type'], partial_block))
                    current_size += len(partial_block)
                    logger.debug(f"Applied partial chain selection: {len(partial_block)}/{best_chain['size']} atoms")
                else:
                    logger.debug(f"No suitable blocks for gap filling ({remaining_gap} atoms needed)")
        
        logger.debug(f"Knapsack selected {len(selected_blocks)} blocks, "
                    f"total size: {current_size}/{target_size} "
                    f"({current_size/target_size:.1%})")
        
        return selected_blocks
    
    def _load_semantic_cache(self):
        """Load preprocessed semantic-block cache from file."""
        try:
            import pickle
            from pathlib import Path
            
            cache_path = Path(self.cache_file)
            if not cache_path.exists():
                logger.warning(f"Cache file not found: {cache_path}")
                return
            
            with open(cache_path, 'rb') as f:
                self.semantic_cache = pickle.load(f)
            
            logger.info(f"Successfully loaded semantic cache from {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to load semantic cache: {e}")
            self.semantic_cache = {}
    
    def _get_cached_mol_info(self, smiles: str) -> Optional[Dict]:
        """Get molecular info from cache or compute on demand."""
        if not smiles:
            return None
        
        # Try cache first
        if self.enable_cache and smiles in self.semantic_cache:
            self.stats['cache_hits'] += 1
            cached_data = self.semantic_cache[smiles]
            # Convert to analyzer.analyze_molecule format
            return {
                'semantic_blocks': cached_data.get('semantic_blocks', {}),
                'functional_group': cached_data.get('functional_group', []),
                'ring_systems': cached_data.get('ring_systems', []),
                'atom_annotations': cached_data.get('atom_annotations', {}),
                'num_atoms': cached_data.get('num_atoms', 0),
                'num_bonds': cached_data.get('num_bonds', 0)
            }
        
        # Cache miss -> compute
        self.stats['cache_misses'] += 1
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            return self.analyzer.analyze_molecule(mol, smiles)
            
        except Exception as e:
            logger.warning(f"Failed to analyze molecule {smiles}: {e}")
            return None
    
    def set_mask_seed(self, step: int, epoch: int = 0):
        """Set deterministic seed for reproducible masking (optional)."""
        if self.use_deterministic:
            # Derive deterministic seed from (step, epoch)
            mask_seed = self.base_seed + step * 1000 + epoch * 100000
            random.seed(mask_seed)
            torch.manual_seed(mask_seed)
            try:
                np.random.seed(mask_seed)
            except Exception:
                pass
            # CUDA deterministic if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(mask_seed)
                except Exception:
                    pass
            
            # Save seed for debugging
            self.mask_rng_state = {
                'step': step,
                'epoch': epoch,
                'seed': mask_seed
            }
            
            logger.debug(f"Set mask seed: {mask_seed} (step={step}, epoch={epoch})")
    
    def get_mask_state(self) -> Dict:
        """Get current RNG state used for masking."""
        return {
            'rng_state': self.mask_rng_state,
            'stats': self.stats.copy()
        }
    
    def _generate_edge_mask(self, num_edges: int, device: torch.device) -> torch.Tensor:
        """Generate edge mask with simple random strategy."""
        if num_edges == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        base_edge = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else self.edge_mask_ratio
        return torch.rand(num_edges, device=device) < base_edge
    
    def _generate_random_masks(self, num_nodes: int, num_edges: int, 
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random fallback masks with tiny-molecule safeguards.
        
        Args:
            num_nodes: number of nodes
            num_edges: number of hyperedges
            device: device
        
        Returns:
            (node_mask, edge_mask)
        """
        # Strict ratio with mild conservatism on tiny molecules; ensure non-empty
        base_ratio = float(self._ratio_override) if getattr(self, '_ratio_override', None) is not None else float(self.node_mask_ratio)
        if num_nodes <= 6:
            target_ratio = min(base_ratio * 0.5, 0.4)
        elif num_nodes <= 12:
            target_ratio = min(base_ratio * 0.7, 0.55)
        else:
            target_ratio = base_ratio

        import math
        k = 0 if target_ratio <= 0 or num_nodes == 0 else max(1, min(num_nodes, int(math.ceil(num_nodes * target_ratio))))
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if k > 0:
            idx = torch.randperm(num_nodes, device=device)[:k]
            node_mask[idx] = True

        edge_mask = self._generate_edge_mask(num_edges, device)
        
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(f"Random masking applied: {int(node_mask.sum())}/{num_nodes} nodes, "
                             f"{int(edge_mask.sum())}/{num_edges} edges")
            except Exception:
                pass
        
        return node_mask, edge_mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for the masking strategy."""
        total_attempts = self.stats['semantic_attempts'] + self.stats['random_fallbacks']
        
        return {
            'total_masks_generated': self.stats['total_masks_generated'],
            'semantic_success_rate': (
                self.stats['semantic_successes'] / max(self.stats['semantic_attempts'], 1)
            ),
            'semantic_usage_rate': (
                self.stats['semantic_attempts'] / max(total_attempts, 1)
            ),
            'random_fallback_rate': (
                self.stats['random_fallbacks'] / max(total_attempts, 1)
            ),
            'block_type_usage': dict(self.stats['block_type_usage']),
            'config': {
                'semantic_priority': self.semantic_priority,
                'node_mask_ratio': self.node_mask_ratio,
                'edge_mask_ratio': self.edge_mask_ratio,
                'block_types': self.block_types,
                'important_groups': list(self.important_groups)
            }
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        from collections import defaultdict
        self.stats = {
            'semantic_attempts': 0,
            'semantic_successes': 0,
            'random_fallbacks': 0,
            'total_masks_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'block_type_usage': defaultdict(int)
        }
        for bt in self.block_types:
            self.stats['block_type_usage'][bt] = 0


def create_semantic_masking(config: Dict = None) -> SemanticMasking:
    """Factory to create a semantic masking strategy instance."""
    return SemanticMasking(config)
