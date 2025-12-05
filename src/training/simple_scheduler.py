"""
Simple masking scheduler (KISS).

Focuses on:
- Strategy creation and management
- Molecular analysis caching
- Simple strategy selection
- Basic performance monitoring

Complex features intentionally removed:
- Curriculum progression management
- Bandit-based strategy selection
- Heavy performance history tracking
- Dynamic difficulty adjustment
"""

import torch
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path
from rdkit import Chem

from .semantic_masking import SemanticMasking, create_semantic_masking
from .masking_strategies import MaskingStrategy, create_masking_strategy
from ..data.molecular_semantics import MolecularSemanticAnalyzer

logger = logging.getLogger(__name__)


class SimpleMaskingScheduler:
    """
    Minimal masking scheduler with caching and basic stats.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.masking_config = config.get('masking', {})
        
        # Strategy management
        self.strategy_type = self.masking_config.get('strategy', 'semantic')
        self.strategy = None
        
        # Molecular analysis cache
        self.molecular_cache = {}
        self.cache_size_limit = self.masking_config.get('cache_size_limit', 1000)
        self.cache_access_order = []  # LRU access order
        
        # Molecular semantic analyzer
        self.analyzer = MolecularSemanticAnalyzer()
        
        # Basic performance stats
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_analysis_time': 0.0,
            'mask_generation_time': 0.0,
            'error_count': 0,
            'strategy_usage': defaultdict(int)
        }
        
        # Initialize strategy
        self._initialize_strategy()
        
        logger.info(f"SimpleMaskingScheduler initialized with strategy: {self.strategy_type}")
    
    def _initialize_strategy(self):
        """Initialize masking strategy."""
        try:
            if self.strategy_type == 'semantic':
                self.strategy = create_semantic_masking(self.masking_config)
            else:
                self.strategy = create_masking_strategy(self.strategy_type, self.masking_config)
            
            logger.info(f"Successfully initialized {self.strategy_type} masking strategy")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.strategy_type} strategy: {e}")
            
            # Fallback to random strategy
            logger.warning("Falling back to random masking strategy")
            self.strategy = create_masking_strategy('random', {'mask_ratio': 0.7})
            self.strategy_type = 'random'
    
    def generate_masks(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                       hyperedge_attr: torch.Tensor, global_step: int = 0,
                       max_steps: int = 1000, epoch: int = 0, smiles: str = None,
                       mol: Chem.Mol = None, recent_loss: float = None,
                       **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks (simplified path).

        Args:
            x: Node features
            hyperedge_index: Hyperedge incidence
            hyperedge_attr: Hyperedge attributes
            global_step: Kept for compatibility
            max_steps: Kept for compatibility
            epoch: Kept for compatibility
            smiles: Optional SMILES
            mol: Optional RDKit Mol
            recent_loss: Kept for compatibility

        Returns:
            Tuple of (node_mask, edge_mask)
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        self.stats['strategy_usage'][self.strategy_type] += 1
        
        try:
            # Get molecular info (via cache)
            mol_info = None
            if smiles and self.strategy_type == 'semantic':
                # For batch SMILES, let strategy handle per-molecule; avoid hashing lists
                if isinstance(smiles, (list, tuple)):
                    mol_info = None
                else:
                    mol_info = self._get_molecular_info(smiles)
            
            # Generate masks
            mask_start_time = time.time()
            node_mask, edge_mask = self.strategy.generate_masks(
                x, hyperedge_index, hyperedge_attr,
                smiles=smiles, mol_info=mol_info,
                epoch=epoch, **kwargs
            )
            
            self.stats['mask_generation_time'] += time.time() - mask_start_time
            
            # Validate masks
            self._validate_masks(node_mask, edge_mask, x.size(0), 
                               hyperedge_attr.size(0) if hyperedge_attr is not None else 0)
            
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    nm = int(node_mask.sum())
                    em = int(edge_mask.sum()) if edge_mask.numel() > 0 else 0
                    logger.debug(f"Generated masks: {nm}/{node_mask.size(0)} nodes, {em}/{edge_mask.size(0)} edges")
                except Exception:
                    pass
            
            return node_mask, edge_mask
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Error in mask generation: {e}")
            
            # Emergency fallback: simple random masks
            return self._generate_emergency_masks(x, hyperedge_attr)
        
        finally:
            total_time = time.time() - start_time
            logger.debug(f"Mask generation took {total_time:.4f}s")
    
    def _get_molecular_info(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Get molecular analysis info with simple LRU cache.

        Args:
            smiles: SMILES string

        Returns:
            Molecular analysis result or None
        """
        if not smiles:
            return None
        
        # Cache lookup
        if smiles in self.molecular_cache:
            self.stats['cache_hits'] += 1
            
            # Update LRU order
            if smiles in self.cache_access_order:
                self.cache_access_order.remove(smiles)
            self.cache_access_order.append(smiles)
            
            return self.molecular_cache[smiles]
        
        # Cache miss: analyze molecule
        self.stats['cache_misses'] += 1
        
        try:
            analysis_start_time = time.time()
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            mol_info = self.analyzer.analyze_molecule(mol, smiles)
            
            self.stats['semantic_analysis_time'] += time.time() - analysis_start_time
            
            # Add to cache
            self._add_to_cache(smiles, mol_info)
            
            return mol_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze molecule {smiles}: {e}")
            return None
    
    def _add_to_cache(self, smiles: str, mol_info: Dict[str, Any]):
        """
        Add molecular info to cache (LRU policy).

        Args:
            smiles: SMILES string
            mol_info: Molecular analysis result
        """
        # Enforce cache size limit
        while len(self.molecular_cache) >= self.cache_size_limit:
            # Evict least-recently-used item
            if self.cache_access_order:
                oldest_smiles = self.cache_access_order.pop(0)
                if oldest_smiles in self.molecular_cache:
                    del self.molecular_cache[oldest_smiles]
            else:
                break
        
        # Insert new entry
        self.molecular_cache[smiles] = mol_info
        self.cache_access_order.append(smiles)
        
        logger.debug(f"Added {smiles} to molecular cache (size: {len(self.molecular_cache)})")
    
    def _validate_masks(self, node_mask: torch.Tensor, edge_mask: torch.Tensor,
                        num_nodes: int, num_edges: int):
        """
        Validate mask shapes and basic ratios.

        Args:
            node_mask: Node mask
            edge_mask: Edge mask
            num_nodes: Expected node count
            num_edges: Expected edge count
        """
        # Shape checks
        if node_mask.size(0) != num_nodes:
            raise ValueError(f"Node mask size {node_mask.size(0)} doesn't match num_nodes {num_nodes}")
        
        if edge_mask.size(0) != num_edges:
            raise ValueError(f"Edge mask size {edge_mask.size(0)} doesn't match num_edges {num_edges}")
        
        # Dtype checks
        if node_mask.dtype != torch.bool:
            logger.warning(f"Node mask dtype is {node_mask.dtype}, expected bool")
        
        if edge_mask.dtype != torch.bool:
            logger.warning(f"Edge mask dtype is {edge_mask.dtype}, expected bool")
        
        # Ratio checks (avoid .item() during compile)
        node_mask_ratio = node_mask.float().mean()
        edge_mask_ratio = edge_mask.float().mean() if num_edges > 0 else 0.0
        
        # Convert to scalars only for logging
        if node_mask_ratio < 0.1 or node_mask_ratio > 0.95:
            logger.warning(f"Unusual node mask ratio: {node_mask_ratio.item():.3f}")
        
        if num_edges > 0 and (edge_mask_ratio < 0.1 or edge_mask_ratio > 0.95):
            logger.warning(f"Unusual edge mask ratio: {edge_mask_ratio.item():.3f}")
    
    def _generate_emergency_masks(self, x: torch.Tensor,
                                  hyperedge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Emergency fallback: generate simple random masks.

        Args:
            x: Node features
            hyperedge_attr: Hyperedge attributes

        Returns:
            Tuple of (node_mask, edge_mask)
        """
        num_nodes = x.size(0)
        num_edges = hyperedge_attr.size(0) if hyperedge_attr is not None else 0
        device = x.device
        
        # Simple random masks
        node_mask = torch.rand(num_nodes, device=device) < 0.7
        edge_mask = torch.rand(num_edges, device=device) < 0.7 if num_edges > 0 else torch.empty(0, dtype=torch.bool, device=device)
        
        logger.warning(f"Using emergency random masks: {node_mask.sum().item()}/{num_nodes} nodes")
        
        return node_mask, edge_mask
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        """
        total_calls = max(self.stats['total_calls'], 1)
        cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
        
        summary = {
            'scheduler_stats': {
                'strategy_type': self.strategy_type,
                'total_calls': self.stats['total_calls'],
                'error_rate': self.stats['error_count'] / total_calls,
                'avg_mask_time': self.stats['mask_generation_time'] / total_calls,
                'strategy_usage': dict(self.stats['strategy_usage'])
            },
            'cache_stats': {
                'cache_size': len(self.molecular_cache),
                'cache_limit': self.cache_size_limit,
                'cache_hit_rate': self.stats['cache_hits'] / max(cache_total, 1),
                'total_cache_requests': cache_total,
                'avg_analysis_time': (
                    self.stats['semantic_analysis_time'] / max(self.stats['cache_misses'], 1)
                )
            }
        }
        
        # Append strategy-specific stats if available
        if hasattr(self.strategy, 'get_statistics'):
            summary['strategy_stats'] = self.strategy.get_statistics()
        
        return summary
    
    def clear_cache(self):
        """Clear molecular analysis cache."""
        self.molecular_cache.clear()
        self.cache_access_order.clear()
        logger.info("Molecular analysis cache cleared")
    
    def save_cache(self, filepath: str):
        """
        Save cache to a file.

        Args:
            filepath: Output path
        """
        try:
            import pickle
            cache_data = {
                'molecular_cache': self.molecular_cache,
                'cache_access_order': self.cache_access_order,
                'stats': self.stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, filepath: str):
        """
        Load cache from a file.

        Args:
            filepath: Input path
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.molecular_cache = cache_data.get('molecular_cache', {})
            self.cache_access_order = cache_data.get('cache_access_order', [])
            
            # Optionally load saved stats
            saved_stats = cache_data.get('stats', {})
            for key in ['cache_hits', 'cache_misses', 'semantic_analysis_time']:
                if key in saved_stats:
                    self.stats[key] = saved_stats[key]
            
            logger.info(f"Cache loaded from {filepath} (size: {len(self.molecular_cache)})")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


def create_simple_masking_scheduler(config: Dict) -> SimpleMaskingScheduler:
    """Create a simple masking scheduler instance."""
    return SimpleMaskingScheduler(config)
