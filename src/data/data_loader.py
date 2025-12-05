"""
Data loading utilities for HyperGraph-MAE.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import logging
import random
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Global target hyperedge feature dimension for collate_fn alignment.
# If None, collate will fall back to per-batch max dimension logic.
_TARGET_HYPEREDGE_DIM: Optional[int] = None


def set_collate_hyperedge_dim(dim: Optional[int]):
    """Set a global target hyperedge feature dimension for batching.

    Args:
        dim: Target feature dimension. If None, per-batch max dimension is used.
    """
    global _TARGET_HYPEREDGE_DIM
    if dim is not None and dim <= 0:
        raise ValueError("collate hyperedge dim must be positive")
    _TARGET_HYPEREDGE_DIM = dim


class MolecularHypergraphDataset(Dataset):
    """
    Dataset for molecular hypergraphs.
    """

    def __init__(self, data_dir: Path, transform: Optional[Callable] = None,
                 max_graphs: Optional[int] = None):
        """
        Initialize molecular hypergraph dataset.

        Args:
            data_dir: Directory containing preprocessed data files
            transform: Optional transform to apply to data
            max_graphs: Maximum number of graphs to load
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.graphs = []

        # Load graphs
        self._load_graphs(max_graphs)

        if not self.graphs:
            raise ValueError(f"No valid graphs found in {data_dir}")

        logger.info(f"Loaded {len(self.graphs)} molecular graphs")

    def _load_graphs(self, max_graphs: Optional[int] = None):
        """Load graphs from preprocessed files."""
        data_files = sorted(self.data_dir.glob("batch_*.pt"))

        if not data_files:
            raise FileNotFoundError(f"No preprocessed data found in {self.data_dir}")

        loaded_count = 0

        for file in tqdm(data_files, desc="Loading data"):
            try:
                # Work around PyTorch 2.6 weights_only quirk; explicit file handle
                with open(file, 'rb') as f:
                    batch_graphs = torch.load(f, map_location="cpu", weights_only=False)

                for g in batch_graphs:
                    if g is None:
                        continue

                    # Ensure correct data types
                    g = self._validate_graph(g)
                    if g is not None:
                        self.graphs.append(g)
                        loaded_count += 1

                        if max_graphs is not None and loaded_count >= max_graphs:
                            return

            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
                continue

    def _validate_graph(self, graph: Data) -> Optional[Data]:
        """Validate and fix graph data."""
        try:
            # Ensure all tensors are float32 and contiguous
            graph.x = graph.x.float().contiguous()

            if hasattr(graph, 'hyperedge_attr') and graph.hyperedge_attr is not None:
                graph.hyperedge_attr = graph.hyperedge_attr.float().contiguous()
            else:
                # Read dimension from data (single source of truth)
                default_dim = getattr(graph, 'hyperedge_dim', 1)  # Use embedded dimension
                graph.hyperedge_attr = torch.zeros((0, default_dim), dtype=torch.float32)

            if hasattr(graph, 'hyperedge_index') and graph.hyperedge_index is not None:
                graph.hyperedge_index = graph.hyperedge_index.long().contiguous()
            else:
                graph.hyperedge_index = torch.zeros((2, 0), dtype=torch.long)

            # Ensure masks exist
            if not hasattr(graph, 'node_mask'):
                graph.node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool)
            else:
                graph.node_mask = graph.node_mask.bool().contiguous()

            if not hasattr(graph, 'edge_mask'):
                num_edges = graph.hyperedge_attr.size(0)
                graph.edge_mask = torch.zeros(num_edges, dtype=torch.bool)
            else:
                graph.edge_mask = graph.edge_mask.bool().contiguous()

            return graph

        except Exception as e:
            logger.debug(f"Invalid graph: {e}")
            return None

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Get a graph by index."""
        graph = self.graphs[idx]

        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_feature_dim(self) -> int:
        """Get input feature dimension."""
        if len(self.graphs) > 0:
            return self.graphs[0].x.size(1)
        return 0

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_graphs': len(self.graphs),
            'num_nodes': [],
            'num_edges': [],
            'feature_dim': self.get_feature_dim()
        }

        for g in self.graphs:
            stats['num_nodes'].append(g.x.size(0))
            if hasattr(g, 'hyperedge_index') and g.hyperedge_index.numel() > 0:
                num_edges = int(g.hyperedge_index[1].max().item() + 1)
                stats['num_edges'].append(num_edges)
            else:
                stats['num_edges'].append(0)

        # Calculate summary statistics
        stats['avg_nodes'] = np.mean(stats['num_nodes'])
        stats['avg_edges'] = np.mean(stats['num_edges'])
        stats['std_nodes'] = np.std(stats['num_nodes'])
        stats['std_edges'] = np.std(stats['num_edges'])

        return stats


def custom_collate_fn(data_list: List[Data]) -> Batch:
    """
    Optimized collate function for batching hypergraphs with CUDA memory alignment safety.
    """
    from torch_geometric.data import Data as PYGData  # Import at function top
    
    if not data_list:
        raise RuntimeError("Empty batch after filtering")
    
    # Pre-validate data for memory alignment issues
    for i, data in enumerate(data_list):
        if hasattr(data, 'x') and data.x is not None:
            if not data.x.is_contiguous():
                logger.debug(f"Graph {i}: Non-contiguous x tensor, fixing...")
                data.x = data.x.contiguous()
        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            if not data.hyperedge_attr.is_contiguous():
                logger.debug(f"Graph {i}: Non-contiguous hyperedge_attr tensor, fixing...")
                data.hyperedge_attr = data.hyperedge_attr.contiguous()
        if hasattr(data, 'hyperedge_index') and data.hyperedge_index is not None:
            if not data.hyperedge_index.is_contiguous():
                logger.debug(f"Graph {i}: Non-contiguous hyperedge_index tensor, fixing...")
                data.hyperedge_index = data.hyperedge_index.contiguous()
    
    # Sanitize inputs: keep only essential fields so PyG doesn't collate
    # unsupported Python objects. Keep SMILES per-graph as a passthrough list
    # for semantic masking (not part of PyG's internal collate).
    essential_keys = {
        'x', 'hyperedge_index', 'hyperedge_attr', 'node_mask', 'edge_mask', 'y'
    }
    cleaned_data_list = []
    smiles_list: List[str] = []
    for d in data_list:
        payload = {}
        # Collect SMILES as list; attach to output Data later
        if hasattr(d, 'smiles') and getattr(d, 'smiles') is not None:
            try:
                smiles_list.append(str(getattr(d, 'smiles')))
            except Exception:
                smiles_list.append(None)
        else:
            smiles_list.append(None)
        # x
        if hasattr(d, 'x') and getattr(d, 'x') is not None:
            x = d.x
            if not x.is_floating_point():
                x = x.float()
            # Only make contiguous if necessary, avoid unnecessary clone
            payload['x'] = x.contiguous() if not x.is_contiguous() else x
        # hyperedge_index
        if hasattr(d, 'hyperedge_index') and getattr(d, 'hyperedge_index') is not None:
            he_idx = d.hyperedge_index
            he_idx = he_idx.long()
            # Only make contiguous if necessary
            payload['hyperedge_index'] = he_idx.contiguous() if not he_idx.is_contiguous() else he_idx
        # hyperedge_attr
        if hasattr(d, 'hyperedge_attr') and getattr(d, 'hyperedge_attr') is not None:
            he_attr = d.hyperedge_attr
            if not he_attr.is_floating_point():
                he_attr = he_attr.float()
            # Only make contiguous if necessary
            payload['hyperedge_attr'] = he_attr.contiguous() if not he_attr.is_contiguous() else he_attr
        # masks
        if hasattr(d, 'node_mask') and getattr(d, 'node_mask') is not None:
            nm = d.node_mask.bool()
            # Only make contiguous if necessary
            payload['node_mask'] = nm.contiguous() if not nm.is_contiguous() else nm
        if hasattr(d, 'edge_mask') and getattr(d, 'edge_mask') is not None:
            em = d.edge_mask.bool()
            # Only make contiguous if necessary
            payload['edge_mask'] = em.contiguous() if not em.is_contiguous() else em
        # y (optional)
        if hasattr(d, 'y') and getattr(d, 'y') is not None:
            y = d.y
            # Only clone if it has the method (for non-tensor types)
            payload['y'] = y.contiguous() if hasattr(y, 'is_contiguous') and not y.is_contiguous() else y

        # rdkit_idx (optional): per-node mapping to RDKit atom index, -1 for non-atom nodes
        if hasattr(d, 'rdkit_idx') and getattr(d, 'rdkit_idx') is not None:
            rki = d.rdkit_idx.long()
            payload['rdkit_idx'] = rki.contiguous() if not rki.is_contiguous() else rki

        cleaned_data_list.append(PYGData(**payload))

    # Manually collate to avoid PyG's internal resizing on heterogeneous fields
    # 1) Concatenate node features
    xs = []
    node_sizes = []
    for d in cleaned_data_list:
        x = getattr(d, 'x', None)
        if x is None:
            x = torch.zeros((0, 0), dtype=torch.float32)
        xs.append(x)
        node_sizes.append(x.size(0))
    if len(xs) > 0 and xs[0].numel() > 0:
        cat_x = torch.cat(xs, dim=0)
    else:
        # Fallback when graphs might be empty
        total_nodes = sum(node_sizes)
        feat_dim = xs[0].size(1) if len(xs) > 0 and xs[0].dim() == 2 else 0
        cat_x = torch.zeros((total_nodes, feat_dim), dtype=torch.float32)
    
    # Handle hyperedge batching with pre-allocated tensors
    hyperedge_indices = []
    hyperedge_attrs = []
    # Track per-graph unique hyperedges to correctly remap edge masks
    unique_edges_per_graph = []
    num_edges_per_graph = []
    node_offset = 0
    hyperedge_offset = 0

    for data in cleaned_data_list:
        he_index = data.hyperedge_index  # Avoid unnecessary clone

        if he_index.numel() > 0:
            # Get unique hyperedges and remap indices
            unique_edges, inverse = torch.unique(he_index[1], return_inverse=True)
            num_edges = unique_edges.size(0)
            unique_edges_per_graph.append(unique_edges)
            num_edges_per_graph.append(int(num_edges))

            # Efficient index adjustment (reduced operations)
            adjusted_he_index = he_index.clone()  # Only clone when necessary
            adjusted_he_index[0] += node_offset  # In-place addition for nodes
            adjusted_he_index[1] = inverse + hyperedge_offset
            hyperedge_indices.append(adjusted_he_index)

            # Extract corresponding attributes
            if data.hyperedge_attr.size(0) > 0:
                edge_attrs = data.hyperedge_attr[unique_edges]
                hyperedge_attrs.append(edge_attrs)
            else:
                # Create zero attributes using global target or infer from cleaned data list
                if _TARGET_HYPEREDGE_DIM is not None:
                    ref_dim = int(_TARGET_HYPEREDGE_DIM)
                else:
                    ref_dim = next((
                        d.hyperedge_attr.size(1)
                        for d in cleaned_data_list
                        if hasattr(d, 'hyperedge_attr') and getattr(d, 'hyperedge_attr') is not None
                        and d.hyperedge_attr.dim() == 2 and d.hyperedge_attr.size(1) > 0
                    ), 1)
                hyperedge_attrs.append(torch.zeros((num_edges, ref_dim), dtype=torch.float32))

            hyperedge_offset += num_edges
        else:
            # Handle empty hyperedges
            # Use dimension from data object if available
            if _TARGET_HYPEREDGE_DIM is not None:
                attr_dim = int(_TARGET_HYPEREDGE_DIM)
            else:
                attr_dim = next((
                    d.hyperedge_attr.size(1)
                    for d in cleaned_data_list
                    if hasattr(d, 'hyperedge_attr') and getattr(d, 'hyperedge_attr') is not None
                    and d.hyperedge_attr.dim() == 2 and d.hyperedge_attr.size(1) > 0
                ), 1)
            hyperedge_attrs.append(torch.zeros((0, attr_dim), dtype=torch.float32))
            unique_edges_per_graph.append(torch.zeros((0,), dtype=torch.long))
            num_edges_per_graph.append(0)

        node_offset += data.num_nodes

    # Combine hyperedge data
    if hyperedge_indices:
        batched_hyperedge_index = torch.cat(hyperedge_indices, dim=1)
    else:
        batched_hyperedge_index = torch.zeros((2, 0), dtype=torch.long)

    if hyperedge_attrs:
        # Determine target dimension: prefer global target if set, otherwise per-batch fallback
        if _TARGET_HYPEREDGE_DIM is not None:
            attr_dim = int(_TARGET_HYPEREDGE_DIM)
        else:
            dims = [attr.size(1) for attr in hyperedge_attrs if attr.numel() > 0]
            if dims:
                attr_dim = max(dims)
            else:
                # All attributes are empty, infer from cleaned data or default to 1
                attr_dim = next((
                    d.hyperedge_attr.size(1)
                    for d in cleaned_data_list
                    if hasattr(d, 'hyperedge_attr') and getattr(d, 'hyperedge_attr') is not None
                    and d.hyperedge_attr.dim() == 2 and d.hyperedge_attr.size(1) > 0
                ), 1)

        normalized_attrs = []
        for attr in hyperedge_attrs:
            if attr.numel() == 0:
                continue
            cur_dim = attr.size(1)
            if cur_dim < attr_dim:
                # Pad attributes
                padding = torch.zeros(
                    (attr.size(0), attr_dim - cur_dim),
                    dtype=torch.float32,
                    device=attr.device,
                )
                attr = torch.cat([attr, padding], dim=1)
            elif cur_dim > attr_dim:
                # Truncate attributes
                attr = attr[:, :attr_dim]
            normalized_attrs.append(attr)

        if normalized_attrs:
            batched_hyperedge_attr = torch.cat(normalized_attrs, dim=0)
        else:
            batched_hyperedge_attr = torch.zeros((0, attr_dim), dtype=torch.float32)
    else:
        # No hyperedges - use target dim if set, else from first data object
        if _TARGET_HYPEREDGE_DIM is not None:
            ref_dim = int(_TARGET_HYPEREDGE_DIM)
        else:
            ref_dim = next((
                d.hyperedge_attr.size(1)
                for d in cleaned_data_list
                if hasattr(d, 'hyperedge_attr') and getattr(d, 'hyperedge_attr') is not None
                and d.hyperedge_attr.dim() == 2 and d.hyperedge_attr.size(1) > 0
            ), 1)
        batched_hyperedge_attr = torch.zeros((0, ref_dim), dtype=torch.float32)

    # Combine masks
    batched_node_mask = torch.cat([data.node_mask for data in cleaned_data_list], dim=0)

    # Combine rdkit_idx if present; otherwise fill with arange per-graph
    rdkit_idx_list = []
    for d in cleaned_data_list:
        if hasattr(d, 'rdkit_idx') and getattr(d, 'rdkit_idx') is not None and d.rdkit_idx.numel() == d.num_nodes:
            rdkit_idx_list.append(d.rdkit_idx.long())
        else:
            # Default to 0..n-1 identity mapping
            rdkit_idx_list.append(torch.arange(d.num_nodes, dtype=torch.long))
    batched_rdkit_idx = torch.cat(rdkit_idx_list, dim=0) if rdkit_idx_list else torch.zeros((0,), dtype=torch.long)

    edge_masks = []
    for i, data in enumerate(cleaned_data_list):
        num_edges = num_edges_per_graph[i] if i < len(num_edges_per_graph) else (
            int(torch.unique(data.hyperedge_index[1]).size(0)) if data.hyperedge_index.numel() > 0 else 0
        )
        if hasattr(data, 'edge_mask') and data.edge_mask is not None and data.edge_mask.numel() > 0:
            if num_edges > 0:
                ue = unique_edges_per_graph[i] if i < len(unique_edges_per_graph) else torch.unique(data.hyperedge_index[1])
                edge_masks.append(data.edge_mask[ue])
            else:
                edge_masks.append(torch.zeros(0, dtype=torch.bool))
        else:
            # Create zero mask aligned to the deduplicated edges
            edge_masks.append(torch.zeros(num_edges, dtype=torch.bool))

    if edge_masks:
        batched_edge_mask = torch.cat(edge_masks, dim=0)
    else:
        batched_edge_mask = torch.zeros(0, dtype=torch.bool)

    # Build batch index (maps each node to its graph id) for downstream pooling
    if node_sizes:
        batch_index_parts = [
            torch.full((n,), i, dtype=torch.long) if n > 0 else torch.zeros((0,), dtype=torch.long)
            for i, n in enumerate(node_sizes)
        ]
        batch_index = torch.cat(batch_index_parts, dim=0) if batch_index_parts else torch.zeros((0,), dtype=torch.long)
    else:
        batch_index = torch.zeros((0,), dtype=torch.long)

    # Collate labels (y) and optional y_mask for downstream tasks
    y_list = []
    y_mask_list = []
    for d in cleaned_data_list:
        y = getattr(d, 'y', None)
        m = getattr(d, 'y_mask', None)
        # Normalize y to 1D float tensor if present
        if y is not None:
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            y = y.view(-1).to(torch.float32)
        # Normalize mask to 1D bool tensor if present
        if m is not None:
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)
            m = m.view(-1).to(torch.bool)
        y_list.append(y)
        y_mask_list.append(m)

    batched_y = None
    batched_y_mask = None
    if any(y is not None for y in y_list):
        # Determine target dimension
        dims = [int(y.numel()) for y in y_list if y is not None]
        y_dim = max(dims) if dims else 1
        stacked_y = []
        stacked_m = []
        for y, m in zip(y_list, y_mask_list):
            if y is None:
                y = torch.full((y_dim,), float('nan'), dtype=torch.float32)
                m = torch.zeros((y_dim,), dtype=torch.bool) if m is None else m
            else:
                if y.numel() < y_dim:
                    pad = torch.full((y_dim - y.numel(),), float('nan'), dtype=y.dtype, device=y.device)
                    y = torch.cat([y, pad], dim=0)
                elif y.numel() > y_dim:
                    y = y[:y_dim]
                if m is None:
                    # Default mask: valid where label is finite
                    m = torch.isfinite(y)
            if m.numel() < y_dim:
                m = torch.cat([m, torch.zeros((y_dim - m.numel(),), dtype=torch.bool, device=m.device)], dim=0)
            elif m.numel() > y_dim:
                m = m[:y_dim]
            stacked_y.append(y)
            stacked_m.append(m)
        batched_y = torch.stack(stacked_y, dim=0)
        batched_y_mask = torch.stack(stacked_m, dim=0) if stacked_m else None

    # Build final Data object with proper memory alignment
    # Ensure all tensors are properly aligned for CUDA operations
    def ensure_aligned_tensor(tensor, dtype=None):
        """Ensure tensor is properly aligned for CUDA operations."""
        if tensor is None:
            return tensor
        if dtype is not None:
            tensor = tensor.to(dtype)
        # Ensure contiguous memory layout and proper alignment
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # For PyTorch Geometric compatibility, ensure tensors are on the same device
        return tensor
    
    out = PYGData(
        x=ensure_aligned_tensor(cat_x, dtype=torch.float32),
        hyperedge_index=ensure_aligned_tensor(batched_hyperedge_index, dtype=torch.long),
        hyperedge_attr=ensure_aligned_tensor(batched_hyperedge_attr, dtype=torch.float32),
        node_mask=ensure_aligned_tensor(batched_node_mask, dtype=torch.bool),
        edge_mask=ensure_aligned_tensor(batched_edge_mask, dtype=torch.bool),
        rdkit_idx=ensure_aligned_tensor(batched_rdkit_idx, dtype=torch.long),
        y=ensure_aligned_tensor(batched_y, dtype=torch.float32) if batched_y is not None else None,
        y_mask=ensure_aligned_tensor(batched_y_mask, dtype=torch.bool) if batched_y_mask is not None else None,
    )

    # Attach batch-related helpers expected by downstream code
    out.batch = batch_index.contiguous()
    out.num_graphs = len(cleaned_data_list)
    # Passthrough SMILES list for semantic masking during training
    out.smiles = smiles_list

    return out


def create_data_loaders(train_data: Dataset, val_data: Dataset,
                        batch_size: int = 256, num_workers: int = None,
                        pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create GPU-optimized data loaders for training and validation.

    Args:
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes (auto-optimized if None)
        pin_memory: Whether to use pinned memory

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # GPU-optimized worker configuration
    if num_workers is None:
        # Auto-configure based on GPU availability and CPU cores
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        if torch.cuda.is_available():
            # For GPU training, use moderate worker count
            num_workers = min(cpu_cores // 2, 8)
        else:
            # For CPU training, use more workers
            num_workers = min(cpu_cores, 8)
    
    # Optimize settings for GPU training - conservative FD management
    persistent_workers = False  # Disable to prevent FD accumulation across epochs
    prefetch_factor = 1 if num_workers > 0 else None  # Minimal prefetch to reduce FD pressure
    
    # Use pin_memory directly without intelligent detection
    pin_memory_safe = pin_memory and torch.cuda.is_available()
    
    logger.info(
        f"Optimized dataloader config: workers={num_workers}, prefetch={prefetch_factor}, "
        f"pin_memory={pin_memory_safe}, persistent_workers={persistent_workers}"
    )

    # GPU-optimized training loader
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'collate_fn': custom_collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory_safe,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor,
        'drop_last': True
        # Removed 'multiprocessing_context': use default fork for better FD efficiency
    }
    
    # Only add pin_memory_device if we're actually using pinned memory
    if pin_memory_safe:
        train_loader_kwargs['pin_memory_device'] = 'cuda'
    
    train_loader = DataLoader(train_data, **train_loader_kwargs)

    # GPU-optimized validation loader
    val_loader_kwargs = {
        'batch_size': batch_size * 2,  # Larger batch for validation (no gradients)
        'shuffle': False,
        'collate_fn': custom_collate_fn,
        'num_workers': num_workers // 2 if num_workers > 0 else 0,  # Fixed: no workers when num_workers=0
        'pin_memory': pin_memory_safe,
        'persistent_workers': persistent_workers,  # Already False, consistent with train loader
        'prefetch_factor': prefetch_factor,
        'drop_last': False
        # Removed 'multiprocessing_context': use default fork for better FD efficiency
    }
    
    # Only add pin_memory_device if we're actually using pinned memory
    if pin_memory_safe:
        val_loader_kwargs['pin_memory_device'] = 'cuda'
    
    val_loader = DataLoader(val_data, **val_loader_kwargs)

    return train_loader, val_loader


def create_infinite_dataloader(dataloader: DataLoader):
    """
    Create an infinite iterator from a DataLoader to avoid epoch boundaries.
    
    This is essential for step-based training where we don't want epoch
    boundaries to affect training dynamics.
    
    Args:
        dataloader: PyTorch DataLoader to make infinite
        
    Yields:
        Batches from the dataloader infinitely
    """
    while True:
        for batch in dataloader:
            yield batch


def create_step_based_data_loaders(train_data: Dataset, val_data: Dataset,
                                  batch_size: int = 256, num_workers: int = 8,
                                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders optimized for step-based training.
    
    Key optimizations:
    - Training loader uses drop_last=True to ensure consistent batch sizes
    - Optimized settings for step-based training patterns
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Force consistent batch sizes for step-based training
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    # Training loader - optimized for step-based training
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True  # Critical: ensure all batches have the same size
    )

    # Validation loader - can handle variable batch sizes
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False  # Keep all validation data
    )

    return train_loader, val_loader


def split_dataset(dataset: Dataset, train_ratio: float = 0.8,
                  seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Set random seed
    torch.manual_seed(seed)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} validation")

    return train_dataset, val_dataset


class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups graphs by size for efficiency.
    """

    def __init__(self, dataset: Dataset, batch_size: int,
                 size_fn: Optional[Callable] = None):
        """
        Initialize dynamic batch sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Target batch size
            size_fn: Function to compute graph size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn or (lambda x: x.x.size(0))

        # Group indices by graph size
        self._group_by_size()

    def _group_by_size(self):
        """Group dataset indices by graph size."""
        size_to_indices = {}

        for idx in range(len(self.dataset)):
            graph = self.dataset[idx]
            size = self.size_fn(graph)

            if size not in size_to_indices:
                size_to_indices[size] = []
            size_to_indices[size].append(idx)

        self.size_groups = list(size_to_indices.values())

    def __iter__(self):
        """Iterate over batches."""
        # Shuffle groups
        random.shuffle(self.size_groups)

        # Create batches from each group
        for group in self.size_groups:
            # Shuffle indices within group
            indices = group.copy()
            random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:
                    yield batch

    def __len__(self):
        """Get number of batches."""
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size
            for group in self.size_groups
        )
