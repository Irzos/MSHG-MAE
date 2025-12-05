"""
Data preprocessing utilities for HyperGraph-MAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import os
import gc
import logging
from tqdm.auto import tqdm
import random
from typing import List, Dict, Tuple, Optional, Union
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocessing")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_masked_input(x: torch.Tensor, hyperedge_index: torch.Tensor, 
                         mask_ratio: float, beta: float = 0.2, 
                         noise_std: float = 0.1, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate masked input based on node degree optimization masking strategy.
    
    Args:
        x: Node features
        hyperedge_index: Hyperedge connectivity
        mask_ratio: Base masking ratio
        beta: Degree-based adjustment parameter
        noise_std: Standard deviation for noise
        device: Device to place tensors on
        
    Returns:
        Tuple of (masked_features, mask)
    """
    if device is None:
        device = x.device

    num_nodes = x.size(0)

    # Calculate node degrees for adjusting mask probability
    if hyperedge_index.numel() > 0:
        node_degree = torch.bincount(hyperedge_index[0], minlength=num_nodes).float().to(device)
        max_degree = node_degree.max() if node_degree.max() > 0 else 1.0
        normalized_degree = node_degree / max_degree
    else:
        # If no edges, use uniform masking
        normalized_degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)

    # Calculate mask probability - higher degree nodes have lower mask probability
    p_mask = torch.clamp(mask_ratio + beta * (1 - normalized_degree), max=0.9)

    # Generate mask
    mask = torch.rand(num_nodes, device=device) < p_mask

    # Apply mask and noise
    x_masked = x.clone()
    if noise_std > 0:
        # Only add noise to unmasked nodes (non-inplace)
        noise = torch.randn_like(x) * noise_std
        unmasked_noise = noise * (~mask).float().unsqueeze(-1)
        x_masked = x_masked + unmasked_noise

    # Set masked nodes to 0 (non-inplace)
    x_masked = torch.where(mask.unsqueeze(-1), torch.zeros_like(x_masked), x_masked)

    return x_masked, mask


def preprocess_batch(batch_data: List[Data]) -> List[Data]:
    """
    Preprocess batch data for training (without masking).
    
    Masking is now handled dynamically during training to support
    flexible masking strategies and efficient experimentation.
    
    Args:
        batch_data: List of Data objects
        
    Returns:
        List of preprocessed Data objects (without masks)
    """
    processed_batch = []

    for data in tqdm(batch_data, desc="Preprocessing batch"):
        # Skip invalid data
        if data is None or not hasattr(data, 'x') or data.x is None:
            continue

        # Ensure data is on CPU and float32 type
        data = data.cpu()
        data.x = data.x.float()

        # Hypergraph construction is single source of truth for dimensions
        # Skip placeholder creation - dimensions determined dynamically
        if not hasattr(data, 'hyperedge_index') or data.hyperedge_index is None:
            data.hyperedge_index = torch.zeros((2, 0), dtype=torch.long)

        # No hardcoded hyperedge_attr placeholders
        # Construction module will set proper dimensions

        # NOTE: Masking is now handled dynamically during training
        # This allows for flexible masking strategies without reprocessing data

        # Ensure all tensors are contiguous
        data.x = data.x.contiguous()
        data.hyperedge_index = data.hyperedge_index.contiguous()
        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            data.hyperedge_attr = data.hyperedge_attr.contiguous()

        processed_batch.append(data)

    return processed_batch


def batch_to_device(batch, device: torch.device, non_blocking: bool = True):
    """
    Move batch data to specified device, support non-blocking transfer.
    
    Args:
        batch: Batch data
        device: Target device
        non_blocking: Whether to use non-blocking transfer
        
    Returns:
        Batch data on target device
    """
    if isinstance(batch, (list, tuple)):
        return [batch_to_device(x, device, non_blocking) for x in batch]
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        return batch.to(device, non_blocking=non_blocking)
    else:
        return batch


def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())

    # CPU memory usage
    cpu_memory = process.memory_info().rss / (1024 ** 3)  # GB
    logger.info(f"CPU Memory: {cpu_memory:.2f}GB")

    # GPU memory usage
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"GPU Memory: Allocated={gpu_memory_allocated:.2f}GB, Reserved={gpu_memory_reserved:.2f}GB")


def cleanup_memory():
    """Actively clean up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_batches(batches: List[Data], output_dir: str, batch_size: int = 1000, 
                prefix: str = "batch"):
    """
    Save processed data to disk in batches to reduce memory usage.
    
    Args:
        batches: List of processed data
        output_dir: Output directory
        batch_size: Batch size for saving
        prefix: File prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_batches = len(batches)
    num_files = (total_batches + batch_size - 1) // batch_size

    for i in range(num_files):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_batches)

        batch_to_save = batches[start_idx:end_idx]
        output_file = output_path / f"{prefix}_{i}.pt"

        # Save to disk
        torch.save(batch_to_save, output_file)
        logger.info(f"Saved {len(batch_to_save)} graphs to {output_file}")

        # Clean up memory
        del batch_to_save
        cleanup_memory()


def load_and_process_data(input_files: Union[str, List[str]], output_dir: str, 
                         batch_size: int = 1000,
                         max_files: Optional[int] = None) -> int:
    """
    Load and process multiple input files, generate preprocessed data.
    
    Masking is now handled dynamically during training for flexibility.
    
    Args:
        input_files: Input files or directory
        output_dir: Output directory
        batch_size: Batch size for processing
        max_files: Maximum number of files to process
        
    Returns:
        Total number of processed graphs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect all input files
    if isinstance(input_files, str):
        # If input is directory, collect all .pt files
        if os.path.isdir(input_files):
            files = sorted(Path(input_files).glob("*.pt"))
        else:
            files = [Path(input_files)]
    elif isinstance(input_files, (list, tuple)):
        files = [Path(f) for f in input_files]
    else:
        raise ValueError(f"Unsupported input type: {type(input_files)}")

    # Limit number of files
    if max_files is not None:
        files = files[:max_files]

    total_processed = 0
    batch_counter = 0
    current_batch = []

    # Log initial memory usage
    logger.info("Starting data processing")
    log_memory_usage()

    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Load data
            data = torch.load(file_path, map_location="cpu")

            # Preprocess batch (masking now handled during training)
            processed_data = preprocess_batch(data)

            # Add to current batch
            current_batch.extend(processed_data)
            total_processed += len(processed_data)

            # If current batch is large enough, save and clean up
            if len(current_batch) >= batch_size:
                save_batches(
                    current_batch,
                    output_dir,
                    batch_size=batch_size,
                    prefix=f"batch_{batch_counter}"
                )
                batch_counter += 1
                current_batch = []

                # Log memory usage
                log_memory_usage()

            # Clean up original data memory
            del data, processed_data
            cleanup_memory()

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    # Save remaining batch
    if current_batch:
        save_batches(
            current_batch,
            output_dir,
            batch_size=batch_size,
            prefix=f"batch_{batch_counter}"
        )

    logger.info(f"Data processing completed. Processed {total_processed} graphs total, saved to {batch_counter + 1} batch files.")
    log_memory_usage()

    return total_processed


# Helper functions: create efficient hypergraph
def create_efficient_hypergraph(node_indices: Union[List, torch.Tensor], 
                               edge_indices: Union[List, torch.Tensor], 
                               num_nodes: Optional[int] = None, 
                               num_edges: Optional[int] = None) -> Tuple[torch.Tensor, int, int]:
    """
    Create memory-efficient hypergraph representation.
    
    Args:
        node_indices: Node indices
        edge_indices: Edge indices
        num_nodes: Number of nodes
        num_edges: Number of edges
        
    Returns:
        Tuple of (hyperedge_index, num_nodes, num_edges)
    """
    # Ensure tensors are on CPU to save GPU memory
    device = "cpu"

    # Convert to tensors
    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long, device=device)
    if not isinstance(edge_indices, torch.Tensor):
        edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=device)

    # Determine number of nodes and edges
    if num_nodes is None:
        num_nodes = node_indices.max().item() + 1 if node_indices.numel() > 0 else 0
    if num_edges is None:
        num_edges = edge_indices.max().item() + 1 if edge_indices.numel() > 0 else 0

    # Create hypergraph index
    hyperedge_index = torch.stack([node_indices, edge_indices])

    # Ensure tensor is contiguous
    hyperedge_index = hyperedge_index.contiguous()

    return hyperedge_index, num_nodes, num_edges


# Helper function to convert to PyG data format
def convert_to_pyg_data(x: torch.Tensor, hyperedge_index: torch.Tensor, 
                       hyperedge_attr: Optional[torch.Tensor] = None, 
                       y: Optional[torch.Tensor] = None) -> Data:
    """
    Convert data to PyG Data object.
    
    Args:
        x: Node features
        hyperedge_index: Hyperedge connectivity
        hyperedge_attr: Hyperedge attributes
        y: Labels
        
    Returns:
        PyG Data object
    """
    data = Data(x=x, hyperedge_index=hyperedge_index)

    if hyperedge_attr is not None:
        data.hyperedge_attr = hyperedge_attr

    if y is not None:
        data.y = y

    return data


class DataStatistics:
    """Class for computing and storing dataset statistics."""
    
    def __init__(self):
        self.atom_stats = {}
        self.bond_stats = {}
        self.global_stats = {}
    
    def compute_global_statistics_from_smiles(self, smiles_list: List[str]) -> Dict:
        """
        Compute global statistics from raw SMILES using the correct implementation.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary of global statistics with correct RDKit enums
        """
        from rdkit import Chem
        from .molecule_features import get_global_feature_statistics
        
        logger.info("Computing global statistics from SMILES...")
        
        # Parse molecules from SMILES
        molecules = []
        for smiles in tqdm(smiles_list, desc="Parsing SMILES"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)
            except Exception as e:
                logger.warning(f"Error parsing SMILES {smiles}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(molecules)} molecules")
        
        # Use the correct implementation from molecule_features
        self.global_stats = get_global_feature_statistics(molecules)
        
        logger.info("Global statistics computed successfully using correct RDKit enums")
        return self.global_stats
    
    def compute_global_statistics(self, data_files: List[Path]) -> Dict:
        """
        DEPRECATED: This method has enum consistency issues and will return empty stats.
        Use compute_global_statistics_from_smiles() instead for correct RDKit enum handling.
        """
        logger.error("DEPRECATED METHOD CALLED: compute_global_statistics()")
        logger.error("This method has enum consistency issues. Use compute_global_statistics_from_smiles() instead.")
        
        # Return empty stats to force users to use the correct method
        self.global_stats = {}
        return self.global_stats
    
    def save_statistics(self, save_path: str):
        """Save statistics to file."""
        torch.save(self.global_stats, save_path)
        logger.info(f"Statistics saved to {save_path}")
    
    def load_statistics(self, load_path: str):
        """Load statistics from file."""
        self.global_stats = torch.load(load_path, map_location="cpu")
        logger.info(f"Statistics loaded from {load_path}")


if __name__ == "__main__":
    # Example usage
    input_dir = "/path/to/raw/data"
    output_dir = "/path/to/processed/data"

    # Process data (masking now handled during training)
    total_processed = load_and_process_data(
        input_dir,
        output_dir,
        batch_size=500
    )

    print(f"Processed {total_processed} graphs total")
