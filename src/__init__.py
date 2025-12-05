"""
HyperGraph-MAE: A Masked Autoencoder for Molecular Hypergraph Representation Learning

This package implements a self-supervised learning framework for molecular representation
learning using hypergraph neural networks and masked autoencoders.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easier access
from .models.hypergraph_mae import EnhancedHyperGraphMAE, PretrainedHyperGraphMAE
from .data.data_loader import MolecularHypergraphDataset
from .training.trainer import HyperGraphMAETrainer

__all__ = [
    "EnhancedHyperGraphMAE",
    "PretrainedHyperGraphMAE",
    "MolecularHypergraphDataset",
    "HyperGraphMAETrainer",
]
