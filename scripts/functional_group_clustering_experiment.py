#!/usr/bin/env python
"""
Functional group clustering experiment (patched variant).

Assess representation quality of the multi-scale hypergraph MAE by clustering
functional groups in embedding space. Includes semantic/random/uniform masking
variants and stricter evaluation (PCA-50 clustering, multi-label analysis,
statistical tests, linear probe separability).
"""

import argparse
import logging
import warnings
import copy

# Reduce third-party log noise
import matplotlib
matplotlib.set_loglevel("WARNING")

# Silence verbose warnings from numba/umap
warnings.filterwarnings('ignore', category=UserWarning, module='numba')
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

# Set third-party log levels
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 
logging.getLogger('umap').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('plotly').setLevel(logging.WARNING)
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
from datetime import datetime
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Scientific computing and statistics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu, fisher_exact
import statsmodels.stats.multitest as smt
from statsmodels.stats.contingency_tables import Table2x2

# Clustering and dimensionality reduction
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_mutual_info_score
import umap
import hdbscan  # Use hdbscan library for clustering

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go
import plotly.express as px

# RDKit chemistry
from rdkit import Chem
try:
    from rdkit.Chem import rdMolStandardize
except ImportError:
    # Older RDKit versions
    rdMolStandardize = None

# Project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hypergraph_mae import EnhancedHyperGraphMAE
from src.data.data_loader import MolecularHypergraphDataset
from src.utils.visualization import setup_plotting_style

# Logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Dataset constants (reusing definitions from probe3_92.py)
CLASSIFICATION_DATASETS = {"BBBP", "BACE", "HIV", "SIDER", "Tox21"}
REGRESSION_DATASETS = {"ESOL", "Lipo", "FreeSolv"}
ALL_DATASETS = CLASSIFICATION_DATASETS | REGRESSION_DATASETS

# Dataset aliases (compat for common names)
DATASET_ALIASES = {
    # MoleculeNet original name → internal project name
    'Lipophilicity': 'Lipo',
}
CANONICAL_TO_ALIASES = {
    'Lipo': ['Lipophilicity'],
}

def load_model_config(model_path: str, explicit_config: Optional[str] = None) -> Dict:
    """Load config near the model path."""
    model_path = Path(model_path)

    # Candidates in priority order: explicit > inferred relatives
    candidate_paths = []
    if explicit_config:
        candidate_paths.append(Path(explicit_config))
    candidate_paths.extend([
        model_path.parent / 'config.json',
        model_path.parent.parent / 'config.json',
        model_path.parent.parent.parent / 'config.json',
    ])

    for config_path in candidate_paths:
        if config_path.exists():
            logger.info(f"Found model config: {config_path}")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                logger.warning(f"Failed to load config {config_path}: {e}")
                continue

    # Fallback default config
    logger.warning("Model config not found; using defaults")
    return {
        'hypergraph_types': ['bond', 'ring', 'functional_group', 'hydrogen_bond', 'conjugated_system'],
        'features': {'hyperedge_dim': 25}
    }


def prepare_model_config(config: Dict) -> Dict:
    """Fill required fields for inference-time model construction."""
    cfg = copy.deepcopy(config)
    features = cfg.setdefault('features', {})
    atom_cfg = features.setdefault('atom', {})

    if atom_cfg.get('dim') is None:
        feature_dim = cfg.get('model', {}).get('feature_dim')
        hyperedge_dim = features.get('hyperedge_dim')
        fallback_dim = feature_dim or hyperedge_dim
        if fallback_dim is not None:
            atom_cfg['dim'] = int(fallback_dim)

    return cfg



def configure_hyperedge_alignment(config: Dict):
    """Configure hyperedge-dim alignment."""
    try:
        from src.data.data_loader import set_collate_hyperedge_dim
        
        # Read hyperedge_dim from config
        hyperedge_dim = config.get('features', {}).get('hyperedge_dim', 25)
        set_collate_hyperedge_dim(int(hyperedge_dim))
        logger.info(f"Configured hyperedge_dim alignment: {hyperedge_dim}")
        
    except ImportError as e:
        logger.warning(f"Failed to configure hyperedge_dim alignment (import error): {e}")
    except Exception as e:
        logger.error(f"Failed to configure hyperedge_dim alignment: {e}")



class MolecularDataset(Dataset):
    """
    Molecular dataset wrapper compatible with existing HypergraphDataset
    and new experimental requirements.
    Supports both preprocessed hypergraph data and raw SMILES data.
    """
    
    def __init__(self, data_path: str, sample_size: Optional[int] = None, 
                 random_seed: int = 42, use_hypergraph_data: bool = True):
        """
        Initialize molecular dataset.
        
        Args:
            data_path: Data path
            sample_size: Optional sample size
            random_seed: Random seed
            use_hypergraph_data: Whether to use preprocessed hypergraph data
        """
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.use_hypergraph_data = use_hypergraph_data
        self.target_hypergraph_types = None  # To be set from model config
        
        # Molecule standardizer
        if not use_hypergraph_data and rdMolStandardize is not None:
            self.standardizer = rdMolStandardize.Standardizer()
        else:
            self.standardizer = None
        
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} molecules from {data_path}")
    
    def set_target_hypergraph_types(self, hypergraph_types: List[str]):
        """Set target hypergraph types (from model config)."""
        self.target_hypergraph_types = hypergraph_types
        logger.info(f"Using target hypergraph types: {hypergraph_types}")
        
    def _load_data(self) -> List[Dict]:
        """Load and preprocess molecular data."""
        if self.use_hypergraph_data:
            return self._load_hypergraph_data()
        else:
            return self._load_smiles_data()
    
    def _load_hypergraph_data(self) -> List[Dict]:
        """Load preprocessed hypergraph data."""
        try:
            from src.data.data_loader import MolecularHypergraphDataset
            
            # Use existing dataset loader
            hypergraph_dataset = MolecularHypergraphDataset(
                self.data_path, 
                max_graphs=self.sample_size
            )
            
            # Convert to the format expected by this experiment
            data = []
            for i in range(len(hypergraph_dataset)):
                graph_data = hypergraph_dataset[i]
                
                # Extract SMILES and molecule information
                smiles = getattr(graph_data, 'smiles', None)
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            data.append({
                                'smiles': smiles,
                                'mol': mol,
                                'graph_data': graph_data,  # Keep original hypergraph data
                                'original_item': graph_data
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse SMILES from hypergraph data: {e}")
                        continue
            
            # Stratified sampling
            if self.sample_size and len(data) > self.sample_size:
                data = self._stratified_sampling(data)
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to load hypergraph data: {e}")
            # Fallback to SMILES-based loading
            return self._load_smiles_data()
    
    def _load_smiles_data(self) -> List[Dict]:
        """Load raw SMILES data using IncrementalHypergraphLoader."""
        logger.info("Detected raw SMILES data; trying IncrementalHypergraphLoader...")
        
        try:
            # Try IncrementalHypergraphLoader first
            return self._load_with_incremental_hypergraph()
        except Exception as e:
            logger.error(f"Incremental hypergraph loading failed: {e}")
            logger.info("Falling back to simple SMILES loading...")
            return self._load_smiles_data_simple()
    
    def _load_with_incremental_hypergraph(self) -> List[Dict]:
        """Load data using IncrementalHypergraphLoader."""
        logger.info("Trying IncrementalHypergraphLoader for SMILES data...")
        
        try:
            # Import IncrementalHypergraphLoader
            from src.data.incremental_hypergraph_loader import IncrementalHypergraphLoader
            
            # Check whether a bond-only data directory exists for extension
            bondonly_dir = self._find_bondonly_data_dir()
            if not bondonly_dir:
                raise FileNotFoundError("Bond-only data directory not found; cannot build incremental hypergraph.")
            
            # Use target hypergraph types or sensible defaults
            hypergraph_types = self.target_hypergraph_types or ['bond', 'ring', 'functional_group', 'hydrogen_bond', 'conjugated_system']
            config = {
                'hypergraph_types': hypergraph_types,
                'timeout_seconds': 30,
                'filter': {
                    'problematic_metals': ['Co', 'Fe', 'Ni', 'Pt', 'Pd', 'Ru']
                }
            }
            logger.info(f"Using hypergraph types: {hypergraph_types}")
            
            # Create IncrementalHypergraphLoader
            cache_dir = Path(__file__).parent.parent / "hydra" / "version2" / "cache" / "clustering_cache"
            loader = IncrementalHypergraphLoader(
                bonds_data_dir=str(bondonly_dir),
                config=config,
                cache_dir=str(cache_dir)
            )
            
            # Load dataset
            logger.info("Starting incremental hypergraph construction...")
            dataset = loader.load_dataset()
            logger.info(f"Successfully built incremental hypergraph dataset with {len(dataset)} graphs")
            
            # Convert to required format
            data = []
            max_samples = min(len(dataset), self.sample_size) if self.sample_size else len(dataset)
            
            for i in range(max_samples):
                try:
                    graph_data = dataset[i]
                    smiles = getattr(graph_data, 'smiles', None)
                    
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            data.append({
                                'smiles': smiles,
                                'mol': mol,
                                'graph_data': graph_data,  # Full hypergraph data
                                'original_item': {'smiles': smiles}
                            })
                except Exception as e:
                    logger.warning(f"Failed to process graph data index {i}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(data)} molecules with incremental hypergraphs")
            return data
            
        except ImportError as e:
            logger.error(f"Failed to import IncrementalHypergraphLoader: {e}")
            raise NotImplementedError("IncrementalHypergraphLoader is not available")
        except Exception as e:
            logger.error(f"Incremental hypergraph loading failed: {e}")
            raise
    
    def _find_bondonly_data_dir(self) -> Optional[Path]:
        """Find a suitable bond-only data directory."""
        # If current path already looks like a bond-only dataset, use it directly
        if self.data_path.is_dir() and list(self.data_path.glob("batch_*.pt")):
            logger.info(f"Using existing bond-only data directory: {self.data_path}")
            return self.data_path
        
        # Try a few common bond-only data paths (relative to current dataset)
        base_paths = [
            self.data_path.parent / "zinc_druglikehypergraph",
            self.data_path.parent / "hypergraph_bondonly" / "zinc_druglikehypergraph",
            self.data_path.parent / "graph" / "zinc_druglikehypergraph",
        ]
        
        for path in base_paths:
            if path.exists() and list(path.glob("batch_*.pt")):
                logger.info(f"Found bond-only data directory: {path}")
                return path
        
        logger.warning("Bond-only data directory not found")
        return None
    
    
    def _load_smiles_data_simple(self) -> List[Dict]:
        """Simple SMILES-based data loading (fallback)."""
        data = []
        
        if self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                raw_data = pickle.load(f)
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
        else:
            # Assume a text file with one SMILES per line
            with open(self.data_path, 'r') as f:
                raw_data = [{'smiles': line.strip()} for line in f if line.strip()]
        
        # Preprocess data and apply stratified sampling
        processed_data = []
        for item in raw_data:
            if isinstance(item, str):
                smiles = item
            elif isinstance(item, dict):
                smiles = item.get('smiles', item.get('SMILES', ''))
            else:
                continue

            if smiles:
                # Molecule standardization
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        if hasattr(self, 'standardizer') and self.standardizer is not None:
                            standardized_mol = self.standardizer.standardize(mol)
                            standardized_smiles = Chem.MolToSmiles(standardized_mol)
                        else:
                            standardized_mol = mol
                            standardized_smiles = smiles

                        processed_data.append({
                            'smiles': standardized_smiles,
                            'mol': standardized_mol,
                            'graph_data': None,  # No preprocessed hypergraph data
                            'original_item': item
                        })
                except Exception as e:
                    logger.warning(f"Failed to process SMILES {smiles}: {e}")
                    continue
        
        # Stratified sampling
        if self.sample_size and len(processed_data) > self.sample_size:
            processed_data = self._stratified_sampling(processed_data)
            
        return processed_data
    
    
    def _stratified_sampling(self, data: List[Dict]) -> List[Dict]:
        """Stratified sampling based on simple molecular features."""
        logger.info(f"Performing stratified sampling: {len(data)} -> {self.sample_size}")
        
        # Compute stratification features: molecular weight and number of rings
        features = []
        for item in data:
            mol = item['mol']
            try:
                mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                features.append((mw, num_rings))
            except Exception as e:
                logger.warning(f"Failed to calculate molecular features: {e}")
                features.append((200.0, 1))  # Default fallback
        
        # Create strata based on molecular weight and ring count
        mw_values = [f[0] for f in features]
        ring_values = [f[1] for f in features]
        
        if len(set(mw_values)) > 1:
            mw_quartiles = np.percentile(mw_values, [25, 50, 75])
        else:
            mw_quartiles = [mw_values[0], mw_values[0], mw_values[0]]
            
        if len(set(ring_values)) > 1:
            ring_quartiles = np.percentile(ring_values, [25, 50, 75])
        else:
            ring_quartiles = [ring_values[0], ring_values[0], ring_values[0]]
        
        # Assign samples to strata
        strata = defaultdict(list)
        for i, (mw, rings) in enumerate(features):
            mw_bin = np.searchsorted(mw_quartiles, mw)
            ring_bin = np.searchsorted(ring_quartiles, rings)
            stratum = f"{mw_bin}_{ring_bin}"
            strata[stratum].append(i)
        
        # Sample proportionally from each stratum
        np.random.seed(self.random_seed)
        sampled_indices = []
        
        if len(strata) > 0:
            samples_per_stratum = max(1, self.sample_size // len(strata))
            
            for stratum_indices in strata.values():
                if len(stratum_indices) <= samples_per_stratum:
                    sampled_indices.extend(stratum_indices)
                else:
                    sampled = np.random.choice(stratum_indices, samples_per_stratum, replace=False)
                    sampled_indices.extend(sampled)
        
        # If more samples are needed, randomly fill from remaining indices
        if len(sampled_indices) < self.sample_size:
            remaining = self.sample_size - len(sampled_indices)
            all_indices = set(range(len(data)))
            unused_indices = list(all_indices - set(sampled_indices))
            if unused_indices:
                additional = np.random.choice(unused_indices, 
                                           min(remaining, len(unused_indices)), 
                                           replace=False)
                sampled_indices.extend(additional)
        
        return [data[i] for i in sampled_indices[:self.sample_size]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class FunctionalGroupAnnotator:
    """Functional-group annotator using standard RDKit SMARTS patterns."""
    
    def __init__(self):
        """Initialize functional-group SMARTS patterns."""
        self.functional_group_patterns = {
            # Basic functional groups
            'carboxyl': '[CX3](=O)[OX2H1]',
            'carboxylate': '[CX3](=O)[OX1-]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'amino_tertiary': '[NX3;H0;!$(NC=O)]',
            'nitro': '[NX3](=O)=O',
            'cyano': '[CX2]#N',
            'aldehyde': '[CX3H1](=O)[#6]',
            'ketone': '[#6][CX3](=O)[#6]',
            'ester': '[#6][CX3](=O)[OX2][#6]',
            'amide': '[NX3][CX3](=[OX1])[#6]',
            'ether': '[OD2]([#6])[#6]',
            'thiol': '[SX2H]',
            'disulfide': '[SX2][SX2]',
            'halogen': '[F,Cl,Br,I]',
            
            # Sulfur-containing groups
            'sulfone': '[SX4](=[OX1])(=[OX1])([#6])[#6]',
            'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])',
            
            # Phosphorus-containing groups  
            'phosphate': '[PX4](=[OX1])([OX2])([OX2])([OX2])',
            
            # Aromatic ring systems
            'phenyl': 'c1ccccc1',
            'pyridyl': 'c1ccncc1',
            'pyrimidinyl': 'c1cncnc1',
            'imidazolyl': 'c1c[nH]cn1',
            'furanyl': 'c1ccoc1',
            'thiopheneyl': 'c1ccsc1',
            'indolyl': 'c1ccc2[nH]ccc2c1',
            
            # Pharmacologically important groups
            'guanidine': '[NX3]([#1,#6])[CX3](=[NX3+,NX2+0])[NX3]([#1,#6])',
            'urea': '[NX3][CX3](=[OX1])[NX3]',
            'imino': '[CX3]=[NX2]',
            'alkyne': '[CX2]#[CX2]',
            'alkene': '[CX3]=[CX3]',
        }
        
        # Compile SMARTS patterns for efficiency
        self.compiled_patterns = {}
        for name, pattern in self.functional_group_patterns.items():
            try:
                self.compiled_patterns[name] = Chem.MolFromSmarts(pattern)
            except Exception as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")
    
    def annotate_molecule(self, mol: Chem.Mol) -> Dict[str, List[int]]:
        """
        Annotate functional groups in a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Mapping from functional-group name to list of atom indices
        """
        annotations = defaultdict(list)
        
        for fg_name, pattern in self.compiled_patterns.items():
            if pattern is None:
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                annotations[fg_name].extend(list(match))
        
        # Deduplicate atom indices
        for fg_name in annotations:
            annotations[fg_name] = list(set(annotations[fg_name]))
            
        return dict(annotations)
    
    def create_multilabel_matrix(self, molecules: List[Chem.Mol]) -> Tuple[np.ndarray, List[str]]:
        """
        Create a multilabel matrix for functional-group presence.
        
        Args:
            molecules: List of molecules
            
        Returns:
            (multilabel_matrix, functional_group_names)
            multilabel_matrix: [num_molecules, num_functional_groups]
            functional_group_names: list of functional-group names
        """
        all_annotations = []
        for mol in molecules:
            annotations = self.annotate_molecule(mol)
            all_annotations.append(annotations)
        
        # Collect all functional groups observed in the dataset
        all_fg_names = set()
        for annotations in all_annotations:
            all_fg_names.update(annotations.keys())
        
        fg_names = sorted(list(all_fg_names))
        
        # Create multilabel presence matrix
        matrix = np.zeros((len(molecules), len(fg_names)), dtype=int)
        for i, annotations in enumerate(all_annotations):
            for j, fg_name in enumerate(fg_names):
                if fg_name in annotations and len(annotations[fg_name]) > 0:
                    matrix[i, j] = 1
        
        return matrix, fg_names


class LinearProbe:
    """Linear probe to evaluate separability of functional groups."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def evaluate_functional_group(self, embeddings: np.ndarray, labels: np.ndarray,
                                fg_name: str) -> Dict[str, float]:
        """
        Evaluate separability of a single functional group with baseline comparison and balanced sampling.
        
        Args:
            embeddings: Molecular embeddings [num_molecules, embedding_dim]
            labels: Functional-group labels [num_molecules] (0/1)
            fg_name: Functional-group name
            
        Returns:
            Dictionary of evaluation metrics (including baseline comparison and confidence intervals)
        """
        if len(np.unique(labels)) < 2:
            logger.warning(f"Insufficient classes for {fg_name}, skipping")
            return {'auroc': 0.5, 'auprc': 0.0, 'n_positive': np.sum(labels)}
        
        try:
            # Baseline AUPRC for a random classifier equals positive ratio
            positive_ratio = np.sum(labels) / len(labels)
            baseline_auprc = positive_ratio
            
            # Dynamically adjust number of CV folds for small sample sizes
            n_samples = len(labels)
            n_splits = min(5, n_samples // 10) if n_samples < 100 else 5
            n_splits = max(2, n_splits)
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            
            # L2-regularized logistic regression with class_weight='balanced' for imbalance
            lr = LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000, 
                C=1.0,
                class_weight='balanced'  # Automatically balance class weights
            )
            
            # Cross-validation evaluation (avoid preprocessing leakage: standardize within folds)
            auroc_scores = []
            auprc_scores = []
            
            for train_idx, test_idx in cv.split(embeddings, labels):
                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
                
                # Ensure test split contains both classes
                if len(np.unique(y_test)) < 2:
                    continue
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                lr.fit(X_train_s, y_train)
                y_prob = lr.predict_proba(X_test_s)[:, 1]
                
                auroc = roc_auc_score(y_test, y_prob)
                auprc = average_precision_score(y_test, y_prob)
                
                auroc_scores.append(auroc)
                auprc_scores.append(auprc)
            
            if not auroc_scores:
                return {'auroc': 0.5, 'auprc': baseline_auprc, 'n_positive': np.sum(labels)}
            
            # Compute 95% bootstrap confidence intervals
            mean_auroc = np.mean(auroc_scores)
            mean_auprc = np.mean(auprc_scores)
            
            # Improvement relative to baseline
            auprc_lift = mean_auprc - baseline_auprc
            relative_lift = auprc_lift / baseline_auprc if baseline_auprc > 0 else 0
            
            return {
                'auroc': mean_auroc,
                'auroc_std': np.std(auroc_scores),
                'auroc_ci_lower': np.percentile(auroc_scores, 2.5),
                'auroc_ci_upper': np.percentile(auroc_scores, 97.5),
                'auprc': mean_auprc,
                'auprc_std': np.std(auprc_scores),
                'auprc_ci_lower': np.percentile(auprc_scores, 2.5),
                'auprc_ci_upper': np.percentile(auprc_scores, 97.5),
                'auprc_baseline': baseline_auprc,
                'auprc_lift': auprc_lift,
                'relative_lift': relative_lift,
                'n_positive': np.sum(labels),
                'n_total': len(labels),
                'positive_ratio': positive_ratio,
                'n_folds': n_splits
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {fg_name}: {e}")
            return {
                'auroc': 0.5, 
                'auprc': positive_ratio, 
                'auprc_baseline': positive_ratio,
                'n_positive': np.sum(labels)
            }


class ClusteringAnalysis:
    """Clustering analysis class."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def perform_clustering(self, embeddings: np.ndarray, 
                          method: str = 'hdbscan',
                          optimize_params: bool = True,
                          kmeans_k: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform clustering analysis.
        
        Args:
            embeddings: embedding vectors [num_samples, embedding_dim]
            method: clustering method
            optimize_params: whether to run parameter optimization
            
        Returns:
            (cluster_labels, metrics)
        """
        apply_postprocess = False
        if method == 'hdbscan':
            n_samples = len(embeddings)
            
            if optimize_params and n_samples >= 100:  # Run parameter optimization for large samples
                logger.info(f"Optimizing HDBSCAN parameters for {n_samples} samples...")
                
                # Use optimized parameters
                optimization_result = self.optimize_hdbscan_parameters(embeddings)
                
                best_params = optimization_result['best_params']
                cluster_labels = optimization_result['best_result']['cluster_labels']
                optimization_metrics = optimization_result['best_result']['metrics']
                
                logger.info(f"Optimized parameters: {best_params}")
                logger.info(f"Optimization found {optimization_result['n_trials']} valid configurations")
                
            else:
                # Use heuristic parameters (small samples or optimization disabled)
                if n_samples < 100:
                    # Small sample: use very loose parameters
                    min_cluster_size = max(3, min(5, n_samples // 12))
                    min_samples = max(1, min(3, min_cluster_size // 2))
                    epsilon = 0.0
                else:
                    # Large sample: use standard parameters
                    min_cluster_size = max(10, int(0.01 * n_samples))  # Slightly relaxed to 1%
                    min_samples = max(1, min_cluster_size // 5)  # Reduce min_samples to lower noise
                    epsilon = 0.05  # Slightly relax epsilon
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='cosine',
                    cluster_selection_method='leaf',  # Finer-grained clustering
                    cluster_selection_epsilon=epsilon
                )
                
                logger.debug(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, "
                            f"min_samples={min_samples}, epsilon={epsilon}, metric=cosine, method=leaf")
                cluster_labels = clusterer.fit_predict(embeddings)
            apply_postprocess = True
        elif method == 'kmeans':
            n_samples = len(embeddings)
            # Choose K: prefer provided value, otherwise select by CH index over candidate Ks
            if kmeans_k is None:
                candidate_ks = [6, 8, 10, 12, 15]
                candidate_ks = [k for k in candidate_ks if 1 < k < n_samples]
                best_k = None
                best_score = -np.inf
                best_labels = None
                for k in candidate_ks:
                    try:
                        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                        labels = km.fit_predict(embeddings)
                        # Use CH index as primary ranking metric
                        ch = calinski_harabasz_score(embeddings, labels)
                        if ch > best_score:
                            best_score = ch
                            best_k = k
                            best_labels = labels
                    except Exception as e:
                        logger.debug(f"KMeans k={k} failed: {e}")
                        continue
                if best_labels is None:
                    # Fallback: choose a conservative K
                    best_k = min(8, max(2, n_samples // 20))
                    km = KMeans(n_clusters=best_k, n_init=10, random_state=self.random_state)
                    best_labels = km.fit_predict(embeddings)
                cluster_labels = best_labels
            else:
                k = int(max(2, min(kmeans_k, n_samples - 1)))
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                cluster_labels = km.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Compute clustering quality metrics
        if method == 'hdbscan' and optimize_params and n_samples >= 100 and 'optimization_metrics' in locals():
            # If optimization was used, reuse metrics from the optimization run
            metrics = optimization_metrics.copy()
        else:
            # Regular metric computation
            metrics = {}
            
            # Compute metrics after excluding noise points
            valid_mask = cluster_labels >= 0 if method == 'hdbscan' else np.ones_like(cluster_labels, dtype=bool)
            if np.sum(valid_mask) > 1:
                valid_embeddings = embeddings[valid_mask]
                valid_labels = cluster_labels[valid_mask]
                
                if len(np.unique(valid_labels)) > 1:
                    try:
                        metrics['calinski_harabasz'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                        metrics['davies_bouldin'] = davies_bouldin_score(valid_embeddings, valid_labels)
                        # HDBSCAN uses cosine; KMeans branch still uses Euclidean
                        sil_metric = 'cosine' if method == 'hdbscan' else 'euclidean'
                        metrics['silhouette'] = silhouette_score(valid_embeddings, valid_labels, metric=sil_metric)
                    except Exception as e:
                        logger.warning(f"Failed to compute clustering metrics: {e}")
                        # Set default values instead of skipping
                        metrics['calinski_harabasz'] = 0.0
                        metrics['davies_bouldin'] = 0.0
                        metrics['silhouette'] = 0.0
                        
            # DBCV requires original cluster labels (including noise)
            try:
                if method == 'hdbscan' and 'clusterer' in locals() and hasattr(clusterer, 'dbcv_'):
                    metrics['dbcv'] = clusterer.dbcv_
                else:
                    # Manually compute a simplified DBCV
                    metrics['dbcv'] = self._compute_dbcv_approximation(embeddings, cluster_labels) if method == 'hdbscan' else 0.0
            except Exception as e:
                logger.warning(f"Failed to compute DBCV: {e}")
                
            # Basic clustering statistics
            if method == 'hdbscan':
                metrics['n_clusters'] = len(np.unique(cluster_labels[cluster_labels >= 0]))
                metrics['n_noise'] = np.sum(cluster_labels == -1)
                metrics['noise_ratio'] = np.sum(cluster_labels == -1) / len(cluster_labels)
            else:
                metrics['n_clusters'] = len(np.unique(cluster_labels))
                metrics['n_noise'] = 0
                metrics['noise_ratio'] = 0.0
        
        # Post-processing: merge similar micro-clusters
        if apply_postprocess and len(np.unique(cluster_labels[cluster_labels >= 0])) > 1:
            original_clusters = metrics['n_clusters']
            cluster_labels = self.post_process_clusters(embeddings, cluster_labels)
            
            # Recompute statistics after post-processing
            metrics['n_clusters'] = len(np.unique(cluster_labels[cluster_labels >= 0]))
            metrics['n_noise'] = np.sum(cluster_labels == -1)
            metrics['noise_ratio'] = np.sum(cluster_labels == -1) / len(cluster_labels)
            
            if metrics['n_clusters'] != original_clusters:
                logger.info(f"Post-processing merged clusters: {original_clusters} -> {metrics['n_clusters']}")
        
        return cluster_labels, metrics
    
    def _compute_dbcv_approximation(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Approximate DBCV value."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
                
            # Simplified DBCV computation
            nn = NearestNeighbors(n_neighbors=5).fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)
            
            cluster_scores = []
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                cluster_mask = labels == label
                if np.sum(cluster_mask) < 2:
                    continue
                    
                cluster_distances = distances[cluster_mask]
                cluster_score = np.mean(cluster_distances)
                cluster_scores.append(cluster_score)
            
            return np.mean(cluster_scores) if cluster_scores else 0.0
            
        except Exception as e:
            logger.warning(f"DBCV approximation failed: {e}")
            return 0.0
    
    def optimize_hdbscan_parameters(self, embeddings: np.ndarray, 
                                  target_noise_ratio: float = 0.35,
                                  max_noise_ratio: float = 0.6) -> Dict[str, Any]:
        """
        Grid search optimization for HDBSCAN parameters (lenient version that absorbs edge samples).
        
        Args:
            embeddings: L2-normalized embedding vectors [num_samples, embedding_dim]
            target_noise_ratio: target noise ratio (35%)
            max_noise_ratio: maximum allowed noise ratio (up to 60% to allow optimization to work)
            
        Returns:
            Best parameter configuration and performance metrics
        """
        logger.info("Starting aggressive HDBSCAN parameter optimization...")
        
        n_samples = len(embeddings)
        
        # More lenient parameter grid (further relaxed to reduce noise rate)
        param_grid = {
            'min_cluster_size': [
                max(5, int(0.003 * n_samples)),   # 0.3%, at least 5
                max(8, int(0.006 * n_samples)),   # 0.6%, at least 8
                max(10, int(0.01 * n_samples)),   # 1.0%, at least 10
                max(15, int(0.015 * n_samples)),  # 1.5%, at least 15
            ],
            'min_samples': [],  # Will be dynamically computed from min_cluster_size
            'cluster_selection_epsilon': [0.0, 0.05, 0.1, 0.15, 0.2]  # Stronger cluster-merging effect
        }
        
        # Dynamically compute min_samples values
        for mcs in param_grid['min_cluster_size']:
            param_grid['min_samples'].extend([
                1,
                max(1, mcs // 10),
                max(1, mcs // 5),
            ])
        param_grid['min_samples'] = sorted(list(set(param_grid['min_samples'])))
        
        logger.info(f"Parameter grid: min_cluster_size={param_grid['min_cluster_size']}, "
                   f"min_samples={param_grid['min_samples']}, "
                   f"epsilon={param_grid['cluster_selection_epsilon']}")
        
        best_params = None
        best_score = -np.inf
        best_result = None
        results = []
        
        # Grid search
        for mcs in param_grid['min_cluster_size']:
            for ms in param_grid['min_samples']:
                for epsilon in param_grid['cluster_selection_epsilon']:
                    # Skip unreasonable parameter combinations
                    if ms > mcs:
                        continue
                        
                    try:
                        # Create HDBSCAN clusterer (using cosine distance)
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric='cosine',
                            cluster_selection_method='leaf',
                            cluster_selection_epsilon=epsilon
                        )
                        
                        # Run clustering
                        cluster_labels = clusterer.fit_predict(embeddings)
                        
                        # Compute metrics
                        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
                        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
                        
                        # Skip results with too much noise
                        if noise_ratio > max_noise_ratio:
                            continue
                            
                        # Skip results with too few clusters
                        if n_clusters < 2:
                            continue
                            
                        # Compute clustering quality metrics
                        metrics = {}
                        valid_mask = cluster_labels >= 0
                        if np.sum(valid_mask) > 1:
                            valid_embeddings = embeddings[valid_mask]
                            valid_labels = cluster_labels[valid_mask]
                            
                            if len(np.unique(valid_labels)) > 1:
                                try:
                                    metrics['calinski_harabasz'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                                    metrics['davies_bouldin'] = davies_bouldin_score(valid_embeddings, valid_labels)
                                    metrics['silhouette'] = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
                                except Exception as e:
                                    logger.warning(f"Failed to compute sklearn metrics for params mcs={mcs}, ms={ms}, eps={epsilon}: {e}")
                                    # Do not drop this parameter combination; only mark these metrics as failed
                                    metrics['calinski_harabasz'] = 0.0
                                    metrics['davies_bouldin'] = 0.0
                                    metrics['silhouette'] = 0.0
                            
                        # Detect micro-clusters (clusters with <3% of samples)
                        cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0])
                        micro_clusters = np.sum(cluster_sizes < (n_samples * 0.03))
                        micro_cluster_penalty = micro_clusters * 0.1  # Penalty for micro-clusters
                        
                        # Compute composite score (reweight noise and micro-cluster penalties)
                        dbcv_score = self._compute_dbcv_approximation(embeddings, cluster_labels)
                        noise_penalty = abs(noise_ratio - target_noise_ratio)
                        
                        # Coverage bonus: encourage more samples to be assigned to valid clusters
                        coverage_ratio = (n_samples - np.sum(cluster_labels == -1)) / n_samples
                        coverage_bonus = 0.5 * max(0, coverage_ratio - 0.5)  # Reward when coverage >50%
                        
                        # Composite score: emphasize coverage and clustering quality, de-emphasize strict noise rate
                        composite_score = (
                            2.0 * dbcv_score +                          # Increase DBCV weight
                            0.8 * metrics.get('silhouette', 0.0) +      # Increase silhouette weight  
                            - 0.5 * noise_penalty +                     # Further reduce noise penalty
                            - 0.2 * metrics.get('davies_bouldin', 2.0) + # Reduce Davies-Bouldin penalty
                            + coverage_bonus +                           # Coverage bonus
                            - micro_cluster_penalty                      # Micro-cluster penalty
                        )
                        
                        result = {
                            'params': {'min_cluster_size': mcs, 'min_samples': ms, 'cluster_selection_epsilon': epsilon},
                            'metrics': {
                                'n_clusters': n_clusters,
                                'noise_ratio': noise_ratio,
                                'dbcv': dbcv_score,
                                'composite_score': composite_score,
                                **metrics
                            },
                            'cluster_labels': cluster_labels
                        }
                        
                        results.append(result)
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_params = result['params']
                            best_result = result
                            
                        logger.debug(f"Params: mcs={mcs}, ms={ms}, eps={epsilon} | "
                                   f"Clusters: {n_clusters}, Noise: {noise_ratio:.1%}, "
                                   f"Score: {composite_score:.3f}")
                                   
                    except Exception as e:
                        logger.warning(f"Failed to test params mcs={mcs}, ms={ms}, eps={epsilon}: {e}")
                        continue
        
        if best_result is None:
            logger.warning("No valid parameter combination found! Using fallback parameters.")
            # Use relaxed fallback parameters
            best_params = {
                'min_cluster_size': max(3, int(0.01 * n_samples)),
                'min_samples': 1,
                'cluster_selection_epsilon': 0.1
            }
            # Rerun once to obtain results (using cosine distance)
            clusterer = hdbscan.HDBSCAN(**best_params, metric='cosine', cluster_selection_method='leaf')
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Compute full set of clustering metrics
            n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
            noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
            dbcv_score = self._compute_dbcv_approximation(embeddings, cluster_labels)
            
            # Compute CH and DB indices
            fallback_metrics = {}
            valid_mask = cluster_labels >= 0
            if np.sum(valid_mask) > 1:
                valid_embeddings = embeddings[valid_mask]
                valid_labels = cluster_labels[valid_mask]
                
                if len(np.unique(valid_labels)) > 1:
                    try:
                        fallback_metrics['calinski_harabasz'] = calinski_harabasz_score(valid_embeddings, valid_labels)
                        fallback_metrics['davies_bouldin'] = davies_bouldin_score(valid_embeddings, valid_labels)
                        fallback_metrics['silhouette'] = silhouette_score(valid_embeddings, valid_labels, metric='euclidean')
                    except Exception as e:
                        logger.warning(f"Failed to compute fallback metrics: {e}")
                        fallback_metrics['calinski_harabasz'] = 0.0
                        fallback_metrics['davies_bouldin'] = 0.0
                        fallback_metrics['silhouette'] = 0.0
                else:
                    fallback_metrics['calinski_harabasz'] = 0.0
                    fallback_metrics['davies_bouldin'] = 0.0
                    fallback_metrics['silhouette'] = 0.0
            else:
                fallback_metrics['calinski_harabasz'] = 0.0
                fallback_metrics['davies_bouldin'] = 0.0
                fallback_metrics['silhouette'] = 0.0
            
            best_result = {
                'params': best_params,
                'metrics': {
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'dbcv': dbcv_score,
                    'composite_score': 0.0,
                    **fallback_metrics
                },
                'cluster_labels': cluster_labels
            }
        
        logger.info(f"Optimal parameters found: {best_params}")
        logger.info(f"Best performance: {best_result['metrics']['n_clusters']} clusters, "
                   f"{best_result['metrics']['noise_ratio']:.1%} noise, "
                   f"DBCV={best_result['metrics']['dbcv']:.3f}")
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': results,
            'n_trials': len(results)
        }
    
    def post_process_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                             min_cluster_ratio: float = 0.03,
                             merge_threshold: float = 0.95) -> np.ndarray:
        """
        Post-process clustering results: merge micro-clusters and similar clusters to form ~8–12 business clusters.
        
        Args:
            embeddings: embedding vectors [num_samples, embedding_dim]
            cluster_labels: original cluster labels
            min_cluster_ratio: minimum cluster-size ratio threshold (3%; clusters below this are treated as micro-clusters to merge)
            merge_threshold: cosine-similarity threshold for merging clusters
            
        Returns:
            Post-processed cluster labels
        """
        n_samples = len(embeddings)
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        
        if len(unique_clusters) <= 1:
            return cluster_labels
            
        # Compute cluster centers
        cluster_centers = {}
        cluster_sizes = {}
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_centers[cluster_id] = np.mean(embeddings[mask], axis=0)
            cluster_sizes[cluster_id] = np.sum(mask)
            
        # Identify micro-clusters (clusters below ratio threshold)
        micro_clusters = []
        normal_clusters = []
        
        for cluster_id in unique_clusters:
            ratio = cluster_sizes[cluster_id] / n_samples
            if ratio < min_cluster_ratio:
                micro_clusters.append(cluster_id)
            else:
                normal_clusters.append(cluster_id)
                
        logger.info(f"Found {len(micro_clusters)} micro-clusters and {len(normal_clusters)} normal clusters")

        # Step 1: absorb micro-clusters into nearest large clusters (by center similarity/distance)
        if len(micro_clusters) > 0 and len(normal_clusters) > 0:
            # Prepare matrix of large-cluster centers
            normal_centers = np.array([cluster_centers[cid] for cid in normal_clusters])
            # Absorb micro-clusters one by one
            for cid in micro_clusters:
                c_center = cluster_centers[cid][None, :]  # [1, d]
                # Use cosine distance (1 - cosine similarity)
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(c_center, normal_centers)[0]
                dists = 1.0 - sims
                target_idx = int(np.argmin(dists))
                target_cid = normal_clusters[target_idx]
                cluster_labels[cluster_labels == cid] = target_cid
                logger.info(f"Absorb micro-cluster {cid} into {target_cid} (cosine distance={dists[target_idx]:.4f})")

            # Recompute centers
            unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
            cluster_centers = {cid: np.mean(embeddings[cluster_labels == cid], axis=0) for cid in unique_clusters}

        # Compute cosine similarity matrix between clusters
        if len(unique_clusters) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            centers_matrix = np.array([cluster_centers[cid] for cid in unique_clusters])
            similarity_matrix = cosine_similarity(centers_matrix)
            
            # Merge highly similar clusters
            merged_mapping = {}  # old_id -> new_id
            next_new_id = 0
            
            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id in merged_mapping:
                    continue
                    
                merged_mapping[cluster_id] = next_new_id
                
                # Find other clusters with high similarity to current cluster
                for j, other_id in enumerate(unique_clusters):
                    if (i != j and other_id not in merged_mapping and 
                        similarity_matrix[i, j] > merge_threshold):
                        merged_mapping[other_id] = next_new_id
                        logger.info(f"Merging cluster {other_id} into {cluster_id} (similarity: {similarity_matrix[i,j]:.3f})")
                
                next_new_id += 1
            
            # Apply merge mapping
            new_labels = cluster_labels.copy()
            for old_id, new_id in merged_mapping.items():
                new_labels[cluster_labels == old_id] = new_id
                
            # Renumber cluster IDs (ensure contiguous)
            unique_new = np.unique(new_labels[new_labels >= 0])
            final_labels = new_labels.copy()
            for i, cluster_id in enumerate(unique_new):
                final_labels[new_labels == cluster_id] = i
                
            logger.info(f"Cluster post-processing: {len(unique_clusters)} -> {len(unique_new)} clusters")
            return final_labels
        
        return cluster_labels


class StatisticalAnalysis:
    """Statistical analysis class."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def fisher_exact_enrichment(self, cluster_assignments: np.ndarray, 
                              fg_matrix: np.ndarray, 
                              fg_names: List[str]) -> Dict[str, Any]:
        """
        Analyze functional-group enrichment within clusters using Fisher's exact test.
        
        Args:
            cluster_assignments: cluster labels [num_molecules]
            fg_matrix: multi-label functional-group matrix [num_molecules, num_fg]
            fg_names: list of functional-group names
            
        Returns:
            Enrichment analysis results
        """
        results = {'functional_groups': {}, 'summary': {}}
        
        unique_clusters = np.unique(cluster_assignments[cluster_assignments >= 0])
        if len(unique_clusters) == 0:
            logger.warning("No valid clusters found for enrichment analysis")
            return results
        
        all_p_values = []
        pairs = []  # Store (fg_name, cluster_id) pairs for mapping FDR results back
        
        for i, fg_name in enumerate(fg_names):
            fg_labels = fg_matrix[:, i]
            fg_results = {'clusters': {}, 'overall_p_value': 1.0}
            
            # Compute global functional-group ratio (for relative enrichment fold)
            global_fg_ratio = np.sum(fg_labels) / len(fg_labels)
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_assignments == cluster_id
                
                # Build 2x2 contingency table
                # [fg_in_cluster, fg_not_in_cluster]
                # [no_fg_in_cluster, no_fg_not_in_cluster]
                fg_in_cluster = np.sum(fg_labels[cluster_mask])
                fg_not_in_cluster = np.sum(fg_labels[~cluster_mask])
                no_fg_in_cluster = np.sum(cluster_mask) - fg_in_cluster
                no_fg_not_in_cluster = np.sum(~cluster_mask) - fg_not_in_cluster
                
                contingency_table = np.array([
                    [fg_in_cluster, fg_not_in_cluster],
                    [no_fg_in_cluster, no_fg_not_in_cluster]
                ])
                
                try:
                    # One-sided Fisher test (enrichment)
                    odds_ratio_raw, p_value = fisher_exact(contingency_table, alternative='greater')
                    # Apply Haldane–Anscombe 0.5 correction and compute OR and confidence interval
                    corrected = contingency_table.astype(float) + 0.5
                    try:
                        table2x2 = Table2x2(corrected)
                        or_corr = float(table2x2.oddsratio)
                        ci_low, ci_up = table2x2.oddsratio_confint(alpha=self.alpha)
                        ci_low = float(ci_low)
                        ci_up = float(ci_up)
                    except Exception:
                        or_corr, ci_low, ci_up = float(odds_ratio_raw), float('nan'), float('nan')
                    
                    # Compute within-cluster functional-group ratio and relative enrichment fold
                    cluster_size = np.sum(cluster_mask)
                    cluster_fg_ratio = fg_in_cluster / cluster_size if cluster_size > 0 else 0.0
                    relative_enrichment = cluster_fg_ratio / global_fg_ratio if global_fg_ratio > 0 else 1.0
                    
                    fg_results['clusters'][int(cluster_id)] = {
                        'odds_ratio': float(or_corr),
                        'p_value': float(p_value),
                        'fg_in_cluster': int(fg_in_cluster),
                        'cluster_size': int(cluster_size),
                        'cluster_fg_ratio': float(cluster_fg_ratio),  # Within-cluster fraction
                        'global_fg_ratio': float(global_fg_ratio),    # Global fraction
                        'enrichment_fold': float(relative_enrichment),  # Relative enrichment fold
                        'or_ci_lower': ci_low,
                        'or_ci_upper': ci_up
                    }
                    
                    all_p_values.append(p_value)
                    pairs.append((fg_name, int(cluster_id)))  # Record mapping
                    
                except Exception as e:
                    logger.warning(f"Fisher exact test failed for {fg_name} in cluster {cluster_id}: {e}")
            
            results['functional_groups'][fg_name] = fg_results
        
        # Benjamini–Hochberg FDR correction
        if all_p_values:
            rejected, p_corrected, alpha_sidak, alpha_bonf = smt.multipletests(
                all_p_values, alpha=self.alpha, method='fdr_bh'
            )
            
            # Fill FDR-corrected results back into per-cluster entries
            for (fg_name, cluster_id), p_corr, is_rejected in zip(pairs, p_corrected, rejected):
                results['functional_groups'][fg_name]['clusters'][cluster_id]['p_fdr'] = float(p_corr)
                results['functional_groups'][fg_name]['clusters'][cluster_id]['significant'] = bool(is_rejected)
            
            # Compute each functional group's overall_p_value (minimum FDR-adjusted p-value)
            for fg_name in fg_names:
                if fg_name in results['functional_groups'] and results['functional_groups'][fg_name]['clusters']:
                    p_fdr_values = [c['p_fdr'] for c in results['functional_groups'][fg_name]['clusters'].values() 
                                   if 'p_fdr' in c]
                    if p_fdr_values:
                        results['functional_groups'][fg_name]['overall_p_value'] = float(min(p_fdr_values))
            
            # Identify top-3 enriched functional groups for each cluster
            cluster_top_fgs = {}
            for cluster_id in unique_clusters:
                cluster_enrichments = []
                for fg_name in fg_names:
                    if (fg_name in results['functional_groups'] and 
                        int(cluster_id) in results['functional_groups'][fg_name]['clusters']):
                        cluster_data = results['functional_groups'][fg_name]['clusters'][int(cluster_id)]
                        if 'p_fdr' in cluster_data and cluster_data.get('significant', False):
                            cluster_enrichments.append({
                                'fg_name': fg_name,
                                'enrichment_fold': cluster_data['enrichment_fold'],
                                'p_fdr': cluster_data['p_fdr'],
                                'cluster_fg_ratio': cluster_data['cluster_fg_ratio']
                            })
                
                # Sort by enrichment fold and take top-3
                cluster_enrichments.sort(key=lambda x: x['enrichment_fold'], reverse=True)
                cluster_top_fgs[int(cluster_id)] = cluster_enrichments[:3]
            
            results['summary'] = {
                'total_tests': len(all_p_values),
                'significant_tests': np.sum(rejected),
                'fdr_alpha': self.alpha,
                'min_p_value': float(np.min(all_p_values)),
                'max_p_corrected': float(np.max(p_corrected)),
                'n_clusters': len(unique_clusters),
                'cluster_top_functional_groups': cluster_top_fgs
            }
        
        return results
    
    def distance_analysis(self, embeddings: np.ndarray, 
                         fg_matrix: np.ndarray,
                         fg_names: List[str],
                         max_pairs: int = 5000,
                         use_gpu: bool = True,
                         distance_metric: str = 'cosine',
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze distance distributions for same vs. different functional groups using sampling and optional GPU acceleration.
        
        Args:
            embeddings: molecular representation vectors
            fg_matrix: multi-label functional-group matrix
            fg_names: functional-group names
            max_pairs: maximum sampled pairs per class to avoid O(n²) memory
            use_gpu: whether to use GPU to accelerate distance computation
            distance_metric: distance metric ('cosine', 'euclidean_zscore', 'euclidean', 'mahalanobis', 'angular', 'correlation')
                - 'cosine': cosine distance in [0,2], comparable across experiments
                - 'euclidean_zscore': Z-score-normalized Euclidean distance 
                - 'euclidean': raw Euclidean distance (backward compatible)
                - 'mahalanobis': global Mahalanobis distance (PCA + whitening, then Euclidean)
                - 'angular': angular distance arccos(cosine) (focuses on directional differences)
                - 'correlation': 1 - Pearson correlation (scale-robust; equivalent to cosine after zero-centering each vector)
            
        Returns:
            Distance analysis results
        """
        results = {}
        
        # GPU optimizations: move to PyTorch tensor/device; whiten first for Mahalanobis
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        effective_metric = distance_metric
        emb_np = embeddings
        if distance_metric == 'mahalanobis':
            try:
                n, d = emb_np.shape
                n_comp = int(min(d, max(1, n - 1)))
                pca_w = PCA(n_components=n_comp, whiten=True, random_state=(int(seed) if seed is not None else 42))
                emb_np = pca_w.fit_transform(emb_np)
                effective_metric = 'euclidean'
                logger.debug(f"Applied PCA whitening for Mahalanobis: n_comp={n_comp}")
            except Exception as e:
                logger.warning(f"Mahalanobis whitening failed, fallback to euclidean: {e}")
                effective_metric = 'euclidean'
        embeddings_tensor = torch.from_numpy(emb_np.astype(np.float32)).to(device)
        logger.debug(f"Using {device} for distance calculations with metric: {distance_metric} (effective: {effective_metric})")
        
        def compute_distances(x1: torch.Tensor, x2: torch.Tensor, metric: str) -> torch.Tensor:
            """Compute distances under different metrics."""
            if metric == 'cosine':
                # Cosine distance = 1 - cosine similarity, range [0,2]
                # For L2-normalized vectors: cosine_distance = euclidean_distance^2 / 2
                cos_sim = torch.nn.functional.cosine_similarity(x1, x2, dim=1)
                return 1.0 - cos_sim
            elif metric in ('angular', 'angle'):
                # Angular distance = arccos(cosine_similarity)
                xi_n = torch.nn.functional.normalize(x1, p=2, dim=1)
                xj_n = torch.nn.functional.normalize(x2, p=2, dim=1)
                cos_sim = torch.sum(xi_n * xj_n, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                return torch.acos(cos_sim)
            elif metric in ('correlation', 'corr'):
                # 1 - Pearson correlation: zero-center each vector, then compute cosine
                x1c = x1 - x1.mean(dim=1, keepdim=True)
                x2c = x2 - x2.mean(dim=1, keepdim=True)
                num = (x1c * x2c).sum(dim=1)
                den = (torch.norm(x1c, dim=1) * torch.norm(x2c, dim=1) + 1e-12)
                corr = num / den
                return 1.0 - corr
                
            elif metric == 'euclidean_zscore':
                # Z-score-normalized Euclidean distance
                euclidean_dist = torch.norm(x1 - x2, dim=1)
                # Apply Z-score normalization to the whole distance distribution (on CPU)
                euclidean_np = euclidean_dist.cpu().numpy()
                if len(euclidean_np) > 1:
                    mean_dist = np.mean(euclidean_np)
                    std_dist = np.std(euclidean_np)
                    if std_dist > 1e-8:  # Avoid division by zero
                        zscore_dist = (euclidean_np - mean_dist) / std_dist
                        return torch.from_numpy(zscore_dist).to(device)
                return euclidean_dist
                
            elif metric == 'euclidean':
                # Raw Euclidean distance (backward compatible)
                return torch.norm(x1 - x2, dim=1)
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
        
        def compute_pairwise_distances(embeddings_batch: torch.Tensor, metric: str) -> torch.Tensor:
            """Compute pairwise distances within a batch."""
            if metric == 'cosine':
                # Use cosine similarity matrix (after normalization)
                emb_norm = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                cos_sim_matrix = torch.mm(emb_norm, emb_norm.t())
                return 1.0 - cos_sim_matrix
            elif metric in ('angular', 'angle'):
                emb_norm = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                cos_sim_matrix = torch.mm(emb_norm, emb_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                return torch.acos(cos_sim_matrix)
            elif metric in ('correlation', 'corr'):
                # Zero-center each row, then compute cosine
                Xc = embeddings_batch - embeddings_batch.mean(dim=1, keepdim=True)
                Xc = Xc / (torch.norm(Xc, dim=1, keepdim=True) + 1e-12)
                corr_mat = torch.mm(Xc, Xc.t())
                return 1.0 - corr_mat
            elif metric == 'euclidean_zscore':
                # First compute Euclidean distances, then apply Z-score normalization
                euclidean_matrix = torch.cdist(embeddings_batch, embeddings_batch, p=2)
                # Apply Z-score normalization to the whole matrix
                euclidean_flat = euclidean_matrix.flatten()
                euclidean_np = euclidean_flat.cpu().numpy()
                if len(euclidean_np) > 1:
                    mean_dist = np.mean(euclidean_np)
                    std_dist = np.std(euclidean_np)
                    if std_dist > 1e-8:
                        zscore_flat = (euclidean_np - mean_dist) / std_dist
                        zscore_tensor = torch.from_numpy(zscore_flat).to(device)
                        return zscore_tensor.reshape(euclidean_matrix.shape)
                return euclidean_matrix
            elif metric == 'euclidean':
                return torch.cdist(embeddings_batch, embeddings_batch, p=2)
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
        
        p_values = []
        fg_keys = []
        for i, fg_name in enumerate(fg_names):
            fg_labels = fg_matrix[:, i]
            
            if np.sum(fg_labels) < 2 or np.sum(1 - fg_labels) < 2:
                continue
            
            # Get indices for same and different molecules
            same_indices = np.where(fg_labels == 1)[0]
            diff_indices = np.where(fg_labels == 0)[0]
            
            # GPU optimization: sample same-class molecular pair distances
            same_distances = []
            if len(same_indices) >= 2:
                same_indices_tensor = torch.from_numpy(same_indices).to(device)
                
                # Limit number of same-class pairs to avoid OOM
                max_same_pairs = min(max_pairs, len(same_indices) * (len(same_indices) - 1) // 2)
                if max_same_pairs < len(same_indices) * (len(same_indices) - 1) // 2:
                    # Need sampling – compute in GPU batches
                    base_seed = seed if seed is not None else 42
                    torch.manual_seed(int(base_seed) + int(i) * 2)  # For reproducible sampling
                    n_samples = max_same_pairs
                    # Generate random index pairs
                    idx1 = torch.randint(0, len(same_indices), (n_samples,), device=device)
                    idx2 = torch.randint(0, len(same_indices), (n_samples,), device=device)
                    # Ensure indices in a pair are different
                    mask = idx1 != idx2
                    idx1, idx2 = idx1[mask][:n_samples], idx2[mask][:n_samples]
                    
                    # Compute distances in batches
                    embeddings1 = embeddings_tensor[same_indices_tensor[idx1]]
                    embeddings2 = embeddings_tensor[same_indices_tensor[idx2]]
                    distances = compute_distances(embeddings1, embeddings2, effective_metric)
                    same_distances = distances.cpu().numpy().tolist()
                else:
                    # GPU batched computation for all same-class pairs
                    same_embeddings = embeddings_tensor[same_indices_tensor]  # [n_same, embedding_dim]
                    # Use unified distance computation function
                    distances = compute_pairwise_distances(same_embeddings, effective_metric)  # [n_same, n_same]
                    # Take upper-triangular matrix (skip duplicates and diagonal)
                    triu_indices = torch.triu_indices(len(same_indices), len(same_indices), offset=1)
                    same_distances = distances[triu_indices[0], triu_indices[1]].cpu().numpy().tolist()
            
            # GPU optimization: sample different-class molecular pair distances
            diff_distances = []
            if len(same_indices) > 0 and len(diff_indices) > 0:
                same_indices_tensor = torch.from_numpy(same_indices).to(device)
                diff_indices_tensor = torch.from_numpy(diff_indices).to(device)
                
                max_diff_pairs = min(max_pairs, len(same_indices) * len(diff_indices))
                if max_diff_pairs < len(same_indices) * len(diff_indices):
                    # Need sampling – compute in GPU batches
                    base_seed = seed if seed is not None else 42
                    torch.manual_seed(int(base_seed) + int(i) * 2 + 1)
                    n_samples = max_diff_pairs
                    # Generate random index pairs
                    same_idx = torch.randint(0, len(same_indices), (n_samples,), device=device)
                    diff_idx = torch.randint(0, len(diff_indices), (n_samples,), device=device)
                    
                    # Compute distances in batches
                    embeddings_same = embeddings_tensor[same_indices_tensor[same_idx]]
                    embeddings_diff = embeddings_tensor[diff_indices_tensor[diff_idx]]
                    distances = compute_distances(embeddings_same, embeddings_diff, effective_metric)
                    diff_distances = distances.cpu().numpy().tolist()
                else:
                    # GPU batched computation for all different-class pairs
                    same_embeddings = embeddings_tensor[same_indices_tensor]  # [n_same, embedding_dim]
                    diff_embeddings = embeddings_tensor[diff_indices_tensor]  # [n_diff, embedding_dim]
                    # Use unified distance computation function for cross distances
                    if effective_metric == 'cosine':
                        # Cross cosine-distance computation (after normalization)
                        same_norm = torch.nn.functional.normalize(same_embeddings, p=2, dim=1)
                        diff_norm = torch.nn.functional.normalize(diff_embeddings, p=2, dim=1)
                        cos_sim_matrix = torch.mm(same_norm, diff_norm.t())
                        distances = 1.0 - cos_sim_matrix
                    elif effective_metric in ('angular', 'angle'):
                        same_norm = torch.nn.functional.normalize(same_embeddings, p=2, dim=1)
                        diff_norm = torch.nn.functional.normalize(diff_embeddings, p=2, dim=1)
                        cos_sim_matrix = torch.mm(same_norm, diff_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                        distances = torch.acos(cos_sim_matrix)
                    elif effective_metric in ('correlation', 'corr'):
                        same_c = same_embeddings - same_embeddings.mean(dim=1, keepdim=True)
                        diff_c = diff_embeddings - diff_embeddings.mean(dim=1, keepdim=True)
                        same_cn = same_c / (torch.norm(same_c, dim=1, keepdim=True) + 1e-12)
                        diff_cn = diff_c / (torch.norm(diff_c, dim=1, keepdim=True) + 1e-12)
                        corr_mat = torch.mm(same_cn, diff_cn.t())
                        distances = 1.0 - corr_mat
                    elif effective_metric == 'euclidean_zscore':
                        euclidean_matrix = torch.cdist(same_embeddings, diff_embeddings, p=2)
                        # Z-score normalization
                        euclidean_flat = euclidean_matrix.flatten()
                        euclidean_np = euclidean_flat.cpu().numpy()
                        if len(euclidean_np) > 1:
                            mean_dist = np.mean(euclidean_np)
                            std_dist = np.std(euclidean_np)
                            if std_dist > 1e-8:
                                zscore_flat = (euclidean_np - mean_dist) / std_dist
                                zscore_tensor = torch.from_numpy(zscore_flat).to(device)
                                distances = zscore_tensor.reshape(euclidean_matrix.shape)
                            else:
                                distances = euclidean_matrix
                        else:
                            distances = euclidean_matrix
                    else:  # euclidean
                        distances = torch.cdist(same_embeddings, diff_embeddings, p=2)
                    diff_distances = distances.flatten().cpu().numpy().tolist()
            
            if same_distances and diff_distances:
                # Mann–Whitney U test
                try:
                    u_stat, p_value = mannwhitneyu(same_distances, diff_distances, 
                                                 alternative='less')  # Hypothesis: same-class distances are smaller
                    
                    # Cliff's delta effect size
                    cliff_delta = self._compute_cliff_delta(same_distances, diff_distances)
                    
                    results[fg_name] = {
                        'same_distance_mean': float(np.mean(same_distances)),
                        'same_distance_std': float(np.std(same_distances)),
                        'diff_distance_mean': float(np.mean(diff_distances)),
                        'diff_distance_std': float(np.std(diff_distances)),
                        'u_statistic': float(u_stat),
                        'p_value': float(p_value),
                        'cliff_delta': float(cliff_delta),
                        'n_same_pairs': len(same_distances),
                        'n_diff_pairs': len(diff_distances)
                    }
                    p_values.append(p_value)
                    fg_keys.append(fg_name)
                    
                except Exception as e:
                    logger.warning(f"Distance analysis failed for {fg_name}: {e}")
        
        # Apply FDR (BH) correction for multiple comparisons in distance analysis
        if p_values:
            try:
                rejected, p_corrected, _, _ = smt.multipletests(p_values, alpha=self.alpha, method='fdr_bh')
                for name, p_corr, rej in zip(fg_keys, p_corrected, rejected):
                    if name in results:
                        results[name]['p_fdr'] = float(p_corr)
                        results[name]['significant'] = bool(rej)
            except Exception as e:
                logger.warning(f"FDR correction failed for distance analysis: {e}")

        return results
    
    def distance_property_consistency(self,
                                      embeddings: np.ndarray,
                                      properties: np.ndarray,
                                      max_pairs: int = 200000,
                                      use_gpu: bool = True,
                                      distance_metric: str = 'cosine',
                                      seed: Optional[int] = None,
                                      pre_transformed: bool = False,
                                      knn_k: int = 15) -> Dict[str, Any]:
        """
        Distance–property differential consistency (correlation-based).
        Computes the Spearman correlation between embedding distances D_ij and property differences |p_i - p_j|
        over sampled pairs, matching the style of distance_analysis without changing existing logic.
        """
        try:
            if embeddings is None or properties is None:
                return {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': distance_metric}
            properties = np.asarray(properties).reshape(-1)
            n = embeddings.shape[0]
            if properties.shape[0] != n:
                logger.warning(f"properties length {properties.shape[0]} != embeddings length {n}")
                return {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': distance_metric}
            if np.unique(properties).size < 2:
                return {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': distance_metric}
            
            # Optional GPU acceleration; whiten first for Mahalanobis
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            emb_np = np.asarray(embeddings, dtype=np.float32)
            eff_metric = distance_metric
            if distance_metric == 'mahalanobis':
                if pre_transformed:
                    # In a pre-transformed space (e.g., PCA), skip whitening and fall back to Euclidean
                    eff_metric = 'euclidean'
                else:
                    try:
                        n, d = emb_np.shape
                        n_comp = int(min(d, max(1, n - 1)))
                        pca_w = PCA(n_components=n_comp, whiten=True, random_state=(int(seed) if seed is not None else 42))
                        emb_np = pca_w.fit_transform(emb_np)
                        eff_metric = 'euclidean'
                    except Exception as e:
                        logger.warning(f"Mahalanobis whitening failed in distance_property_consistency, fallback to euclidean: {e}")
                        eff_metric = 'euclidean'
            x = torch.from_numpy(emb_np.astype(np.float32)).to(device)
            p = torch.from_numpy(properties.astype(np.float32)).to(device)
            
            total_pairs = n * (n - 1) // 2
            n_pairs = min(int(max_pairs), int(total_pairs))
            
            # Reproducible sampling
            rng = np.random.RandomState(int(seed) if seed is not None else 42)
            idx_i = rng.randint(0, n, size=n_pairs)
            idx_j = rng.randint(0, n, size=n_pairs)
            mask = idx_i < idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            # If sample count is insufficient, resample to fill
            while idx_i.size < n_pairs:
                need = n_pairs - idx_i.size
                ii = rng.randint(0, n, size=need)
                jj = rng.randint(0, n, size=need)
                m2 = ii < jj
                idx_i = np.concatenate([idx_i, ii[m2]])
                idx_j = np.concatenate([idx_j, jj[m2]])
            idx_i = idx_i[:n_pairs]; idx_j = idx_j[:n_pairs]
            
            xi, xj = x[idx_i], x[idx_j]
            if eff_metric == 'cosine':
                xi_n = torch.nn.functional.normalize(xi, p=2, dim=1)
                xj_n = torch.nn.functional.normalize(xj, p=2, dim=1)
                dist = 1.0 - torch.sum(xi_n * xj_n, dim=1)
            elif eff_metric in ('angular', 'angle'):
                xi_n = torch.nn.functional.normalize(xi, p=2, dim=1)
                xj_n = torch.nn.functional.normalize(xj, p=2, dim=1)
                cosv = torch.sum(xi_n * xj_n, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                dist = torch.acos(cosv)
            elif eff_metric in ('correlation', 'corr'):
                xi_c = xi - xi.mean(dim=1, keepdim=True)
                xj_c = xj - xj.mean(dim=1, keepdim=True)
                num = (xi_c * xj_c).sum(dim=1)
                den = (torch.norm(xi_c, dim=1) * torch.norm(xj_c, dim=1) + 1e-12)
                corr = num / den
                dist = 1.0 - corr
            elif eff_metric in ('euclidean', 'l2'):
                dist = torch.norm(xi - xj, dim=1)
            elif eff_metric == 'euclidean_zscore':
                d = torch.norm(xi - xj, dim=1).detach().cpu().numpy()
                if d.size > 1:
                    d = (d - d.mean()) / (d.std() + 1e-12)
                dist = torch.from_numpy(d).to(device)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            dp = torch.abs(p[idx_i] - p[idx_j])
            # Spearman correlation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(dist.detach().cpu().numpy(), dp.detach().cpu().numpy())
            rho = float(rho) if rho is not None else float('nan')
            pval = float(pval) if pval is not None else float('nan')

            # Build kNN graph (for spatial autocorrelation and smoothness)
            try:
                from sklearn.neighbors import NearestNeighbors
                y = properties.astype(float)
                n = emb_np.shape[0]
                k = int(max(1, min(knn_k, n - 1)))

                # Choose kNN metric
                nn_metric = 'euclidean'
                X_knn = emb_np
                if eff_metric in ('euclidean', 'l2', 'euclidean_zscore'):
                    nn_metric = 'euclidean'
                elif eff_metric in ('cosine', 'angular'):
                    nn_metric = 'cosine'
                elif eff_metric in ('correlation', 'corr'):
                    # Zero-center each row, then use cosine; equivalent to correlation distance
                    X_knn = emb_np - emb_np.mean(axis=1, keepdims=True)
                    nn_metric = 'cosine'
                else:
                    nn_metric = 'euclidean'

                nn = NearestNeighbors(n_neighbors=k + 1, metric=nn_metric)
                nn.fit(X_knn)
                dists, idx = nn.kneighbors(X_knn, n_neighbors=k + 1, return_distance=True)
                # Drop self-neighbor (first column)
                idx = idx[:, 1:]

                # Build symmetric, unweighted adjacency matrix
                import scipy.sparse as sp
                row = np.repeat(np.arange(n), k)
                col = idx.reshape(-1)
                data = np.ones_like(col, dtype=float)
                W = sp.coo_matrix((data, (row, col)), shape=(n, n))
                # Symmetrize
                W = (W + W.T)
                W.data[:] = 1.0
                W = W.tocsr()

                # Moran's I
                yc = y - y.mean()
                S0 = float(W.sum())
                moran_num = float(yc @ (W @ yc))
                moran_den = float((yc ** 2).sum())
                morans_i = float((n / S0) * (moran_num / moran_den)) if (S0 > 0 and moran_den > 0) else float('nan')

                # Geary's C
                # Use directed sum over (i,j): sum w_ij (y_i - y_j)^2 = 2 y^T L y (when W is symmetric)
                D = sp.diags(W.sum(axis=1).A.ravel())
                L = (D - W)
                dirichlet = float(yc @ (L @ yc))
                geary_num = 2.0 * dirichlet
                gearys_c = float(((n - 1) / (2.0 * S0)) * (geary_num / moran_den)) if (S0 > 0 and moran_den > 0) else float('nan')

                # Dirichlet energy E = y^T L y (more stable for zero-centered y)
                dirichlet_energy = float(dirichlet)

                # kNN neighborhood consistency: kNN mean prediction
                y_hat = np.zeros(n, dtype=float)
                for i in range(n):
                    y_hat[i] = float(np.mean(y[idx[i]])) if k > 0 else float('nan')
                resid = y - y_hat
                rmse_mean = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else float('nan')
                ss_res = float(np.sum(resid ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2_mean = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')

            except Exception as ex:
                logger.warning(f"kNN metrics failed: {ex}")
                morans_i = float('nan'); gearys_c = float('nan'); dirichlet_energy = float('nan')
                rmse_mean = float('nan'); r2_mean = float('nan')

            return {
                'spearman_rho': rho,
                'spearman_p': pval,
                'n_pairs': int(n_pairs),
                'distance_metric': distance_metric,
                'morans_i': morans_i,
                'gearys_c': gearys_c,
                'dirichlet_energy': dirichlet_energy,
                'knn_rmse_mean': rmse_mean,
                'knn_r2_mean': r2_mean,
                'knn_k': int(knn_k)
            }
        except Exception as e:
            logger.warning(f"distance_property_consistency failed: {e}")
            return {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': distance_metric}

    def _compute_cliff_delta(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cliff's delta effect size."""
        try:
            n1, n2 = len(group1), len(group2)
            if n1 == 0 or n2 == 0:
                return 0.0
            
            dominance = 0
            for x1 in group1:
                for x2 in group2:
                    if x1 < x2:
                        dominance += 1
                    elif x1 > x2:
                        dominance -= 1
            
            return dominance / (n1 * n2)
            
        except Exception:
            return 0.0


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, args):
        """
        Initialize the experiment runner.
        
        Args:
            args: command-line arguments
        """
        self.args = args
        # Device selection: prefer CLI argument, otherwise choose automatically
        if getattr(args, 'device', None):
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        
        # Initialize components
        self.annotator = FunctionalGroupAnnotator()
        self.linear_probe = LinearProbe(random_state=args.random_seed)
        self.clustering = ClusteringAnalysis(random_state=args.random_seed)
        self.stats = StatisticalAnalysis()
        # Track dataset indices for which each model successfully produced embeddings, to keep analyses aligned
        self.sample_indices: Dict[str, List[int]] = {}
        
        # Create output directory (timestamped subdirectory already created upstream)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure visualization style
        setup_plotting_style()
        
        logger.info(f"Experiment initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def _extract_property_vector(self, dataset: MolecularDataset, indices: List[int]) -> Optional[np.ndarray]:
        """
        Try to extract a scalar property p_i from dataset.graph_data or original_item.
        Compatible fields: y/label/labels/target/targets; for multi-task targets, take the first task.
        Returns an np.ndarray with the same length as indices; returns None if no property can be extracted.
        """
        props: List[float] = []
        had = False
        for idx in indices:
            try:
                item = dataset[idx]
                val = None
                gd = item.get('graph_data', None)
                # 1) Common fields on graph_data
                for attr in ('y', 'label', 'labels', 'target', 'targets'):
                    if gd is not None and hasattr(gd, attr):
                        v = getattr(gd, attr)
                        try:
                            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                            scalar = float(arr.reshape(-1)[0])
                            if not (np.isnan(scalar) or np.isinf(scalar)):
                                val = scalar
                                break
                        except Exception:
                            pass
                # 2) Fallback to original_item
                if val is None and isinstance(item.get('original_item', None), dict):
                    oi = item['original_item']
                    for key in ('y', 'label', 'labels', 'target', 'targets'):
                        if key in oi:
                            try:
                                arr = np.asarray(oi[key])
                                scalar = float(arr.reshape(-1)[0])
                                if not (np.isnan(scalar) or np.isinf(scalar)):
                                    val = scalar
                                    break
                            except Exception:
                                pass
                if val is None:
                    props.append(np.nan)
                else:
                    props.append(val); had = True
            except Exception:
                props.append(np.nan)
        arr = np.asarray(props, dtype=float)
        if not had:
            logger.warning("No property values found for selected samples; skip distance–property consistency")
            return None
        return arr

    def load_models(self) -> Dict[str, Any]:
        """Load models and return a dict, using explicit config + state dict paths."""
        models: Dict[str, Any] = {}

        # Parse semantic models: support 1 or 2 paths; 2 paths = comparison mode (order: pre -> post)
        sem_paths = self.args.semantic_model_path
        sem_cfgs = getattr(self.args, 'semantic_model_config', None)

        if isinstance(sem_paths, list):
            if len(sem_paths) not in (1, 2):
                raise ValueError('--semantic_model_path only allows 1 or 2 paths')
            if sem_cfgs is None:
                raise ValueError('Both comparison mode and single-model mode require --semantic_model_config with matching count')
            if not isinstance(sem_cfgs, list):
                raise ValueError('--semantic_model_config must be a list with 1 or 2 paths, matching --semantic_model_path')
            if len(sem_cfgs) != len(sem_paths):
                raise ValueError('--semantic_model_path and --semantic_model_config must have the same number of entries')

            if len(sem_paths) == 1:
                entries = [('semantic', (sem_paths[0], sem_cfgs[0]))]
            else:
                # Comparison mode: base (original / before continued training) → delta (after continued training)
                entries = [('base', (sem_paths[0], sem_cfgs[0])), ('delta', (sem_paths[1], sem_cfgs[1]))]
        else:
            # Backward-compatible single-path string (not recommended)
            if sem_paths is None:
                entries = []
            else:
                if sem_cfgs is None or isinstance(sem_cfgs, list):
                    raise ValueError('In single-model mode, --semantic_model_config must be a single config path')
                entries = [('semantic', (sem_paths, sem_cfgs))]

        # Other model entries stay as-is (optional)
        other_entries = []
        if getattr(self.args, 'random_model_path', None):
            other_entries.append(('random', (self.args.random_model_path, getattr(self.args, 'random_model_config', None))))
        if getattr(self.args, 'uniform_model_path', None):
            other_entries.append(('uniform', (self.args.uniform_model_path, getattr(self.args, 'uniform_model_config', None))))

        model_entries = entries + other_entries

        for model_type, (model_path, config_path) in model_entries:
            if not model_path:
                continue

            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                logger.warning(f"Model path not found for {model_type}: {model_path}")
                continue

            try:
                logger.info(f"Loading {model_type} model from {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                logger.error(f"Failed to load checkpoint for {model_type}: {e}")
                continue

            # Read config and fill in critical fields
            raw_config = load_model_config(str(model_path_obj), config_path)
            model_config = prepare_model_config(raw_config)

            if hasattr(checkpoint, 'get_embedding'):
                model = checkpoint
                state_dict = None
            else:
                state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
                if state_dict is None:
                    logger.warning(f"Checkpoint for {model_type} missing model_state_dict/state_dict")
                    continue
                state_dict = dict(state_dict)

                model_params = model_config.get('model', {})
                feature_cfg = model_config.get('features', {})
                in_dim = (feature_cfg.get('atom', {}).get('dim') or
                          feature_cfg.get('hyperedge_dim') or
                          model_params.get('feature_dim') or 25)

                model = EnhancedHyperGraphMAE(
                    in_dim=int(in_dim),
                    hidden_dim=model_params.get('hidden_dim', 384),
                    latent_dim=model_params.get('latent_dim', 192),
                    proj_dim=model_params.get('proj_dim', 256),
                    heads=model_params.get('heads', 8),
                    num_layers=model_params.get('num_layers', 4),
                    mask_ratio=model_params.get('mask_ratio', 0.7),
                    config=model_config,
                )

                # Adjust descriptor head structure based on state dict
                descriptor_keys = [k for k in state_dict.keys() if k.startswith('descriptor_head')]
                if descriptor_keys:
                    has_linear_layout = 'descriptor_head.weight' in state_dict
                    has_sequential_layout = any('.' in k[len('descriptor_head'):] for k in descriptor_keys if k != 'descriptor_head.weight')
                    if has_linear_layout and not has_sequential_layout:
                        if isinstance(model.descriptor_head, nn.Sequential):
                            out_dim = state_dict['descriptor_head.bias'].shape[0]
                            in_dim_head = state_dict['descriptor_head.weight'].shape[1]
                            model.descriptor_head = nn.Linear(in_dim_head, out_dim)
                    elif not has_linear_layout and isinstance(model.descriptor_head, nn.Linear):
                        desc_cfg = model_config.get('descriptor_head', {})
                        hidden_dim = int(desc_cfg.get('hidden_dim', 128))
                        out_dim = len(model.descriptor_names)
                        proj_dim_val = int(model_params.get('proj_dim', 256))
                        model.descriptor_head = nn.Sequential(
                            nn.Linear(proj_dim_val, hidden_dim),
                            nn.ELU(inplace=True),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim, out_dim)
                        )
                else:
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('descriptor_head')}

                load_info = model.load_state_dict(state_dict, strict=False)
                missing = getattr(load_info, 'missing_keys', [])
                unexpected = getattr(load_info, 'unexpected_keys', [])
                if missing or unexpected:
                    logger.debug(f"State dict alignment for {model_type}: missing={missing}, unexpected={unexpected}")

            model.to(self.device)
            model.eval()

            if not hasattr(model, 'get_embedding'):
                logger.error(f"Model {model_type} missing get_embedding method")
                continue

            models[model_type] = {
                'model': model,
                'path': str(model_path_obj),
                'config': model_config,
            }
            logger.info(f"Successfully loaded {model_type} model")

        if not models:
            raise ValueError("No valid models loaded. Please provide valid model paths.")

            logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")
        return models


    def load_datasets(self, models: Dict[str, Any] = None) -> List[Tuple[str, MolecularDataset]]:
        """Load datasets, supporting a single or multiple datasets."""
        datasets = []

        # Obtain hypergraph_types and hyperedge-dimension configuration from model config
        model_config = None
        if models and 'semantic' in models:
            # Prefer config cached when loading the model
            model_config = models['semantic'].get('config')
            if model_config is None:
                semantic_model_path = models['semantic']['path']
                model_config = prepare_model_config(load_model_config(semantic_model_path))
            configure_hyperedge_alignment(model_config)

        if self.args.dataset_path:
            # Classic single-dataset mode
            logger.info(f"Loading single dataset from {self.args.dataset_path}")
            dataset = self._load_single_dataset(self.args.dataset_path, model_config)
            datasets.append(("custom", dataset))

        elif self.args.data_root and self.args.datasets:
            # Multi-dataset mode (with alias support)
            logger.info(f"Loading multiple datasets from {self.args.data_root}")
            for raw_name in self.args.datasets:
                canonical_name = DATASET_ALIASES.get(raw_name, raw_name)
                primary = Path(self.args.data_root) / raw_name
                secondary = Path(self.args.data_root) / canonical_name
                rev_aliases = CANONICAL_TO_ALIASES.get(canonical_name, [])
                alt_paths = [Path(self.args.data_root) / alt for alt in rev_aliases]
                candidates = [primary, secondary] + alt_paths

                dataset_path = next((p for p in candidates if p.exists()), None)
                if dataset_path is not None:
                    logger.info(f"Loading dataset: {canonical_name} (from {dataset_path.name})")
                    dataset = self._load_single_dataset(str(dataset_path), model_config)
                    datasets.append((canonical_name, dataset))
                else:
                    logger.warning(f"Dataset {raw_name} not found under {self.args.data_root} (tried: {[p.name for p in candidates]} )")

        if not datasets:
            raise ValueError("No datasets loaded successfully")

        logger.info(f"Successfully loaded {len(datasets)} dataset(s): {[name for name, _ in datasets]}")
        return datasets


    def _load_single_dataset(self, data_path: str, model_config: Dict = None) -> MolecularDataset:
        """Load a single dataset."""
        data_path = Path(data_path)
        
        # Heuristically detect whether this is preprocessed hypergraph data
        use_hypergraph_data = False
        
        if data_path.is_dir():
            # Check whether preprocessed .pt files are present
            if any(data_path.glob("batch_*.pt")):
                use_hypergraph_data = True
                logger.debug(f"Detected preprocessed hypergraph data: {data_path}")
            elif any(data_path.glob("*.pt")):
                use_hypergraph_data = True
                logger.debug(f"Detected PyTorch data files: {data_path}")
        elif data_path.suffix in ['.pt', '.pth']:
            use_hypergraph_data = True
            logger.debug(f"Detected PyTorch data file: {data_path}")
        else:
            logger.debug(f"Using SMILES data loading mode: {data_path}")
        
        dataset = MolecularDataset(
            data_path=str(data_path),
            sample_size=self.args.sample_size,
            random_seed=self.args.random_seed,
            use_hypergraph_data=use_hypergraph_data
        )
        
        # If a model config is available, set target hypergraph types
        if model_config and 'hypergraph_types' in model_config:
            target_types = model_config['hypergraph_types']
            dataset.set_target_hypergraph_types(target_types)
            logger.debug(f"Set target hypergraph types: {target_types}")
        
        return dataset

    def extract_embeddings(self, models: Dict[str, Any], 
                          dataset: MolecularDataset) -> Dict[str, np.ndarray]:
        """
        Extract molecular representations from all models.
        
        Args:
            models: model dictionary
            dataset: molecular dataset
            
        Returns:
            Mapping from model name to embedding matrix
        """
        embeddings = {}
        
        for model_type, model_info in models.items():
            logger.info(f"Extracting embeddings from {model_type} model")
            model = model_info['model']
            
            # Detect and log model parameter dtype (once)
            model_dtype = next(model.parameters()).dtype
            logger.info(f"Model dtype: {model_dtype}")
            
            model_embeddings = []
            
            # Process molecules in batches
            batch_size = int(getattr(self.args, 'batch_size', 32))
            valid_indices: List[int] = []
            total = len(dataset)
            for i in range(0, total, batch_size):
                start = i
                end = min(i + batch_size, total)
                batch_items = [dataset[j] for j in range(start, end)]
                
                try:
                    batch_embeddings, valid_flags = self._extract_batch_embeddings(model, batch_items)
                    model_embeddings.append(batch_embeddings)
                    # Collect original sample indices corresponding to appended embeddings
                    for local_idx, ok in enumerate(valid_flags.tolist()):
                        if ok:
                            valid_indices.append(start + local_idx)
                except Exception as e:
                    logger.error(f"Failed to extract embeddings for batch {i//batch_size}: {e}")
                    continue
                # Progress logging
                if ((i // batch_size) % max(1, 1000 // max(1, batch_size))) == 0 or end == total:
                    logger.info(f"{model_type}: processed {end}/{total} molecules")
            
            if model_embeddings:
                # Concatenate embeddings from all batches (keep raw space; avoid global normalization that could change directions/distances)
                all_embeddings = np.vstack(model_embeddings)
                
                embeddings[model_type] = all_embeddings
                # Save sample-index mapping for downstream FG/NMI/visualization alignment
                self.sample_indices[model_type] = valid_indices
                logger.info(f"Extracted {len(all_embeddings)} embeddings from {model_type} model")
        
        return embeddings

    def _extract_batch_embeddings(self, model: nn.Module, batch_items: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for a single batch, handling preprocessed data and raw SMILES.
        """
        batch_embeddings = []
        valid_flags: List[bool] = []
        
        # Detect model parameter dtype
        model_dtype = next(model.parameters()).dtype
        
        with torch.no_grad():
            for item in batch_items:
                try:
                    mol = item['mol']
                    smiles = item['smiles']
                    graph_data = item.get('graph_data', None)
                    
                    # Prefer preprocessed hypergraph data when available
                    if graph_data is not None:
                        # Use preprocessed hypergraph data and ensure feature dtypes match the model
                        # Note: only feature tensors need to match the model dtype; index tensors stay long
                        x = graph_data.x.to(self.device, dtype=model_dtype) if hasattr(graph_data, 'x') else None
                        hyperedge_index = graph_data.hyperedge_index.to(self.device, dtype=torch.long) if hasattr(graph_data, 'hyperedge_index') else None
                        hyperedge_attr = graph_data.hyperedge_attr.to(self.device, dtype=model_dtype) if hasattr(graph_data, 'hyperedge_attr') else None
                        
                        
                        if x is not None and hyperedge_index is not None:
                            # Get molecular representation; use autocast to match training (CUDA only)
                            use_amp = torch.cuda.is_available() and self.device.type == 'cuda'
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                embedding = model.get_embedding(x, hyperedge_index, hyperedge_attr)
                            
                            # Pool to molecule-level representation
                            mol_embedding = embedding.mean(dim=0).cpu().numpy()
                            batch_embeddings.append(mol_embedding)
                            valid_flags.append(True)
                            continue
                    
                    # If no preprocessed data is available, skip this molecule
                    logger.warning(f"No preprocessed data available for {smiles}, skip this molecule")
                    valid_flags.append(False)
                    continue
                        
                except Exception as e:
                    logger.warning(f"Error processing molecule {item.get('smiles', 'unknown')}: {e}; skip this molecule")
                    valid_flags.append(False)
                    continue
        
        return np.array(batch_embeddings), np.array(valid_flags, dtype=bool)


    def run_clustering_analysis(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run clustering analysis."""
        logger.info("Running clustering analysis")
        
        clustering_results = {}
        
        for model_type, embedding_matrix in embeddings.items():
            logger.info(f"Clustering analysis for {model_type} model")
            method = getattr(self.args, 'clustering_method', 'hdbscan')
            if method == 'kmeans' and getattr(self.args, 'grid_search_kmeans', False):
                # Grid search: PCA components × K; choose best by CH first, DB second
                best = None
                pca_cands = getattr(self.args, 'pca_candidates', [20, 30, 40])
                k_cands = getattr(self.args, 'kmeans_k_candidates', [6, 8, 10, 12, 15])
                n, d = embedding_matrix.shape
                for p in pca_cands:
                    n_comp = int(min(max(2, p), n - 1, d))
                    try:
                        pca = PCA(n_components=n_comp, whiten=False, random_state=self.args.random_seed)
                        z = pca.fit_transform(embedding_matrix)
                    except Exception as e:
                        logger.debug(f"PCA n_components={n_comp} failed: {e}")
                        continue
                    for k in k_cands:
                        if not (1 < k < n):
                            continue
                        try:
                            km = KMeans(n_clusters=int(k), n_init=10, random_state=self.args.random_seed)
                            labels = km.fit_predict(z)
                            # Metrics
                            ch = calinski_harabasz_score(z, labels)
                            try:
                                db = davies_bouldin_score(z, labels)
                            except Exception:
                                db = np.inf
                            try:
                                sil = silhouette_score(z, labels, metric='euclidean')
                            except Exception:
                                sil = 0.0
                            # Composite score: higher CH, lower DB, higher Sil are better
                            score = ch - 0.5 * db + 50.0 * sil
                            cand = (score, ch, -db, sil, n_comp, int(k), labels, pca, z)
                            if best is None or cand > best:
                                best = cand
                        except Exception as e:
                            logger.debug(f"KMeans k={k} failed: {e}")
                            continue
                if best is None:
                    # Fallback: use default pca_components and automatic k
                    desired_components = int(getattr(self.args, 'pca_components', 30))
                    n_comp = int(min(max(2, desired_components), n - 1, d))
                    pca = PCA(n_components=n_comp, whiten=False, random_state=self.args.random_seed)
                    z = pca.fit_transform(embedding_matrix)
                    cluster_labels, metrics = self.clustering.perform_clustering(
                        z, method='kmeans', optimize_params=True, kmeans_k=getattr(self.args, 'kmeans_k', None)
                    )
                    clustering_results[model_type] = {
                        'pca_embeddings': z,
                        'cluster_labels': cluster_labels,
                        'metrics': metrics,
                        'pca_explained_variance_ratio': pca.explained_variance_ratio_
                    }
                else:
                    score, ch, neg_db, sil, n_comp, k_best, labels, pca, z = best
                    db = -neg_db
                    metrics = {
                        'calinski_harabasz': float(ch),
                        'davies_bouldin': float(db if np.isfinite(db) else 0.0),
                        'silhouette': float(sil),
                        'n_clusters': int(len(np.unique(labels))),
                        'n_noise': 0,
                        'noise_ratio': 0.0,
                        'dbcv': 0.0
                    }
                    logger.info(f"Best KMeans found: pca={n_comp}, k={k_best}, CH={ch:.3f}, DB={db:.3f}, Sil={sil:.3f}")
                    clustering_results[model_type] = {
                        'pca_embeddings': z,
                        'cluster_labels': labels,
                        'metrics': metrics,
                        'pca_explained_variance_ratio': pca.explained_variance_ratio_
                    }
            else:
                # Standard single PCA + clustering pass
                desired_components = int(getattr(self.args, 'pca_components', 30))
                n, d = embedding_matrix.shape
                n_comp = int(min(max(2, desired_components), n - 1, d))
                pca = PCA(n_components=n_comp, whiten=False, random_state=self.args.random_seed)
                logger.info(f"Using {n_comp} whitened PCA components for {n} samples")
                z = pca.fit_transform(embedding_matrix)
                method = getattr(self.args, 'clustering_method', 'hdbscan')
                kmeans_k = getattr(self.args, 'kmeans_k', None)
                cluster_labels, metrics = self.clustering.perform_clustering(
                    z, method=method, optimize_params=True, kmeans_k=kmeans_k
                )
                clustering_results[model_type] = {
                    'pca_embeddings': z,
                    'cluster_labels': cluster_labels,
                    'metrics': metrics,
                    'pca_explained_variance_ratio': pca.explained_variance_ratio_
                }
            
            logger.info(f"{model_type} clustering: {metrics['n_clusters']} clusters, "
                       f"{metrics['noise_ratio']:.2%} noise")
        
        return clustering_results

    def run_functional_group_analysis(self, dataset: MolecularDataset, 
                                    embeddings: Dict[str, np.ndarray],
                                    clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run functional-group analysis."""
        logger.info("Running functional group analysis")
        
        # Cache all molecules up front
        molecules_all = [item['mol'] for item in dataset]
        analysis_results = {}
        
        for model_type in embeddings.keys():
            logger.info(f"Functional group analysis for {model_type} model")
            
            model_embeddings = embeddings[model_type]
            pca_embeddings = clustering_results[model_type]['pca_embeddings']
            cluster_labels = clustering_results[model_type]['cluster_labels']
            # Align samples: only use indices where embeddings were successfully generated
            indices = self.sample_indices.get(model_type, list(range(len(model_embeddings))))
            molecules = [molecules_all[idx] for idx in indices]
            # Rebuild FG matrix from aligned molecules
            fg_matrix, fg_names = self.annotator.create_multilabel_matrix(molecules)
            logger.info(f"Identified {len(fg_names)} functional group types")
            
            # Linear probe analysis
            linear_probe_results = {}
            for i, fg_name in enumerate(fg_names):
                fg_labels = fg_matrix[:, i]
                if np.sum(fg_labels) >= 10:  # Require at least 10 positive samples
                    probe_result = self.linear_probe.evaluate_functional_group(
                        model_embeddings, fg_labels, fg_name
                    )
                    linear_probe_results[fg_name] = probe_result
            
            # External validation via NMI: consistency between clusters and functional groups
            nmi_results = {}
            nmi_scores = []
            
            for i, fg_name in enumerate(fg_names):
                fg_labels = fg_matrix[:, i]
                if np.sum(fg_labels) >= 5:  # Require at least 5 positives to compute NMI
                    # Compute adjusted mutual information (correcting for chance)
                    nmi_all = adjusted_mutual_info_score(cluster_labels, fg_labels)
                    # Noise-free perspective (valid only for HDBSCAN)
                    if np.any(cluster_labels < 0):
                        valid_mask = cluster_labels >= 0
                        if np.sum(valid_mask) > 1:
                            nmi_valid = adjusted_mutual_info_score(cluster_labels[valid_mask], fg_labels[valid_mask])
                        else:
                            nmi_valid = 0.0
                    else:
                        nmi_valid = nmi_all
                    nmi_results[fg_name] = {
                        'nmi_all': float(nmi_all),
                        'nmi_valid': float(nmi_valid),
                        'n_positive': int(np.sum(fg_labels)),
                        'n_negative': int(np.sum(1 - fg_labels))
                    }
                    nmi_scores.append(nmi_valid)
            
            # NMI summary statistics
            nmi_summary = {
                'mean_nmi_valid': float(np.mean(nmi_scores)) if nmi_scores else 0.0,
                'max_nmi_valid': float(np.max(nmi_scores)) if nmi_scores else 0.0,
                'std_nmi_valid': float(np.std(nmi_scores)) if nmi_scores else 0.0,
                'n_functional_groups': len(nmi_scores)
            }
            
            logger.info(f"NMI analysis: mean_valid={nmi_summary['mean_nmi_valid']:.3f}, "
                       f"max_valid={nmi_summary['max_nmi_valid']:.3f}, "
                       f"groups_analyzed={nmi_summary['n_functional_groups']}")
            
            # Fisher exact-test enrichment analysis
            enrichment_results = self.stats.fisher_exact_enrichment(
                cluster_labels, fg_matrix, fg_names
            )
            
            # Distance analysis and distance–property consistency using configurable metric (--distance_metric)
            # Support raw/PCA space (--distance_property_space, default raw)
            try:
                indices = self.sample_indices.get(model_type, list(range(len(model_embeddings))))
                prop_values = self._extract_property_vector(dataset, indices)
                distance_property = None
                if prop_values is not None:
                    mask_valid = ~np.isnan(prop_values)
                    if mask_valid.sum() >= 2:
                        dp_space = getattr(self.args, 'distance_property_space', 'raw')
                        dp_source = model_embeddings if dp_space == 'raw' else pca_embeddings
                        dp_result = self.stats.distance_property_consistency(
                            dp_source[mask_valid],
                            prop_values[mask_valid],
                            max_pairs=int(getattr(self.args, 'max_property_pairs', 50000)),
                            use_gpu=(self.device.type == 'cuda'),
                            distance_metric=getattr(self.args, 'distance_metric', 'cosine'),
                            seed=self.args.random_seed,
                            pre_transformed=(dp_space == 'pca'),
                            knn_k=int(getattr(self.args, 'knn_k', 15))
                        )
                        # Attach space and dimensionality information
                        dp_result['space'] = dp_space
                        if dp_space == 'pca':
                            try:
                                dp_result['n_components'] = int(pca_embeddings.shape[1])
                            except Exception:
                                dp_result['n_components'] = None
                        distance_property = dp_result
                if distance_property is None:
                    distance_property = {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': 'euclidean'}
            except Exception as e:
                logger.warning(f"distance–property consistency failed: {e}")
                distance_property = {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': 'euclidean'}

            distance_results = self.stats.distance_analysis(
                pca_embeddings, fg_matrix, fg_names,
                use_gpu=(self.device.type == 'cuda'),
                distance_metric=getattr(self.args, 'distance_metric', 'cosine'),
                seed=self.args.random_seed
            )
            
            analysis_results[model_type] = {
                'linear_probe': linear_probe_results,
                'nmi_analysis': nmi_results,
                'nmi_summary': nmi_summary,
                'enrichment': enrichment_results,
                'distance_analysis': distance_results,
                'distance_property': distance_property,
                'fg_matrix': fg_matrix,
                'fg_names': fg_names
            }
            
            logger.info(f"Completed analysis for {model_type}: "
                       f"{len(linear_probe_results)} functional groups analyzed")
        
        return analysis_results

    def create_visualizations(self, dataset: MolecularDataset,
                            embeddings: Dict[str, np.ndarray],
                            clustering_results: Dict[str, Any],
                            fg_analysis: Dict[str, Any]) -> None:
        """Create visualization figures."""
        if getattr(self.args, 'skip_visualization', False):
            logger.info("Skipping visualizations as requested")
            return
        logger.info("Creating visualizations")
        
        # 1. UMAP visualization
        self._create_umap_visualizations(embeddings, clustering_results, fg_analysis)

        # 2. Functional-group enrichment heatmaps
        self._create_enrichment_heatmaps(fg_analysis)

        # 3. Linear-probe and distance visualizations (per model, to avoid complex grouping)
        self._create_linear_probe_plots(fg_analysis)
        self._create_distance_plots(fg_analysis)

        logger.info("Visualizations saved to output directory")

    def _create_umap_visualizations(self, embeddings: Dict[str, np.ndarray],
                                  clustering_results: Dict[str, Any],
                                  fg_analysis: Dict[str, Any]) -> None:
        """Create UMAP visualizations."""
        for model_type, embedding_matrix in embeddings.items():
            logger.info(f"Creating UMAP visualization for {model_type}")
            
            # Use the same PCA space as clustering for UMAP, to keep things consistent
            pca_embeddings = clustering_results[model_type]['pca_embeddings']
            reducer = umap.UMAP(
                n_components=2, 
                random_state=self.args.random_seed, 
                n_neighbors=15,
                n_jobs=-1  # Use all CPU cores
            )
            umap_2d = reducer.fit_transform(pca_embeddings)
            
            cluster_labels = clustering_results[model_type]['cluster_labels']
            
            # Create interactive visualization
            fig = go.Figure()
            
            # Color by cluster
            unique_clusters = np.unique(cluster_labels)
            # Use modulo to avoid running out of palette colors
            palette = px.colors.qualitative.Set3
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_labels == cluster_id
                cluster_name = f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise'
                color = palette[i % len(palette)]  # Reuse colors cyclically
                
                fig.add_trace(go.Scatter(
                    x=umap_2d[mask, 0],
                    y=umap_2d[mask, 1],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(
                        color=color if cluster_id >= 0 else 'lightgray',
                        size=5,
                        opacity=0.7
                    ),
                    showlegend=True
                ))
            
            fig.update_layout(
                title=f'UMAP Visualization - {model_type.title()} Model',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                width=800,
                height=600
            )
            
            fig.write_html(self.output_dir / f'umap_{model_type}_clusters.html')

    def _create_enrichment_heatmaps(self, fg_analysis: Dict[str, Any]) -> None:
        """Create functional-group enrichment heatmaps."""
        for model_type, analysis in fg_analysis.items():
            enrichment = analysis['enrichment']
            
            # Extract enrichment data
            enrichment_data = []
            
            for fg_name, fg_results in enrichment['functional_groups'].items():
                for cluster_id, cluster_info in fg_results['clusters'].items():
                    enrichment_data.append({
                        'functional_group': fg_name,
                        'cluster': f'Cluster {cluster_id}',
                        'enrichment_fold': cluster_info['enrichment_fold'],  # Use the new field name
                        'cluster_fg_ratio': cluster_info['cluster_fg_ratio'],  # Within-cluster fraction
                        'p_value': cluster_info['p_value'],
                        'p_fdr': cluster_info.get('p_fdr', cluster_info['p_value']),  # FDR-adjusted p-value
                        'significant': cluster_info.get('significant', False),
                        'odds_ratio': cluster_info['odds_ratio']
                    })
            
            if enrichment_data:
                df = pd.DataFrame(enrichment_data)
                for col in ['enrichment_fold', 'p_value', 'p_fdr', 'cluster_fg_ratio', 'odds_ratio']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                pivot_enrichment = df.pivot(index='functional_group',
                                          columns='cluster',
                                          values='enrichment_fold')
                # p_fdr pivot table (for significance masking)
                pivot_pval = df.pivot(index='functional_group', columns='cluster', values='p_fdr')

                row_labels = list(pivot_enrichment.index.astype(str))
                col_labels = list(pivot_enrichment.columns.astype(str))
                data_matrix = np.asarray(pivot_enrichment.reindex(index=row_labels, columns=col_labels).fillna(np.nan).to_numpy(), dtype=float)
                if data_matrix.size == 0 or data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
                    logger.warning(f"Empty enrichment matrix for {model_type}, skipping heatmap")
                    continue

                # Significance mask: p_fdr < alpha → significant
                alpha = getattr(self.stats, 'alpha', 0.05)
                pval_mat = np.asarray(pivot_pval.reindex(index=row_labels, columns=col_labels).fillna(1.0).to_numpy(), dtype=float)
                sig_mask = pval_mat < alpha

                # Two-layer heatmap:
                # - Non-significant layer (grayed out, no colorbar)
                # - Significant layer (RdBu colors, with star marker in text)
                z_ns = np.where(~sig_mask, 1.0, np.nan)
                text_ns = [[f"{data_matrix[i,j]:.2f}" if (not sig_mask[i,j]) and np.isfinite(data_matrix[i,j]) else ""
                            for j in range(data_matrix.shape[1])] for i in range(data_matrix.shape[0])]

                z_sig = np.where(sig_mask, data_matrix, np.nan)
                text_sig = [[(f"{data_matrix[i,j]:.2f}★" if sig_mask[i,j] and np.isfinite(data_matrix[i,j]) else "")
                            for j in range(data_matrix.shape[1])] for i in range(data_matrix.shape[0])]

                fig = go.Figure()
                # Non-significant layer (gray)
                fig.add_trace(go.Heatmap(
                    z=z_ns,
                    x=col_labels,
                    y=row_labels,
                    colorscale=[[0, '#E0E0E0'], [1, '#E0E0E0']],
                    showscale=False,
                    hoverinfo='skip',
                    text=text_ns,
                    texttemplate="%{text}",
                ))
                # Significant layer (colored + star)
                fig.add_trace(go.Heatmap(
                    z=z_sig,
                    x=col_labels,
                    y=row_labels,
                    colorscale='RdBu',
                    reversescale=True,
                    colorbar=dict(title='Enrichment Fold'),
                    text=text_sig,
                    texttemplate="%{text}",
                ))

                fig.update_layout(
                    title=f'Functional Group Enrichment - {model_type.title()} Model (★: FDR<{alpha})',
                    xaxis_title='Cluster', yaxis_title='Functional Group',
                    width=900, height=600
                )
                fig.write_html(self.output_dir / f'enrichment_heatmap_{model_type}.html')

    def _create_linear_probe_plots(self, fg_analysis: Dict[str, Any]) -> None:
        """Create per-model bar charts for linear-probe AUROC/AUPRC (Plotly HTML)."""
        min_pos = max(5, int(getattr(self.args, 'min_functional_group_samples', 10)))
        top_k = 20
        for model_type, analysis in fg_analysis.items():
            lp = analysis.get('linear_probe', {})
            rows = []
            for fg_name, m in lp.items():
                try:
                    npos = int(m.get('n_positive', 0))
                    auroc = float(m.get('auroc', 0.0))
                    auprc = float(m.get('auprc', 0.0))
                except Exception:
                    continue
                if npos >= min_pos:
                    rows.append((fg_name, auroc, auprc))
            if not rows:
                continue
            # AUROC plot (HTML)
            rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)[:top_k]
            labels = [r[0] for r in rows_sorted]
            values = np.asarray([r[1] for r in rows_sorted], dtype=float)
            fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color='#4C72B0'))
            fig.update_layout(title=f'Linear Probe AUROC - {model_type}', xaxis=dict(range=[0,1]))
            fig.write_html(self.output_dir / f'linear_probe_auroc_{model_type}.html')

            # AUPRC plot (HTML)
            rows_sorted = sorted(rows, key=lambda x: x[2], reverse=True)[:top_k]
            labels = [r[0] for r in rows_sorted]
            values = np.asarray([r[2] for r in rows_sorted], dtype=float)
            fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color='#55A868'))
            fig.update_layout(title=f'Linear Probe AUPRC - {model_type}', xaxis=dict(range=[0,1]))
            fig.write_html(self.output_dir / f'linear_probe_auprc_{model_type}.html')

    def _create_distance_plots(self, fg_analysis: Dict[str, Any]) -> None:
        """Create per-model bar charts for distance analysis (Cliff's delta and distance difference)."""
        p_thresh = 0.05
        top_k = 20
        for model_type, analysis in fg_analysis.items():
            dist = analysis.get('distance_analysis', {})
            rows = []
            for fg_name, m in dist.items():
                try:
                    p = float(m.get('p_value', 1.0))
                    cd = float(m.get('cliff_delta', 0.0))
                    same_mean = float(m.get('same_distance_mean', np.nan))
                    diff_mean = float(m.get('diff_distance_mean', np.nan))
                    ddiff = same_mean - diff_mean
                except Exception:
                    continue
                if not np.isfinite(cd) or not np.isfinite(ddiff):
                    continue
                if p < p_thresh:
                    rows.append((fg_name, cd, ddiff))
            if not rows:
                continue
            # Cliff's delta plot ([-1,1]) (HTML)
            rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:top_k]
            labels = [r[0] for r in rows_sorted]
            values = np.asarray([r[1] for r in rows_sorted], dtype=float)
            colors = np.where(values >= 0, '#C44E52', '#8172B2')
            fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color=colors))
            fig.update_layout(
                title=f'Distance Effect Size - {model_type}',
                xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor='black')
            )
            fig.write_html(self.output_dir / f'distance_effect_{model_type}.html')

            # Distance difference plot (HTML)
            rows_sorted = sorted(rows, key=lambda x: abs(x[2]), reverse=True)[:top_k]
            labels = [r[0] for r in rows_sorted]
            values = np.asarray([r[2] for r in rows_sorted], dtype=float)
            colors = np.where(values <= 0, '#4C72B0', '#DD8452')
            vmax = float(np.nanmax(np.abs(values)))
            vmax = 1.0 if not np.isfinite(vmax) else max(0.1, vmax)
            fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color=colors))
            fig.update_layout(
                title=f'Distance Difference - {model_type}',
                xaxis=dict(range=[-vmax, vmax], zeroline=True, zerolinecolor='black'),
                xaxis_title='Distance Difference (same - different)'
            )
            fig.write_html(self.output_dir / f'distance_difference_{model_type}.html')

    def generate_report(self, clustering_results: Dict[str, Any],
                       fg_analysis: Dict[str, Any]) -> None:
        """Generate an experiment report."""
        logger.info("Generating experiment report")
        
        # Human-friendly model names (for display)
        def _model_display(name: str) -> str:
            if name == 'base':
                return 'Structure-centric (Base)'
            if name == 'delta':
                return 'Property-centric (Delta)'
            if name == 'semantic':
                return 'Semantic'
            return name.title()

        model_keys = list(clustering_results.keys())
        model_display = {k: _model_display(k) for k in model_keys}

        report = {
            'experiment_info': {
                'dataset_path': str(self.args.dataset_path),
                'sample_size': self.args.sample_size,
                'random_seed': self.args.random_seed,
                'models_used_raw': model_keys,
                'models_used': [model_display[k] for k in model_keys],
                'model_display': model_display,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'clustering_summary': {},
            'functional_group_summary': {},
            'distance_property_summary': {},
            'key_findings': []
        }
        
        # Clustering summary
        for model_type, results in clustering_results.items():
            metrics = results['metrics']
            report['clustering_summary'][model_type] = {
                'n_clusters': metrics.get('n_clusters', 0),
                'noise_ratio': metrics.get('noise_ratio', 0.0),
                'calinski_harabasz': metrics.get('calinski_harabasz', 0.0),
                'davies_bouldin': metrics.get('davies_bouldin', 0.0),
                'dbcv': metrics.get('dbcv', 0.0)
            }
        
        # Functional-group analysis summary + distance–property differential consistency summary
        for model_type, analysis in fg_analysis.items():
            linear_probe = analysis['linear_probe']
            
            # Compute average performance
            auroc_scores = [metrics['auroc'] for metrics in linear_probe.values()]
            auprc_scores = [metrics['auprc'] for metrics in linear_probe.values()]
            
            report['functional_group_summary'][model_type] = {
                'n_functional_groups_analyzed': len(linear_probe),
                'mean_auroc': np.mean(auroc_scores) if auroc_scores else 0.0,
                'std_auroc': np.std(auroc_scores) if auroc_scores else 0.0,
                'mean_auprc': np.mean(auprc_scores) if auprc_scores else 0.0,
                'std_auprc': np.std(auprc_scores) if auprc_scores else 0.0,
                'best_performing_fg': max(linear_probe.items(), 
                                        key=lambda x: x[1]['auroc'])[0] if linear_probe else None
            }

            # Distance–property differential consistency (if present, include in summary)
            dp = analysis.get('distance_property', {}) or {}
            dp_entry = {
                'spearman_rho': float(dp.get('spearman_rho', np.nan)) if dp.get('spearman_rho', None) is not None else float('nan'),
                'spearman_p': float(dp.get('spearman_p', np.nan)) if dp.get('spearman_p', None) is not None else float('nan'),
                'n_pairs': int(dp.get('n_pairs', 0)) if dp.get('n_pairs', None) is not None else 0,
                'distance_metric': dp.get('distance_metric', 'euclidean'),
                'space': dp.get('space', 'raw'),
                'n_components': int(dp.get('n_components')) if isinstance(dp.get('n_components', None), (int, float)) else None,
                'morans_i': float(dp.get('morans_i', np.nan)) if dp.get('morans_i', None) is not None else float('nan'),
                'gearys_c': float(dp.get('gearys_c', np.nan)) if dp.get('gearys_c', None) is not None else float('nan'),
                'dirichlet_energy': float(dp.get('dirichlet_energy', np.nan)) if dp.get('dirichlet_energy', None) is not None else float('nan'),
                'knn_rmse_mean': float(dp.get('knn_rmse_mean', np.nan)) if dp.get('knn_rmse_mean', None) is not None else float('nan'),
                'knn_r2_mean': float(dp.get('knn_r2_mean', np.nan)) if dp.get('knn_r2_mean', None) is not None else float('nan'),
                'knn_k': int(dp.get('knn_k', 0)) if dp.get('knn_k', None) is not None else 0,
            }
            report['distance_property_summary'][model_type] = dp_entry
        
        # Key findings
        if len(clustering_results) > 1:
            # Compare clustering quality across models
            best_clustering_model = max(clustering_results.items(), 
                                      key=lambda x: x[1]['metrics'].get('calinski_harabasz', 0))
            report['key_findings'].append(
                f"Best clustering quality achieved by {best_clustering_model[0]} model "
                f"(Calinski-Harabasz: {best_clustering_model[1]['metrics'].get('calinski_harabasz', 0):.3f})"
            )
            
            # Compare linear-probe performance
            model_aurocs = {}
            for model_type, analysis in fg_analysis.items():
                linear_probe = analysis['linear_probe']
                auroc_scores = [metrics['auroc'] for metrics in linear_probe.values()]
                model_aurocs[model_type] = np.mean(auroc_scores) if auroc_scores else 0.0
            
            if model_aurocs:
                best_probe_model = max(model_aurocs.items(), key=lambda x: x[1])
                report['key_findings'].append(
                    f"Best linear probe performance by {best_probe_model[0]} model "
                    f"(Mean AUROC: {best_probe_model[1]:.3f})"
                )
        
        # Save report
        report_path = self.output_dir / 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate a human-readable text report
        text_report = self._generate_text_report(report)
        text_report_path = self.output_dir / 'experiment_report.txt'
        with open(text_report_path, 'w') as f:
            f.write(text_report)
        
        logger.info(f"Reports saved to {report_path} and {text_report_path}")

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report."""
        lines = [
            "="*80,
            "FUNCTIONAL GROUP CLUSTERING EXPERIMENT REPORT",
            "="*80,
            "",
            f"Experiment Date: {report['experiment_info']['timestamp']}",
            f"Dataset: {report['experiment_info']['dataset_path']}",
            f"Sample Size: {report['experiment_info']['sample_size']}",
            f"Random Seed: {report['experiment_info']['random_seed']}",
            f"Models Analyzed: {', '.join(report['experiment_info'].get('models_used', []))}",
            "",
            "CLUSTERING RESULTS",
            "-"*40
        ]
        
        for model_type, metrics in report['clustering_summary'].items():
            disp = report['experiment_info'].get('model_display', {}).get(model_type, model_type.upper())
            lines.extend([
                f"\n{disp}:",
                f"  Number of Clusters: {metrics['n_clusters']}",
                f"  Noise Ratio: {metrics['noise_ratio']:.1%}",
                f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.3f}",
                f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}",
                f"  DBCV Score (approx): {metrics['dbcv']:.3f}"
            ])
        
        lines.extend([
            "",
            "FUNCTIONAL GROUP ANALYSIS",
            "-"*40
        ])
        
        for model_type, summary in report['functional_group_summary'].items():
            disp = report['experiment_info'].get('model_display', {}).get(model_type, model_type.upper())
            lines.extend([
                f"\n{disp}:",
                f"  Functional Groups Analyzed: {summary['n_functional_groups_analyzed']}",
                f"  Mean AUROC: {summary['mean_auroc']:.3f} ± {summary['std_auroc']:.3f}",
                f"  Mean AUPRC: {summary['mean_auprc']:.3f} ± {summary['std_auprc']:.3f}",
                f"  Best Performing FG: {summary['best_performing_fg']}"
            ])

        # Distance–property differential consistency (text summary)
        if report.get('distance_property_summary'):
            lines.extend([
                "",
                "DISTANCE–PROPERTY CONSISTENCY",
                "-"*40
            ])
            for model_type, dp in report['distance_property_summary'].items():
                disp = report['experiment_info'].get('model_display', {}).get(model_type, model_type.upper())
                rho = dp.get('spearman_rho', float('nan'))
                pval = dp.get('spearman_p', float('nan'))
                npairs = dp.get('n_pairs', 0)
                metric = dp.get('distance_metric', 'euclidean')
                space = dp.get('space', 'raw')
                knn_k = dp.get('knn_k', None)
                moran = dp.get('morans_i', float('nan'))
                geary = dp.get('gearys_c', float('nan'))
                de = dp.get('dirichlet_energy', float('nan'))
                rmse_mean = dp.get('knn_rmse_mean', float('nan'))
                r2_mean = dp.get('knn_r2_mean', float('nan'))
                # Normalize formatting for output
                try:
                    rho_str = f"{float(rho):.3f}" if np.isfinite(float(rho)) else "nan"
                except Exception:
                    rho_str = str(rho)
                try:
                    p_str = f"{float(pval):.3e}" if np.isfinite(float(pval)) else "nan"
                except Exception:
                    p_str = str(pval)
                knn_info = f", kNN k={int(knn_k)}" if knn_k is not None else ""
                lines.extend([
                    f"\n{disp}:",
                    f"  Spearman rho: {rho_str}",
                    f"  p-value: {p_str}",
                    f"  n_pairs: {int(npairs)} (metric={metric}, space={space}{knn_info})",
                    f"  Moran's I: {('nan' if not np.isfinite(float(moran)) else f'{float(moran):.3f}')}  |  Geary's C: {('nan' if not np.isfinite(float(geary)) else f'{float(geary):.3f}')}",
                    f"  Dirichlet energy: {('nan' if not np.isfinite(float(de)) else f'{float(de):.3e}')}  |  kNN RMSE: {('nan' if not np.isfinite(float(rmse_mean)) else f'{float(rmse_mean):.4f}')}  |  kNN R^2: {('nan' if not np.isfinite(float(r2_mean)) else f'{float(r2_mean):.3f}')}"
                ])
        
        if report['key_findings']:
            lines.extend([
                "",
                "KEY FINDINGS",
                "-"*40
            ])
            for i, finding in enumerate(report['key_findings'], 1):
                lines.append(f"{i}. {finding}")
        
        lines.extend([
            "",
            "="*80,
            "End of Report"
        ])
        
        return "\n".join(lines)

    def run_experiment(self) -> None:
        """Run the complete experiment."""
        logger.info("Starting functional group clustering experiment")
        
        try:
            # 1. Load models
            models = self.load_models()
            
            # 2. Load datasets
            datasets = self.load_datasets(models)
            
            if len(datasets) == 1:
                # Single-dataset mode (backward compatible)
                dataset_name, dataset = datasets[0]
                logger.info(f"Running single dataset experiment: {dataset_name}")
                # Optional: distance–property consistency only
                if getattr(self.args, 'only_distance_property', False):
                    self._run_distance_property_only(models, dataset_name, dataset)
                else:
                    self._run_single_dataset_experiment(models, dataset_name, dataset)
            else:
                # Multi-dataset mode
                logger.info(f"Running multi-dataset experiment with {len(datasets)} datasets")
                self._run_multi_dataset_experiment(models, datasets)
            
            logger.info("Experiment completed successfully")
            
        except Exception as e:
            logger.exception(f"Experiment failed: {e}")
            raise
    
    def _run_single_dataset_experiment(self, models: Dict[str, Any], dataset_name: str, dataset: MolecularDataset):
        """Run a single-dataset experiment (backward compatible)."""
        # Shortcut path to run distance–property consistency only (KISS)
        if getattr(self.args, 'only_distance_property', False):
            return self._run_distance_property_only(models, dataset_name, dataset)

        # 3. Extract molecular representations
        embeddings = self.extract_embeddings(models, dataset)
        # Comparison mode: force sample alignment (use intersection of base and delta indices)
        if 'base' in embeddings and 'delta' in embeddings:
            idx_base = self.sample_indices.get('base', [])
            idx_delta = self.sample_indices.get('delta', [])
            inter = sorted(set(idx_base) & set(idx_delta))
            if len(inter) == 0:
                raise ValueError("Paired compare requires non-empty intersection of samples between base and delta")
            # Build mapping from sample index to position (embedding rows follow valid_indices append order)
            pos_base = {idx: i for i, idx in enumerate(idx_base)}
            pos_delta = {idx: i for i, idx in enumerate(idx_delta)}
            sel_base = [pos_base[i] for i in inter if i in pos_base]
            sel_delta = [pos_delta[i] for i in inter if i in pos_delta]
            embeddings['base'] = embeddings['base'][sel_base]
            embeddings['delta'] = embeddings['delta'][sel_delta]
            self.sample_indices['base'] = inter
            self.sample_indices['delta'] = inter
            logger.info(f"Paired compare: base={len(idx_base)} delta={len(idx_delta)} -> intersection={len(inter)} samples")
        
        # 4. Clustering analysis
        clustering_results = self.run_clustering_analysis(embeddings)
        
        # 5. Functional-group analysis
        fg_analysis = self.run_functional_group_analysis(dataset, embeddings, clustering_results)
        
        # 6. Visualization
        self.create_visualizations(dataset, embeddings, clustering_results, fg_analysis)
        
        # 7. Generate report
        self.generate_report(clustering_results, fg_analysis)

        # 8. In comparison mode, generate base vs. delta difference summary
        if 'base' in clustering_results and 'delta' in clustering_results:
            try:
                self._generate_compare_report(clustering_results, fg_analysis)
            except Exception as e:
                logger.warning(f"Failed to generate compare report: {e}")

    def _run_distance_property_only(self, models: Dict[str, Any], dataset_name: str, dataset: MolecularDataset):
        """Run only distance–property consistency analysis (no clustering/probes/enrichment/visualization)."""
        logger.info("Running distance–property consistency only (KISS mode)")

        # 1) Extract embeddings
        embeddings = self.extract_embeddings(models, dataset)

        # Comparison mode: align base/delta samples
        if 'base' in embeddings and 'delta' in embeddings:
            idx_base = self.sample_indices.get('base', [])
            idx_delta = self.sample_indices.get('delta', [])
            inter = sorted(set(idx_base) & set(idx_delta))
            if len(inter) == 0:
                raise ValueError("Paired compare requires non-empty intersection of samples between base and delta")
            pos_base = {idx: i for i, idx in enumerate(idx_base)}
            pos_delta = {idx: i for i, idx in enumerate(idx_delta)}
            sel_base = [pos_base[i] for i in inter if i in pos_base]
            sel_delta = [pos_delta[i] for i in inter if i in pos_delta]
            embeddings['base'] = embeddings['base'][sel_base]
            embeddings['delta'] = embeddings['delta'][sel_delta]
            self.sample_indices['base'] = inter
            self.sample_indices['delta'] = inter
            logger.info(f"Paired compare aligned: intersection={len(inter)} samples")

        # 2) Optional PCA: used only when selecting PCA space (defaults match main pipeline)
        pca_dict: Dict[str, np.ndarray] = {}
        for model_type, emb in embeddings.items():
            try:
                n, d = emb.shape
                desired = int(getattr(self.args, 'pca_components', 30))
                n_comp = int(min(max(2, desired), n - 1, d))
                pca = PCA(n_components=n_comp, whiten=False, random_state=self.args.random_seed)
                z = pca.fit_transform(emb)
                pca_dict[model_type] = z
            except Exception as e:
                logger.warning(f"PCA failed for {model_type}: {e}")
                pca_dict[model_type] = emb

        # 3) Compute Spearman( D_ij, |Δp| )
        dp_summary: Dict[str, Any] = {}
        dp_space = getattr(self.args, 'distance_property_space', 'raw')
        for model_type, z in pca_dict.items():
            try:
                indices = self.sample_indices.get(model_type, list(range(len(z))))
                prop_values = self._extract_property_vector(dataset, indices)
                if prop_values is None:
                    dp = {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': getattr(self.args, 'distance_metric', 'cosine')}
                else:
                    mask_valid = ~np.isnan(prop_values)
                    if mask_valid.sum() < 2:
                        dp = {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': getattr(self.args, 'distance_metric', 'cosine')}
                    else:
                        src = embeddings[model_type] if dp_space == 'raw' else z
                        dp = self.stats.distance_property_consistency(
                            src[mask_valid], prop_values[mask_valid],
                            max_pairs=int(getattr(self.args, 'max_property_pairs', 50000)),
                            use_gpu=(self.device.type == 'cuda'),
                            distance_metric=getattr(self.args, 'distance_metric', 'cosine'),
                            seed=self.args.random_seed,
                            pre_transformed=(dp_space == 'pca'),
                            knn_k=int(getattr(self.args, 'knn_k', 15))
                        )
                        dp['space'] = dp_space
                        if dp_space == 'pca':
                            try:
                                dp['n_components'] = int(z.shape[1])
                            except Exception:
                                dp['n_components'] = None
                dp_summary[model_type] = dp
            except Exception as e:
                logger.warning(f"distance–property failed for {model_type}: {e}")
                dp_summary[model_type] = {'spearman_rho': float('nan'), 'spearman_p': float('nan'), 'n_pairs': 0, 'distance_metric': getattr(self.args, 'distance_metric', 'cosine')}

        # 4) Save compact report
        info = {
            'dataset': dataset_name,
            'sample_size': self.args.sample_size,
            'random_seed': self.args.random_seed,
            'distance_metric': getattr(self.args, 'distance_metric', 'cosine'),
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        out_json = self.output_dir / f"distance_property_only_{dataset_name}.json"
        with open(out_json, 'w') as f:
            json.dump({'experiment_info': info, 'distance_property_summary': dp_summary}, f, indent=2)
        # Text summary
        lines = [
            f"DISTANCE–PROPERTY CONSISTENCY ONLY ({dataset_name})",
            f"Metric: {info['distance_metric']}",
            f"Samples: {info['sample_size']}",
            f"Seed: {info['random_seed']}",
            ''
        ]
        for k, dp in dp_summary.items():
            rho = dp.get('spearman_rho', float('nan'))
            pval = dp.get('spearman_p', float('nan'))
            npairs = dp.get('n_pairs', 0)
            space = dp.get('space', 'raw')
            knn_k = dp.get('knn_k', None)
            moran = dp.get('morans_i', float('nan'))
            geary = dp.get('gearys_c', float('nan'))
            de = dp.get('dirichlet_energy', float('nan'))
            rmse_mean = dp.get('knn_rmse_mean', float('nan'))
            r2_mean = dp.get('knn_r2_mean', float('nan'))
            try:
                rho_str = f"{float(rho):.3f}" if np.isfinite(float(rho)) else "nan"
            except Exception:
                rho_str = str(rho)
            try:
                p_str = f"{float(pval):.3e}" if np.isfinite(float(pval)) else "nan"
            except Exception:
                p_str = str(pval)
            knn_info = f", kNN k={int(knn_k)}" if knn_k is not None else ""
            lines.extend([
                f"{k}: rho={rho_str}, p={p_str}, n_pairs={int(npairs)}, space={space}{knn_info}",
                f"    Moran's I={('nan' if not np.isfinite(float(moran)) else f'{float(moran):.3f}')}, Geary's C={('nan' if not np.isfinite(float(geary)) else f'{float(geary):.3f}')}",
                f"    Dirichlet energy={('nan' if not np.isfinite(float(de)) else f'{float(de):.3e}')}, kNN RMSE={('nan' if not np.isfinite(float(rmse_mean)) else f'{float(rmse_mean):.4f}')}, kNN R^2={('nan' if not np.isfinite(float(r2_mean)) else f'{float(r2_mean):.3f}')}"
            ])
        out_txt = self.output_dir / f"distance_property_only_{dataset_name}.txt"
        with open(out_txt, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Saved distance–property only report: {out_json} / {out_txt}")

        return {'distance_property_summary': dp_summary}
    
    def _run_multi_dataset_experiment(self, models: Dict[str, Any], datasets: List[Tuple[str, MolecularDataset]]):
        """Run a multi-dataset experiment."""
        all_results = {}
        # Fix root output directory to avoid nested directories in failure cases
        root_output_dir = self.output_dir
        
        for dataset_name, dataset in datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            dataset_output_dir = root_output_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)
            
            # Temporarily switch output directory to dataset subdirectory
            original_output_dir = self.output_dir
            self.output_dir = dataset_output_dir
            try:
                # Run single-dataset analysis (or distance–property-only analysis)
                if getattr(self.args, 'only_distance_property', False):
                    self._run_distance_property_only(models, dataset_name, dataset)
                else:
                    self._run_single_dataset_experiment(models, dataset_name, dataset)
                
                # Collect results for comparative analysis
                all_results[dataset_name] = {
                    'dataset_size': len(dataset),
                    'output_dir': dataset_output_dir
                }
                logger.info(f"Completed analysis for dataset: {dataset_name}")
            except Exception as e:
                logger.exception(f"Failed to process dataset {dataset_name}: {e}")
            finally:
                # Always restore root output directory
                self.output_dir = root_output_dir
        
        # Generate comparative report
        if len(all_results) > 1:
            logger.info("Generating comparative report across datasets")
            self._generate_comparative_report(all_results)
    
    def _generate_comparative_report(self, all_results: Dict[str, Dict]):
        """Generate a comparative report across datasets."""
        try:
            report_lines = [
                "# Multi-dataset functional-group clustering analysis report",
                "=" * 60,
                "",
                f"Experiment time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Number of datasets analyzed: {len(all_results)}",
                "",
                "## Dataset overview",
                ""
            ]
            
            # Dataset statistics
            for dataset_name, info in all_results.items():
                dataset_type = "classification" if dataset_name in CLASSIFICATION_DATASETS else "regression"
                report_lines.extend([
                    f"### {dataset_name} ({dataset_type} dataset)",
                    f"- Number of molecules: {info['dataset_size']}",
                    f"- Result directory: {info['output_dir']}",
                    ""
                ])
            
            # Try to read clustering quality metrics for each dataset
            report_lines.extend([
                "## Clustering quality comparison",
                "",
                "| Dataset | Type | #Molecules | #Clusters | Noise ratio | CH index | DB index |",
                "|--------|------|--------|--------|--------|--------|--------|"
            ])
            
            for dataset_name, info in all_results.items():
                dataset_type = "classification" if dataset_name in CLASSIFICATION_DATASETS else "regression"
                
                # Try to read experiment result files to get clustering metrics
                try:
                    # First try detailed result file
                    report_file = info['output_dir'] / "experiment_report.json"
                    if report_file.exists():
                        with open(report_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # Extract clustering quality metrics
                        clustering_summary = result_data.get('clustering_summary', {})
                        if 'semantic' in clustering_summary:
                            semantic_metrics = clustering_summary['semantic']
                            n_clusters = semantic_metrics.get('n_clusters', 0)
                            ch_index = semantic_metrics.get('calinski_harabasz', 0.0)
                            db_index = semantic_metrics.get('davies_bouldin', 0.0)
                            noise_ratio = semantic_metrics.get('noise_ratio', 0.0)
                            
                            report_lines.append(
                                f"| {dataset_name} | {dataset_type} | {info['dataset_size']} | "
                                f"{n_clusters} | {noise_ratio:.1%} | {ch_index:.3f} | {db_index:.3f} |"
                            )
                        else:
                            report_lines.append(f"| {dataset_name} | {dataset_type} | {info['dataset_size']} | - | - | - | - |")
                    else:
                        # If result file does not exist, check config file
                        config_file = info['output_dir'] / "experiment_config.json"
                        if config_file.exists():
                            report_lines.append(f"| {dataset_name} | {dataset_type} | {info['dataset_size']} | pending | pending | pending | pending |")
                        else:
                            report_lines.append(f"| {dataset_name} | {dataset_type} | {info['dataset_size']} | not run | not run | not run | not run |")
                        
                except Exception as e:
                    logger.warning(f"Failed to read metrics for {dataset_name}: {e}")
                    report_lines.append(f"| {dataset_name} | {dataset_type} | {info['dataset_size']} | Error | Error | Error | Error |")
            
            report_lines.extend([
                "",
                "## Notes",
                "",
                "- **#Clusters**: number of effective clusters detected by HDBSCAN (excluding noise).",
                "- **Noise ratio**: fraction of samples labeled as noise (lower is better, target <40%).",
                "- **CH index**: Calinski–Harabasz index; higher indicates better clustering.",
                "- **DB index**: Davies–Bouldin index; lower indicates better clustering (<1.0 ideal).",
                "",
                "### Dataset characteristics:",
                f"- **Classification datasets**: {', '.join(sorted(CLASSIFICATION_DATASETS & set(all_results.keys())))}",
                f"- **Regression datasets**: {', '.join(sorted(REGRESSION_DATASETS & set(all_results.keys())))}",
                "",
                "### Detailed results",
                "See each dataset's subdirectory for detailed analysis results.",
                "",
                "---",
                f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
            # Save comparative report
            comparative_report_file = self.output_dir / "comparative_analysis_report.md"
            with open(comparative_report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Comparative report saved to: {comparative_report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparative report: {e}")
            import traceback
            traceback.print_exc()

    def _generate_compare_report(self, clustering_results: Dict[str, Any], fg_analysis: Dict[str, Any]):
        """Generate a base vs. delta difference summary in comparison mode (JSON + text)."""
        base_key, delta_key = 'base', 'delta'
        if base_key not in clustering_results or delta_key not in clustering_results:
            return

        # Sample statistics
        base_count = len(self.sample_indices.get(base_key, []))
        delta_count = len(self.sample_indices.get(delta_key, []))
        inter_count = len(set(self.sample_indices.get(base_key, [])) & set(self.sample_indices.get(delta_key, []))) if base_count and delta_count else 0

        def _cluster_summary(res: Dict[str, Any]) -> Dict[str, float]:
            m = res.get('metrics', {})
            return {
                'n_clusters': float(m.get('n_clusters', 0)),
                'noise_ratio': float(m.get('noise_ratio', 0.0)),
                'calinski_harabasz': float(m.get('calinski_harabasz', 0.0)),
                'davies_bouldin': float(m.get('davies_bouldin', 0.0)),
                'silhouette': float(m.get('silhouette', 0.0)),
            }

        def _probe_summary(ana: Dict[str, Any]) -> Dict[str, float]:
            lp = ana.get('linear_probe', {})
            aurocs = [v.get('auroc', 0.0) for v in lp.values()]
            auprc = [v.get('auprc', 0.0) for v in lp.values()]
            return {
                'n_functional_groups': int(len(lp)),
                'mean_auroc': float(np.mean(aurocs) if aurocs else 0.0),
                'std_auroc': float(np.std(aurocs) if aurocs else 0.0),
                'mean_auprc': float(np.mean(auprc) if auprc else 0.0),
                'std_auprc': float(np.std(auprc) if auprc else 0.0),
            }

        def _distance_summary(ana: Dict[str, Any]) -> Dict[str, float]:
            da = ana.get('distance_analysis', {})
            deltas, diffs, n_sig = [], [], 0
            for v in da.values():
                if not isinstance(v, dict):
                    continue
                p = float(v.get('p_fdr', v.get('p_value', 1.0)))
                if p < 0.05:
                    n_sig += 1
                    cd = v.get('cliff_delta', None)
                    if cd is not None and np.isfinite(cd):
                        deltas.append(float(cd))
                    sdm = v.get('same_distance_mean', None)
                    ddm = v.get('diff_distance_mean', None)
                    if sdm is not None and ddm is not None and np.isfinite(sdm) and np.isfinite(ddm):
                        diffs.append(float(sdm) - float(ddm))
            return {
                'n_significant': int(n_sig),
                'mean_cliffs_delta_sig': float(np.mean(deltas) if deltas else 0.0),
                'mean_distance_diff_sig': float(np.mean(diffs) if diffs else 0.0),
            }

        def _enrichment_summary(ana: Dict[str, Any]) -> Dict[str, float]:
            summ = ana.get('enrichment', {}).get('summary', {})
            return {
                'n_tests': float(summ.get('total_tests', 0)),
                'n_significant': float(summ.get('significant_tests', 0)),
            }

        def _diff(d2: Dict[str, float], d1: Dict[str, float]) -> Dict[str, float]:
            keys = set(d1.keys()) | set(d2.keys())
            return {k: float(d2.get(k, 0.0)) - float(d1.get(k, 0.0)) for k in keys}

        def _distance_property_summary(ana: Dict[str, Any]) -> Dict[str, float]:
            dp = ana.get('distance_property', {}) or {}
            res = {}
            try:
                res['spearman_rho'] = float(dp.get('spearman_rho', 0.0))
            except Exception:
                res['spearman_rho'] = 0.0
            try:
                res['n_pairs'] = float(dp.get('n_pairs', 0))
            except Exception:
                res['n_pairs'] = 0.0
            return res

        base_cluster = _cluster_summary(clustering_results[base_key])
        delta_cluster = _cluster_summary(clustering_results[delta_key])
        base_probe = _probe_summary(fg_analysis.get(base_key, {}))
        delta_probe = _probe_summary(fg_analysis.get(delta_key, {}))
        base_dist = _distance_summary(fg_analysis.get(base_key, {}))
        delta_dist = _distance_summary(fg_analysis.get(delta_key, {}))
        base_enr = _enrichment_summary(fg_analysis.get(base_key, {}))
        delta_enr = _enrichment_summary(fg_analysis.get(delta_key, {}))
        base_dp = _distance_property_summary(fg_analysis.get(base_key, {}))
        delta_dp = _distance_property_summary(fg_analysis.get(delta_key, {}))

        compare = {
            'models': ['base', 'delta'],
            'samples': {'base_count': base_count, 'delta_count': delta_count, 'intersection_count': inter_count},
            'clustering': {'base': base_cluster, 'delta': delta_cluster, 'diff': _diff(delta_cluster, base_cluster)},
            'linear_probe': {'base': base_probe, 'delta': delta_probe, 'diff': _diff(delta_probe, base_probe)},
            'distance': {'base': base_dist, 'delta': delta_dist, 'diff': _diff(delta_dist, base_dist)},
            'enrichment': {'base': base_enr, 'delta': delta_enr, 'diff': _diff(delta_enr, base_enr)},
            'distance_property': {'base': base_dp, 'delta': delta_dp, 'diff': _diff(delta_dp, base_dp)},
        }

        # JSON
        path_json = Path(self.output_dir) / 'compare_report.json'
        with open(path_json, 'w') as f:
            json.dump(compare, f, indent=2)

        # Text summary
        def fmt(v: float) -> str:
            try:
                return f"{float(v):.3f}"
            except Exception:
                return str(v)

        lines = [
            'COMPARE REPORT (base → delta)',
            '-' * 60,
            'Clustering:',
            f"  CH: {fmt(base_cluster.get('calinski_harabasz',0))} → {fmt(delta_cluster.get('calinski_harabasz',0))} (Δ {fmt(compare['clustering']['diff'].get('calinski_harabasz',0))})",
            f"  DB: {fmt(base_cluster.get('davies_bouldin',0))} → {fmt(delta_cluster.get('davies_bouldin',0))} (Δ {fmt(compare['clustering']['diff'].get('davies_bouldin',0))})",
            f"  Sil: {fmt(base_cluster.get('silhouette',0))} → {fmt(delta_cluster.get('silhouette',0))} (Δ {fmt(compare['clustering']['diff'].get('silhouette',0))})",
            f"  Clusters: {int(base_cluster.get('n_clusters',0))} → {int(delta_cluster.get('n_clusters',0))} (Δ {int(compare['clustering']['diff'].get('n_clusters',0))})",
            '',
            'Linear Probe:',
            f"  Mean AUROC: {fmt(base_probe.get('mean_auroc',0))} → {fmt(delta_probe.get('mean_auroc',0))} (Δ {fmt(compare['linear_probe']['diff'].get('mean_auroc',0))})",
            f"  Mean AUPRC: {fmt(base_probe.get('mean_auprc',0))} → {fmt(delta_probe.get('mean_auprc',0))} (Δ {fmt(compare['linear_probe']['diff'].get('mean_auprc',0))})",
            '',
            'Distance (significant):',
            f"  n_sig FG: {int(base_dist.get('n_significant',0))} → {int(delta_dist.get('n_significant',0))}",
            f"  Mean CliffΔ: {fmt(base_dist.get('mean_cliffs_delta_sig',0))} → {fmt(delta_dist.get('mean_cliffs_delta_sig',0))} (Δ {fmt(compare['distance']['diff'].get('mean_cliffs_delta_sig',0))})",
            f"  Mean (same-diff): {fmt(base_dist.get('mean_distance_diff_sig',0))} → {fmt(delta_dist.get('mean_distance_diff_sig',0))} (Δ {fmt(compare['distance']['diff'].get('mean_distance_diff_sig',0))})",
            '',
            'Distance–Property:',
            f"  Spearman rho: {fmt(base_dp.get('spearman_rho',0))} → {fmt(delta_dp.get('spearman_rho',0))} (Δ {fmt(compare['distance_property']['diff'].get('spearman_rho',0))})",
            '',
            'Enrichment:',
            f"  n_sig pairs: {int(base_enr.get('n_significant',0))} → {int(delta_enr.get('n_significant',0))} (total base {int(base_enr.get('n_tests',0))}, delta {int(delta_enr.get('n_tests',0))})",
        ]
        path_txt = Path(self.output_dir) / 'compare_report.txt'
        with open(path_txt, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Compare report saved: {path_json} / {path_txt}")


def main():
    """CLI entrypoint for the functional-group clustering experiment."""
    parser = argparse.ArgumentParser(
        description="Functional-group clustering experiment for hypergraph semantic masked autoencoder representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 1. Single custom dataset (backward compatible)
  python functional_group_clustering_experiment.py \\
    --semantic_model_path /path/to/model.pth \\
    --dataset_path /path/to/custom_data/ \\
    --output_dir ./results
  
  # 2. Single standard dataset
  python functional_group_clustering_experiment.py \\
    --semantic_model_path /path/to/model.pth \\
    --data_root /path/to/hypergraph_data_root \\
    --datasets BBBP \\
    --output_dir ./results
    
  # 3. Multiple datasets (batch analysis)
  python functional_group_clustering_experiment.py \\
    --semantic_model_path /path/to/model.pth \\
    --data_root /path/to/hypergraph_data_root \\
    --datasets BBBP BACE HIV SIDER \\
    --output_dir ./results
  
  # Using preprocessed hypergraph data
  python functional_group_clustering_experiment.py \\
    --semantic_model_path /path/to/model.pth \\
    --dataset_path /path/to/hypergraph_data/ \\
    --use_hypergraph_data \\
    --clustering_method hdbscan
        """
    )
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset configuration (choose one mode)')
    dataset_group.add_argument('--dataset_path', type=str,
                              help='Path to a single dataset (backward compatible mode)')
    dataset_group.add_argument('--data_root', type=str,
                              help='Root directory containing datasets (multi-dataset mode)')
    dataset_group.add_argument('--datasets', type=str, nargs='+', 
                              choices=list(ALL_DATASETS),
                              help=f'Names of datasets to analyze (used with --data_root, options: {", ".join(sorted(ALL_DATASETS))})')
    
    # Model path arguments (at least one required)
    model_group = parser.add_argument_group('Model paths (provide at least one)')
    # Comparison mode: pass two paths to --semantic_model_path (order: pre-train -> post-train)
    model_group.add_argument('--semantic_model_path', type=str, nargs='+',
                           help='Semantic masked model checkpoint path(s) (1=single model; 2=comparison mode, order: pre->post)')
    model_group.add_argument('--semantic_model_config', type=str, nargs='+',
                           help='Semantic model config path(s) (must match number of checkpoint paths)')
    model_group.add_argument('--random_model_path', type=str, 
                           help='Random-mask model checkpoint path')
    model_group.add_argument('--random_model_config', type=str,
                           help='Random-mask model config path (optional)')
    model_group.add_argument('--uniform_model_path', type=str,
                           help='Uniform-mask model checkpoint path')
    model_group.add_argument('--uniform_model_config', type=str,
                           help='Uniform-mask model config path (optional)')
    
    # Data-processing arguments
    data_group = parser.add_argument_group('Data processing parameters')
    data_group.add_argument('--sample_size', type=int, default=5000,
                          help='Number of molecules to sample (default: 5000)')
    data_group.add_argument('--use_hypergraph_data', action='store_true',
                          help='Force use of preprocessed hypergraph data (auto-detected otherwise)')
    data_group.add_argument('--stratified_sampling', action='store_true', default=True,
                          help='Use stratified sampling (default: on)')
    
    # Method arguments
    method_group = parser.add_argument_group('Method parameters')
    method_group.add_argument('--clustering_method', type=str, default='hdbscan',
                            choices=['hdbscan', 'kmeans'],
                            help='Clustering method (default: hdbscan)')
    method_group.add_argument('--pca_components', type=int, default=30,
                            help='Number of PCA components for dimensionality reduction (default: 30)')
    method_group.add_argument('--kmeans_k', type=int, default=None,
                            help='Number of KMeans clusters (default: auto-select best from 6/8/10/12/15)')
    method_group.add_argument('--grid_search_kmeans', action='store_true',
                            help='Enable grid search over KMeans (pca_components × k) and select best')
    method_group.add_argument('--kmeans_k_candidates', type=int, nargs='+', default=[6, 8, 10, 12, 15],
                            help='Candidate K values for KMeans (used with --grid_search_kmeans)')
    method_group.add_argument('--pca_candidates', type=int, nargs='+', default=[20, 30, 40],
                            help='Candidate PCA component counts (used with --grid_search_kmeans)')
    method_group.add_argument('--min_functional_group_samples', type=int, default=10,
                            help='Minimum samples per functional group for linear probe analysis (default: 10)')
    method_group.add_argument('--distance_metric', type=str, default='cosine',
                            choices=['cosine', 'euclidean', 'euclidean_zscore', 'mahalanobis', 'angular', 'correlation'],
                            help='Distance metric (default: cosine). mahala=global Mahalanobis via PCA+whitening+Euclidean; angular=arccos(cos); correlation=1-corr.')
    method_group.add_argument('--distance_property_space', type=str, default='raw',
                            choices=['raw', 'pca'],
                            help='Space for distance–property analysis: raw=original embeddings (default), pca=PCA space')
    method_group.add_argument('--knn_k', type=int, default=15,
                            help="kNN neighbor count k for Moran's I / Geary's C / Dirichlet energy / neighborhood consistency (default: 15)")
    method_group.add_argument('--only_distance_property', action='store_true',
                            help='Run only distance–property analysis (skip clustering/probe/enrichment/visualization)')
    
    # Output arguments
    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--output_dir', type=str, default='results',
                            help='Result output directory (default: results)')
    output_group.add_argument('--save_embeddings', action='store_true',
                            help='Save extracted molecular embeddings')
    output_group.add_argument('--save_clustering_data', action='store_true',
                            help='Save clustering result data')
    output_group.add_argument('--generate_html_report', action='store_true', default=True,
                            help='Generate HTML interactive reports (default: on)')
    
    # Experiment control arguments
    control_group = parser.add_argument_group('Experiment control parameters')
    control_group.add_argument('--random_seed', type=int, default=42,
                             help='Random seed (default: 42)')
    control_group.add_argument('--batch_size', type=int, default=32,
                             help='Batch size (default: 32)')
    control_group.add_argument('--num_workers', type=int, default=4,
                             help='Number of data-loader workers (default: 4)')
    control_group.add_argument('--device', type=str, default=None,
                             help='Device (cuda/cpu); default: auto-detect')
    control_group.add_argument('--seeds', type=int, nargs='+', default=None,
                             help='Multiple random seeds (e.g. --seeds 42 43 44) to enable multi-seed aggregation')
    
    # Visualization arguments
    viz_group = parser.add_argument_group('Visualization parameters')
    viz_group.add_argument('--umap_n_neighbors', type=int, default=15,
                         help='UMAP neighbor count (default: 15)')
    viz_group.add_argument('--plot_style', type=str, default='seaborn',
                         choices=['seaborn', 'ggplot', 'default'],
                         help='Matplotlib/Seaborn plot style (default: seaborn)')
    viz_group.add_argument('--figure_dpi', type=int, default=300,
                         help='Figure DPI (default: 300)')
    
    # Debugging arguments
    debug_group = parser.add_argument_group('Debug parameters')
    debug_group.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output mode')
    debug_group.add_argument('--dry_run', action='store_true',
                           help='Dry-run mode (only validate arguments and data loading)')
    debug_group.add_argument('--skip_visualization', action='store_true',
                           help='Skip visualization steps (faster runs)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure at least one model path is provided
    model_paths = [args.semantic_model_path, args.random_model_path, args.uniform_model_path]
    valid_model_paths = []
    for path in model_paths:
        if path is None:
            continue
        if isinstance(path, list):
            valid_model_paths.extend([p for p in path if p is not None])
        else:
            valid_model_paths.append(path)
    
    if not valid_model_paths:
        logger.error("At least one model path is required")
        parser.print_help()
        return
    
    # Validate dataset parameter configuration
    if not args.dataset_path and not (args.data_root and args.datasets):
        logger.error("You must specify a dataset: use --dataset_path or --data_root + --datasets")
        parser.print_help()
        return
    
    if args.dataset_path and (args.data_root or args.datasets):
        logger.error("--dataset_path and --data_root/--datasets cannot be used together")
        parser.print_help()
        return
    
    if args.data_root and not args.datasets:
        logger.error("When using --data_root you must also specify --datasets")
        parser.print_help()
        return
    
    # Validate existence of input paths
    paths_to_check = []
    # Semantic model path/config validation: counts must match (1 or 2). No complex inference; mismatch is an error.
    if args.semantic_model_path:
        if isinstance(args.semantic_model_path, list):
            for i, p in enumerate(args.semantic_model_path):
                paths_to_check.append((f'semantic_model_path[{i}]', p))
        else:
            paths_to_check.append(('semantic_model_path', args.semantic_model_path))
    if getattr(args, 'semantic_model_config', None):
        if isinstance(args.semantic_model_config, list):
            for i, p in enumerate(args.semantic_model_config):
                paths_to_check.append((f'semantic_model_config[{i}]', p))
        else:
            paths_to_check.append(('semantic_model_config', args.semantic_model_config))
    if args.random_model_path:
        paths_to_check.append(('random_model_path', args.random_model_path))
    if getattr(args, 'random_model_config', None):
        paths_to_check.append(('random_model_config', args.random_model_config))
    if args.uniform_model_path:
        paths_to_check.append(('uniform_model_path', args.uniform_model_path))
    if getattr(args, 'uniform_model_config', None):
        paths_to_check.append(('uniform_model_config', args.uniform_model_config))
    
    # Validate dataset paths
    if args.dataset_path:
        paths_to_check.append(('dataset_path', args.dataset_path))
    elif args.data_root:
        paths_to_check.append(('data_root', args.data_root))

    # First check general paths (not per-dataset subdirectories)
    
    for param_name, path in paths_to_check:
        if not Path(path).exists():
            logger.error(f"{param_name} path does not exist: {path}")
            return

    # With data_root + datasets, check each dataset name (with alias support)
    if args.data_root and args.datasets:
        root = Path(args.data_root)
        for raw_name in args.datasets:
            canonical = DATASET_ALIASES.get(raw_name, raw_name)
            aliases = CANONICAL_TO_ALIASES.get(canonical, [])
            candidates = []
            # Priority: raw name -> canonical name -> reverse aliases
            for name in [raw_name, canonical] + aliases:
                p = root / name
                candidates.append(p)
                if p.exists():
                    break
            else:
                tried = ', '.join(str(p) for p in candidates)
                logger.error(f"dataset_{raw_name} path does not exist; tried: {tried}")
                return
    
    # Create timestamped subdirectory to distinguish runs; append dataset name and distance metric
    output_dir = Path(args.output_dir)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    # Infer dataset label (single dataset uses path name; multi-dataset uses name list)
    try:
        if args.dataset_path:
            dp = Path(args.dataset_path)
            dataset_label = dp.name if dp.is_dir() else dp.stem
        elif args.datasets:
            dataset_label = '-'.join(args.datasets)
        else:
            dataset_label = 'data'
    except Exception:
        dataset_label = 'data'
    # Simple cleanup: avoid path separators and spaces
    dataset_label = dataset_label.replace(os.sep, '_').replace(' ', '_')
    metric_label = getattr(args, 'distance_metric', 'cosine')
    output_dir = output_dir / f"{ts}_{dataset_label}_{metric_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info(f"Experiment configuration saved to: {config_path}")
    # Write timestamped subdirectory back into args so downstream components use the same path
    args.output_dir = str(output_dir)
    
    # Dry-run mode: only validate arguments
    if args.dry_run:
        logger.info("Dry run: argument validation passed")
        logger.info(f"Will use {len(valid_model_paths)} model path(s)")
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Output directory: {args.output_dir}")
        return

    # Multi-seed aggregation mode
    seeds = args.seeds if args.seeds and len(args.seeds) >= 1 else None
    if seeds is not None and len(seeds) >= 1:
        used = []
        for s in seeds:
            seed_dir = output_dir / f"seed_{int(s):04d}"
            seed_dir.mkdir(exist_ok=True)
            args.random_seed = int(s)
            args.output_dir = str(seed_dir)
            try:
                runner = ExperimentRunner(args)
                runner.run_experiment()
                used.append(int(s))
            except Exception as e:
                logger.error(f"Seed {s} failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        # Seed aggregation: summarize compare_report across seeds
        try:
            _aggregate_seed_compare_reports(output_dir, used)
            logger.info(f"Aggregated report saved: {output_dir / 'compare_report_agg.json'} / {output_dir / 'compare_report_agg.txt'}")
        except Exception as e:
            logger.error(f"Failed to generate multi-seed aggregated report: {e}")
        return
    
    # Single-seed standard mode
    try:
        runner = ExperimentRunner(args)
        runner.run_experiment()
        logger.info("🎉 Experiment completed! See results:")
        logger.info(f"   📊 Report: {output_dir / 'experiment_report.txt'}")
        if args.generate_html_report:
            logger.info(f"   🌐 Interactive plots: {output_dir}/*.html")
        logger.info(f"   📈 Figures: {output_dir}/*.png")
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return

def _aggregate_seed_compare_reports(root_dir: Path, seeds: List[int]):
    """Scan seed_* subdirectories and aggregate compare_report.json files across seeds."""
    import math
    recs = []
    for s in seeds:
        seed_dir = root_dir / f"seed_{int(s):04d}"
        path = seed_dir / 'compare_report.json'
        if path.exists():
            with open(path, 'r') as f:
                comp = json.load(f)
            # Backfill distance_property (for older runs that did not write it into compare_report)
            try:
                if 'distance_property' not in comp or not comp['distance_property']:
                    exp_path = seed_dir / 'experiment_report.json'
                    if exp_path.exists():
                        with open(exp_path, 'r') as ef:
                            exp = json.load(ef)
                        dps = exp.get('distance_property_summary', {}) or {}
                        base_dp = dps.get('base', {})
                        delta_dp = dps.get('delta', {})
                        if base_dp or delta_dp:
                            def _flt(x, default=0.0):
                                try:
                                    return float(x)
                                except Exception:
                                    return default
                            comp['distance_property'] = {
                                'base': {
                                    'spearman_rho': _flt(base_dp.get('spearman_rho', 0.0)),
                                    'n_pairs': _flt(base_dp.get('n_pairs', 0.0)),
                                },
                                'delta': {
                                    'spearman_rho': _flt(delta_dp.get('spearman_rho', 0.0)),
                                    'n_pairs': _flt(delta_dp.get('n_pairs', 0.0)),
                                },
                                'diff': {
                                    'spearman_rho': _flt(delta_dp.get('spearman_rho', 0.0)) - _flt(base_dp.get('spearman_rho', 0.0)),
                                    'n_pairs': _flt(delta_dp.get('n_pairs', 0.0)) - _flt(base_dp.get('n_pairs', 0.0)),
                                }
                            }
            except Exception:
                pass
            recs.append((int(s), comp))
    if not recs:
        raise ValueError("No compare_report.json files found; cannot aggregate")

    def mean_std(vals):
        if not vals:
            return {'mean': 0.0, 'std': 0.0}
        a = np.asarray(vals, dtype=float)
        return {'mean': float(np.mean(a)), 'std': float(np.std(a, ddof=0))}

    keys_cluster = ['calinski_harabasz', 'davies_bouldin', 'silhouette', 'n_clusters', 'noise_ratio']
    keys_probe = ['mean_auroc', 'mean_auprc']
    keys_dist = ['n_significant', 'mean_cliffs_delta_sig', 'mean_distance_diff_sig']
    keys_enr = ['n_tests', 'n_significant']
    keys_dp = ['spearman_rho', 'n_pairs']

    agg = {'n_seeds': len(recs), 'seeds': [s for s, _ in recs], 'samples': {}}

    # Sample statistics (if available)
    base_counts = []
    delta_counts = []
    inter_counts = []
    for _, r in recs:
        smp = r.get('samples', {})
        if smp:
            base_counts.append(smp.get('base_count', 0))
            delta_counts.append(smp.get('delta_count', 0))
            inter_counts.append(smp.get('intersection_count', 0))
    if base_counts:
        agg['samples']['base_count'] = mean_std(base_counts)
        agg['samples']['delta_count'] = mean_std(delta_counts)
        agg['samples']['intersection_count'] = mean_std(inter_counts)

    def collect(section: str, sub: str, key: str):
        vals = []
        for _, r in recs:
            try:
                vals.append(float(r[section][sub].get(key, 0)))
            except Exception:
                vals.append(0.0)
        return mean_std(vals)

    def collect_diff(section: str, key: str):
        vals = []
        for _, r in recs:
            try:
                vals.append(float(r[section]['diff'].get(key, 0)))
            except Exception:
                vals.append(0.0)
        return mean_std(vals)

    # Clustering
    agg['clustering'] = {'base': {}, 'delta': {}, 'diff': {}}
    for k in keys_cluster:
        agg['clustering']['base'][k] = collect('clustering', 'base', k)
        agg['clustering']['delta'][k] = collect('clustering', 'delta', k)
        agg['clustering']['diff'][k] = collect_diff('clustering', k)

    # Linear probe
    agg['linear_probe'] = {'base': {}, 'delta': {}, 'diff': {}}
    for k in keys_probe:
        agg['linear_probe']['base'][k] = collect('linear_probe', 'base', k)
        agg['linear_probe']['delta'][k] = collect('linear_probe', 'delta', k)
        agg['linear_probe']['diff'][k] = collect_diff('linear_probe', k)

    # Distance
    agg['distance'] = {'base': {}, 'delta': {}, 'diff': {}}
    for k in keys_dist:
        agg['distance']['base'][k] = collect('distance', 'base', k)
        agg['distance']['delta'][k] = collect('distance', 'delta', k)
        agg['distance']['diff'][k] = collect_diff('distance', k)

    # Enrichment
    agg['enrichment'] = {'base': {}, 'delta': {}, 'diff': {}}
    for k in keys_enr:
        agg['enrichment']['base'][k] = collect('enrichment', 'base', k)
        agg['enrichment']['delta'][k] = collect('enrichment', 'delta', k)
        agg['enrichment']['diff'][k] = collect_diff('enrichment', k)

    # Distance–Property
    agg['distance_property'] = {'base': {}, 'delta': {}, 'diff': {}}
    for k in keys_dp:
        agg['distance_property']['base'][k] = collect('distance_property', 'base', k)
        agg['distance_property']['delta'][k] = collect('distance_property', 'delta', k)
        agg['distance_property']['diff'][k] = collect_diff('distance_property', k)

    # Write JSON
    with open(root_dir / 'compare_report_agg.json', 'w') as f:
        json.dump(agg, f, indent=2)

    # Write text
    def fmt(ms):
        return f"{ms['mean']:.3f}±{ms['std']:.3f}"
    lines = [
        'COMPARE REPORT (aggregated over seeds)',
        '-' * 60,
        f"Seeds: {', '.join(str(s) for s, _ in recs)}",
        'Clustering:',
        f"  CH: {fmt(agg['clustering']['base']['calinski_harabasz'])} → {fmt(agg['clustering']['delta']['calinski_harabasz'])} (Δ {fmt(agg['clustering']['diff']['calinski_harabasz'])})",
        f"  DB: {fmt(agg['clustering']['base']['davies_bouldin'])} → {fmt(agg['clustering']['delta']['davies_bouldin'])} (Δ {fmt(agg['clustering']['diff']['davies_bouldin'])})",
        f"  Sil: {fmt(agg['clustering']['base']['silhouette'])} → {fmt(agg['clustering']['delta']['silhouette'])} (Δ {fmt(agg['clustering']['diff']['silhouette'])})",
        '',
        'Linear Probe:',
        f"  Mean AUROC: {fmt(agg['linear_probe']['base']['mean_auroc'])} → {fmt(agg['linear_probe']['delta']['mean_auroc'])} (Δ {fmt(agg['linear_probe']['diff']['mean_auroc'])})",
        f"  Mean AUPRC: {fmt(agg['linear_probe']['base']['mean_auprc'])} → {fmt(agg['linear_probe']['delta']['mean_auprc'])} (Δ {fmt(agg['linear_probe']['diff']['mean_auprc'])})",
        '',
        'Distance (significant):',
        f"  n_sig FG: {fmt(agg['distance']['base']['n_significant'])} → {fmt(agg['distance']['delta']['n_significant'])} (Δ {fmt(agg['distance']['diff']['n_significant'])})",
        f"  Mean CliffΔ: {fmt(agg['distance']['base']['mean_cliffs_delta_sig'])} → {fmt(agg['distance']['delta']['mean_cliffs_delta_sig'])} (Δ {fmt(agg['distance']['diff']['mean_cliffs_delta_sig'])})",
        f"  Mean (same-diff): {fmt(agg['distance']['base']['mean_distance_diff_sig'])} → {fmt(agg['distance']['delta']['mean_distance_diff_sig'])} (Δ {fmt(agg['distance']['diff']['mean_distance_diff_sig'])})",
        '',
        'Enrichment:',
        f"  n_sig pairs: {fmt(agg['enrichment']['base']['n_significant'])} → {fmt(agg['enrichment']['delta']['n_significant'])} (Δ {fmt(agg['enrichment']['diff']['n_significant'])})",
        '',
        'Distance–Property:',
        f"  Spearman rho: {fmt(agg['distance_property']['base']['spearman_rho'])} → {fmt(agg['distance_property']['delta']['spearman_rho'])} (Δ {fmt(agg['distance_property']['diff']['spearman_rho'])})",
    ]
    with open(root_dir / 'compare_report_agg.txt', 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
