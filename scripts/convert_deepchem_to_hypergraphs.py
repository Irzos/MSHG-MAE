#!/usr/bin/env python
"""
Convert DeepChem MolNet datasets (BBBP, BACE, SIDER, ESOL, Tox21, ClinTox, HIV, FreeSolv, Lipophilicity, ToxCast)
into HyperGraph-MAE compatible hypergraph files (batch_*.pt).

Output layout per dataset (under --output_root):
  <output_root>/BBBP/batch_0000.pt, batch_0001.pt, ...
  <output_root>/BACE/...
  <output_root>/SIDER/...
  <output_root>/ESOL/...
  <output_root>/Tox21/...
  <output_root>/ClinTox/...

Each saved file contains a Python list of torch_geometric.data.Data objects,
with fields at least: x, hyperedge_index, hyperedge_attr, y (1-D vector), y_mask (1-D bool vector for multi-task),
smiles, and metadata fields required by the training pipeline.
"""

import argparse
import logging
import math
import os
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.data.molecule_standardizer import MoleculeStandardizer
from src.data.preprocessing import DataStatistics
from src.data.hypergraph_construction import smiles_to_hypergraph, TimeoutException


def _serialize_data_for_ipc(data):
    """Convert torch_geometric.data.Data to a pure-Python dict (numpy arrays, ints, strs)."""
    try:
        payload = {}
        # Known tensor fields (masks removed - handled dynamically during training)
        tensor_fields = [
            'x', 'hyperedge_index', 'hyperedge_attr', 'edge_index', 'edge_attr',
            'hyperedge_type', 'rdkit_idx'
        ]
        for field in tensor_fields:
            if hasattr(data, field) and getattr(data, field) is not None:
                tensor = getattr(data, field)
                payload[field] = tensor.detach().cpu().numpy()
        # Scalars / metadata
        if hasattr(data, 'num_nodes'):
            payload['num_nodes'] = int(getattr(data, 'num_nodes'))
        if hasattr(data, 'hyperedge_dim'):
            payload['hyperedge_dim'] = int(getattr(data, 'hyperedge_dim'))
        if hasattr(data, 'num_hyperedges'):
            payload['num_hyperedges'] = int(getattr(data, 'num_hyperedges'))
        if hasattr(data, 'num_edges'):
            payload['num_edges'] = int(getattr(data, 'num_edges'))
        if hasattr(data, 'edge_dim'):
            payload['edge_dim'] = int(getattr(data, 'edge_dim'))
        if hasattr(data, 'meta_info'):
            payload['meta_info'] = getattr(data, 'meta_info')
        if hasattr(data, 'id'):
            payload['id'] = getattr(data, 'id')
        if hasattr(data, 'smiles'):
            payload['smiles'] = getattr(data, 'smiles')
        return payload
    except Exception as e:
        return e


def _deserialize_data_from_ipc(payload):
    """Rebuild torch_geometric.data.Data from the pure-Python dict."""
    from torch_geometric.data import Data as GeoData
    try:
        # Rebuild tensors with appropriate dtypes
        def to_tensor(arr, default_dtype=None):
            import numpy as _np
            if arr is None:
                return None
            if arr.dtype == _np.bool_:
                return torch.tensor(arr, dtype=torch.bool)
            if _np.issubdtype(arr.dtype, _np.integer):
                return torch.tensor(arr, dtype=torch.long)
            if _np.issubdtype(arr.dtype, _np.floating):
                return torch.tensor(arr, dtype=torch.float32 if default_dtype is None else default_dtype)
            return torch.tensor(arr)

        x = to_tensor(payload.get('x'))
        hyperedge_index = to_tensor(payload.get('hyperedge_index'), default_dtype=torch.long)
        hyperedge_attr = to_tensor(payload.get('hyperedge_attr'), default_dtype=torch.float32)
        edge_index = to_tensor(payload.get('edge_index'), default_dtype=torch.long)
        edge_attr = to_tensor(payload.get('edge_attr'), default_dtype=torch.float32)
        hyperedge_type = to_tensor(payload.get('hyperedge_type'), default_dtype=torch.long)
        rdkit_idx = to_tensor(payload.get('rdkit_idx'), default_dtype=torch.long)

        data = GeoData(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=payload.get('num_nodes'),
        )
        # Optional fields (masks removed - handled dynamically during training)
        if 'hyperedge_dim' in payload:
            data.hyperedge_dim = payload['hyperedge_dim']
        if 'num_hyperedges' in payload and hyperedge_attr is not None:
            data.num_hyperedges = int(hyperedge_attr.shape[0])
        if 'num_edges' in payload:
            data.num_edges = payload['num_edges']
        if 'edge_dim' in payload:
            data.edge_dim = payload['edge_dim']
        if 'meta_info' in payload:
            data.meta_info = payload['meta_info']
        if 'id' in payload:
            data.id = payload['id']
        if 'smiles' in payload:
            data.smiles = payload['smiles']
        if hyperedge_type is not None:
            data.hyperedge_type = hyperedge_type
        if rdkit_idx is not None:
            data.rdkit_idx = rdkit_idx
        return data
    except Exception as e:
        return e


def _smiles_to_hypergraph_worker(args):
    """Worker function for multiprocessing - unpacks arguments and calls smiles_to_hypergraph"""
    smiles, mol_id, config, global_stats, device_str = args
    # Convert device string back to torch.device in subprocess
    device = torch.device(device_str)
    try:
        result = smiles_to_hypergraph(smiles, mol_id, config, global_stats, device)
        if result is None:
            return None
        # Serialize to pure-Python for safe IPC
        return _serialize_data_for_ipc(result)
    except Exception as e:
        # Return the exception instead of raising it
        return e


class SafeMoleculeProcessor:
    """Safe molecule processor with persistent process pool"""
    
    def __init__(self, config, global_stats, device, timeout=30):
        self.config = config
        self.global_stats = global_stats
        self.device_str = str(device)
        self.timeout = timeout
        self.pool = None
        self._init_pool()
    
    def _init_pool(self):
        """Initialize process pool"""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        self.pool = mp.Pool(processes=1)
    
    def process_molecule(self, smiles, mol_id):
        """Process a single molecule with timeout protection"""
        args = (smiles, mol_id, self.config, self.global_stats, self.device_str)
        
        try:
            # Submit the job and wait with timeout
            result = self.pool.apply_async(_smiles_to_hypergraph_worker, (args,))
            data = result.get(timeout=self.timeout)
            
            # Check if result is an exception
            if isinstance(data, Exception):
                raise data
                
            return data
            
        except mp.TimeoutError:
            # Restart the pool after timeout (process might be corrupted)
            self._init_pool()
            raise TimeoutException(f"Multiprocessing timeout ({self.timeout}s) for molecule {mol_id}")
        except Exception as e:
            # Restart the pool after any error
            self._init_pool()
            raise e
    
    def close(self):
        """Clean up the process pool"""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
    
    def __del__(self):
        """Cleanup on destruction"""
        self.close()


SUPPORTED_DATASETS = ["BBBP", "BACE", "SIDER", "ESOL", "Tox21", "ClinTox"]

# Dataset type groups for label masking rules
# KISS: extend only per common MolNet definitions
CLASSIFICATION_DATASETS = {"BBBP", "BACE", "HIV"}
MULTILABEL_DATASETS = {"Tox21", "SIDER", "ClinTox", "ToxCast"}
REGRESSION_DATASETS = {"ESOL", "FreeSolv", "Lipophilicity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepChem MolNet datasets to hypergraph files")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory to save converted datasets")
    parser.add_argument("--datasets", type=str, nargs="*", default=SUPPORTED_DATASETS,
                        help=f"Datasets to convert (default: {SUPPORTED_DATASETS})")
    parser.add_argument("--batch_size_save", type=int, default=1000,
                        help="Number of graphs per saved batch_*.pt file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of samples per dataset for quick conversion")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device used during hypergraph construction (default: cpu)")
    parser.add_argument("--config", type=str, default="config/normal_config.yaml",
                        help="Configuration file path (default: config/normal_config.yaml)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing dataset directory if present")
    return parser.parse_args()


def _load_deepchem_dataset(name: str):
    """Load a MolNet dataset from DeepChem with raw featurizer.

    Returns:
        tasks, (train_ds, valid_ds, test_ds)
    """
    try:
        import deepchem as dc
    except Exception as e:
        raise RuntimeError(
            "DeepChem is required for this script. Install with: pip install deepchem"
        ) from e

    uname = name.upper()
    if uname == "BBBP":
        return dc.molnet.load_bbbp(featurizer="Raw", splitter="scaffold")
    if uname == "BACE":
        return dc.molnet.load_bace_classification(featurizer="Raw", splitter="scaffold")
    if uname == "SIDER":
        return dc.molnet.load_sider(featurizer="Raw", splitter="scaffold")
    if uname == "ESOL":
        return dc.molnet.load_delaney(featurizer="Raw", splitter="scaffold")
    if uname == "TOX21" or uname == "TOX-21" or uname == "TOX_21" or uname == "TOX 21" or uname == "TOX21":
        return dc.molnet.load_tox21(featurizer="Raw", splitter="scaffold")
    if uname == "CLINTOX":
        return dc.molnet.load_clintox(featurizer="Raw", splitter="scaffold")
    if uname == "HIV":
        return dc.molnet.load_hiv(featurizer="Raw", splitter="scaffold")
    if uname == "FREESOLV":
        return dc.molnet.load_freesolv(featurizer="Raw", splitter="scaffold")
    if uname in ("LIPO", "LIPOPHILICITY"):
        return dc.molnet.load_lipo(featurizer="Raw", splitter="scaffold")
    if uname == "TOXCAST":
        return dc.molnet.load_toxcast(featurizer="Raw", splitter="scaffold")
    # QM7 support removed

    raise ValueError(f"Unsupported dataset: {name}")


def _ids_and_labels_from_ds(ds) -> Tuple[List[str], np.ndarray]:
    """Extract SMILES (ids) and labels from a single DeepChem dataset split.

    Requires featurizer='Raw' so that ds.ids are SMILES.
    Returns labels as np.ndarray (keeps original shape; caller will normalize to 2D).
    """
    ids = list(ds.ids) if getattr(ds, "ids", None) is not None else None
    if ids is None:
        raise RuntimeError(
            "DeepChem dataset has no ids (SMILES). Ensure loaders use featurizer='Raw' so ids are SMILES."
        )
    y = np.array(ds.y)
    return ids, y


def _save_batches(graphs: List, out_dir: Path, batch_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(graphs)
    if total == 0:
        return 0
    num_files = int(math.ceil(total / float(batch_size)))
    for i in range(num_files):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        chunk = graphs[start:end]
        out_path = out_dir / f"batch_{i:04d}.pt"
        torch.save(chunk, out_path)
    return num_files


def convert_one_dataset(dataset_name: str, output_root: Path, batch_size_save: int,
                        max_samples: int, device: torch.device, config: dict, logger: logging.Logger) -> Dict:
    logger.info(f"Loading DeepChem dataset: {dataset_name}")
    tasks, datasets, transformers = _load_deepchem_dataset(dataset_name)
    train_ds, valid_ds, test_ds = datasets

    # Target metadata (units/scaling) for transparency
    target_info = {
        'unit': None,
        'scaling': 'raw',  # No standardization of targets by default
        'description': None,
    }
    if dataset_name == 'FreeSolv':
        target_info.update({
            'unit': 'kcal/mol',
            'description': 'Hydration free energy (experimental/computational)'
        })
    # QM7 target metadata removed

    # Helper: attempt to untransform labels from DeepChem NormalizationTransformer if present
    def _maybe_untransform_labels(y_arr: np.ndarray) -> np.ndarray:
        try:
            if transformers is None:
                return y_arr
            # DeepChem returns a list of transformers
            for t in transformers:
                try:
                    # Only care about label normalization transformers
                    if hasattr(t, 'transform_y') and getattr(t, 'transform_y'):
                        # Try to recover means/stds from common attribute names
                        means = None
                        stds = None
                        for key in ('y_means', '_y_means', 'means_', 'means'):
                            if hasattr(t, key):
                                means = getattr(t, key)
                                break
                        for key in ('y_stds', '_y_stds', 'stds_', 'stds'):
                            if hasattr(t, key):
                                stds = getattr(t, key)
                                break
                        if means is not None and stds is not None:
                            m = np.asarray(means).reshape(-1)
                            s = np.asarray(stds).reshape(-1)
                            # Avoid division-by-zero in case of degenerate stds
                            s = np.where(np.abs(s) < 1e-12, 1.0, s)
                            y = np.asarray(y_arr)
                            if y.ndim == 1:
                                if m.size == 1:
                                    y_raw = y * s[0] + m[0]
                                else:
                                    # Single target expected; fallback no-op
                                    y_raw = y
                                return y_raw
                            elif y.ndim == 2:
                                # Broadcast per-target if dimensions match
                                if y.shape[1] == m.size:
                                    y_raw = y * s.reshape(1, -1) + m.reshape(1, -1)
                                    return y_raw
                                else:
                                    return y
                        # Fallback: try method-based undo
                        if hasattr(t, 'untransform'):  # Rare API
                            try:
                                return t.untransform(np.asarray(y_arr))
                            except Exception:
                                pass
                        if hasattr(t, 'undo_transform'):
                            try:
                                return t.undo_transform(np.asarray(y_arr))
                            except Exception:
                                pass
                except Exception:
                    continue
            return y_arr
        except Exception:
            return y_arr

    # Extract per-split SMILES and labels
    smiles_train, labels_train = _ids_and_labels_from_ds(train_ds)
    smiles_valid, labels_valid = _ids_and_labels_from_ds(valid_ds)
    smiles_test, labels_test = _ids_and_labels_from_ds(test_ds)

    # Try undo any DeepChem normalization on labels for regression datasets only
    if dataset_name in REGRESSION_DATASETS:
        try:
            # Detect if DeepChem applied a label transform
            transformed_y = False
            if transformers is not None:
                for t in transformers:
                    try:
                        if getattr(t, 'transform_y', False) or 'normalization' in type(t).__name__.lower():
                            transformed_y = True
                            break
                    except Exception:
                        continue

            if transformed_y:
                # Always adopt untransformed labels when a y-transformer is detected
                labels_train = _maybe_untransform_labels(labels_train)
                labels_valid = _maybe_untransform_labels(labels_valid)
                labels_test = _maybe_untransform_labels(labels_test)
                target_info['scaling'] = 'raw'
            else:
                # No label transformer detected; assume already raw
                target_info['scaling'] = 'raw'
        except Exception:
            # If anything goes wrong, keep original labels but mark unknown
            target_info['scaling'] = 'unknown'

    # Optional: MedDRA task filtering for SIDER
    kept_task_names = None
    removed_task_names = None
    if dataset_name == "SIDER" and isinstance(tasks, (list, tuple)) and len(tasks) > 0:
        try:
            task_names = list(tasks)
            num_tasks = len(task_names)
            keep_mask = np.ones(num_tasks, dtype=bool)

            # 1) Remove specified SOCs
            exclude_socs = {
                "social circumstances",
                "product issues",
                "surgical and medical procedures",
                "injury, poisoning and procedural complications",
            }
            for i, t in enumerate(task_names):
                if isinstance(t, str) and t.strip().casefold() in exclude_socs:
                    keep_mask[i] = False

            def apply_keep(arr, mask):
                if arr is None:
                    return arr
                if arr.ndim == 1:
                    return arr  # single-task, nothing to filter
                return arr[:, mask]

            labels_train_f = apply_keep(np.array(labels_train), keep_mask)
            labels_valid_f = apply_keep(np.array(labels_valid), keep_mask)
            labels_test_f  = apply_keep(np.array(labels_test),  keep_mask)

            # 2) Keep tasks with positives >= 30 and positive ratio >= 0.5%
            if labels_train_f is not None and labels_train_f.ndim == 2:
                combined = np.concatenate([labels_train_f, labels_valid_f, labels_test_f], axis=0)
                # valid mask: labels in {0,1}; missing often -1
                valid_per_task = np.sum((combined == 0) | (combined == 1), axis=0)
                pos_per_task = np.sum(combined == 1, axis=0)
                ratio = np.divide(pos_per_task, np.maximum(valid_per_task, 1), where=valid_per_task>0)
                crit_mask = (pos_per_task >= 30) & (ratio >= 0.005)
                # Combine masks
                if crit_mask.shape[0] == keep_mask.shape[0]:
                    keep_mask = keep_mask & crit_mask

                # Apply final keep mask
                labels_train = apply_keep(np.array(labels_train), keep_mask)
                labels_valid = apply_keep(np.array(labels_valid), keep_mask)
                labels_test  = apply_keep(np.array(labels_test),  keep_mask)

                kept_task_names = [t for t, k in zip(task_names, keep_mask) if k]
                removed_task_names = [t for t, k in zip(task_names, keep_mask) if not k]
                logger.info(f"SIDER MedDRA filtering: kept {len(kept_task_names)}/{num_tasks} tasks; removed {len(removed_task_names)}")
        except Exception as e:
            logger.warning(f"SIDER MedDRA filtering skipped due to error: {e}")

    # Build global statistics ONLY from train split (avoid leakage)
    smiles_train_valid = [s for s in smiles_train if isinstance(s, str) and len(s) > 0]
    # Standardize train SMILES to fix valence/salt issues and reduce RDKit errors
    standardizer = MoleculeStandardizer(remove_metals=True)
    std_train_smiles, std_failed_idx = standardizer.batch_standardize(smiles_train_valid)
    std_train_smiles_clean = [s for s in std_train_smiles if isinstance(s, str) and len(s) > 0]
    if std_failed_idx:
        logger.info(f"Standardization failed for {len(std_failed_idx)} train SMILES; they will be excluded from statistics")
    logger.info(f"Computing global statistics from {len(std_train_smiles_clean)} standardized train SMILES")
    stats_builder = DataStatistics()
    global_stats = stats_builder.compute_global_statistics_from_smiles(std_train_smiles_clean) if std_train_smiles_clean else {
        'cont_mean': [0.0, 0.0, 0.0],
        'cont_std': [1.0, 1.0, 1.0],
        'degree_cat': list(range(10)),
        'hybridization_cats': [0, 1, 2, 3, 4, 5, 6],
        'atom_chiral_cats': [0, 1, 2, 3],
        'bond_cont_means': [0.0],
        'bond_cont_stds': [1.0],
        'bond_stereo_cats': [0, 1, 2, 3, 4, 5],
        'bond_type_cats': [1, 2, 3, 12],
    }

    # Concatenate splits for conversion output (train/valid/test order)
    smiles_all: List[str] = list(smiles_train) + list(smiles_valid) + list(smiles_test)
    splits_all: List[str] = (
        ["train"] * len(smiles_train)
        + ["valid"] * len(smiles_valid)
        + ["test"] * len(smiles_test)
    )
    labels_list: List[np.ndarray] = []
    for arr in (labels_train, labels_valid, labels_test):
        if arr.ndim == 1:
            labels_list.append(arr[:, None])
        else:
            labels_list.append(arr)
    labels_all = np.concatenate(labels_list, axis=0)

    # Optional sample cap for quick conversion
    if max_samples is not None:
        smiles_all = smiles_all[:max_samples]
        labels_all = labels_all[:max_samples]
        splits_all = splits_all[:max_samples]

    graphs: List[object] = []
    skipped = 0
    timeout_count = 0
    error_count = 0
    
    # Configuration will be loaded from file
    
    # Create safe processor with persistent process pool - always use CPU for multiprocessing
    cpu_device = torch.device("cpu")
    processor = SafeMoleculeProcessor(config, global_stats, cpu_device, timeout=30)
    
    # Manual skip list (only for SIDER). Indices refer to zero-based order during conversion
    sider_manual_skip_indices = set()
    if dataset_name == "SIDER":
        sider_manual_skip_indices = {
            745, 747, 748, 765, 769, 770, 798, 855, 911, 979,
            1105, 1147, 1275, 1277, 1289, 1299, 1301, 1341, 1344,
            1347, 1353, 1365, 1410, 1414
        }
    
    try:
        for idx, (smi, yrow, split_tag) in enumerate(tqdm(zip(smiles_all, labels_all, splits_all), total=len(smiles_all), desc=f"{dataset_name}: converting")):
            if not isinstance(smi, str) or len(smi) == 0:
                skipped += 1
                continue
            
            # Skip known problematic molecules in SIDER by explicit index list
            if dataset_name == "SIDER" and idx in sider_manual_skip_indices:
                logger.warning(f"Skipping pre-identified problematic SIDER molecule at index {idx}: {smi}")
                skipped += 1
                continue
            try:
                # Standardize individual SMILES to fix common valence/salt/normalization issues
                std_smi = standardizer.standardize_smiles(smi)
                if std_smi is None:
                    error_count += 1
                    skipped += 1
                    continue
                # Use safe processor to handle C++ hangs
                data = processor.process_molecule(std_smi, mol_id=str(idx))
                if data is None:
                    skipped += 1
                    continue
                
                # Deserialize IPC payload back into torch_geometric Data
                data = _deserialize_data_from_ipc(data)
                if isinstance(data, Exception) or data is None:
                    skipped += 1
                    continue
                # Attach label vector (1-D) and mask (classification: valid {0,1}; multi-label: valid {0,1}; regression: finite)
                yrow_np = np.array(yrow)
                if yrow_np.ndim == 0:
                    y_vec = np.array([float(yrow_np)], dtype=float)
                elif yrow_np.size == 1:
                    y_vec = np.array([float(yrow_np.reshape(-1)[0])], dtype=float)
                else:
                    y_vec = yrow_np.astype(float).reshape(-1)

                if dataset_name in REGRESSION_DATASETS:
                    # ESOL: regression — mask finite values
                    y_mask = np.isfinite(y_vec)
                elif dataset_name in MULTILABEL_DATASETS:
                    # Tox21: multi-label — valid labels are exactly 0 or 1; -1 treated as missing
                    y_mask = np.isin(y_vec, [0.0, 1.0])
                    # Optionally clamp label values into {0,1} for downstream safety
                    y_vec = np.where(y_vec >= 0.5, 1.0, np.where(y_vec < 0.5, 0.0, y_vec))
                else:
                    # Binary classification datasets (BBBP/BACE): explicit label mapping
                    vals = np.unique(y_vec[~np.isnan(y_vec)])
                    logger.debug(f"{dataset_name} sample {idx}: label values = {vals}")
                    
                    if set(vals).issubset({0.0, 1.0}):
                        # DeepChem BACE/BBBP standard: 0/1; keep original values
                        y_mask = np.isin(y_vec, [0.0, 1.0])
                        # y_vec unchanged - no transformation
                    elif set(vals).issubset({-1.0, 1.0}):
                        # Explicit mapping from {-1,1} to {0,1}
                        y_mask = np.isin(y_vec, [-1.0, 1.0])
                        y_vec = (y_vec == 1.0).astype(float)
                        logger.info(f"{dataset_name}: mapped {{-1,1}} labels to {{0,1}}")
                    else:
                        raise ValueError(f"{dataset_name}: unexpected label set {vals}. Define explicit mapping.")
                data.y = torch.tensor(y_vec, dtype=torch.float32)
                data.y_mask = torch.tensor(y_mask, dtype=torch.bool)
                # Attach target metadata for downstream transparency (unit/scaling)
                try:
                    if dataset_name in REGRESSION_DATASETS:
                        data.y_unit = target_info.get('unit', None)
                        data.y_scaling = target_info.get('scaling', 'raw')
                        data.y_description = target_info.get('description', None)
                except Exception:
                    pass
                # Ensure SMILES and split recorded
                try:
                    data.smiles = std_smi
                    data.split = split_tag
                except Exception:
                    pass
                graphs.append(data.cpu())
            except TimeoutException as e:
                logger.warning(f"Timeout converting sample {idx} ({smi}): {e}")
                timeout_count += 1
                skipped += 1
                continue
            except Exception as e:
                logger.debug(f"Failed to convert sample {idx} ({smi}): {e}")
                error_count += 1
                skipped += 1
                continue

    finally:
        # Always cleanup the processor
        processor.close()

    # Log label statistics for transparency
    try:
        if dataset_name in REGRESSION_DATASETS:
            # Regression datasets: report basic stats (no normalization)
            split_stats = {}
            for split_name in ["train", "valid", "test"]:
                split_graphs = [g for g in graphs if hasattr(g, 'split') and g.split == split_name]
                if split_graphs:
                    labels = torch.cat([g.y for g in split_graphs]).float()
                    masks = torch.cat([g.y_mask for g in split_graphs])
                    vals = labels[masks]
                    if len(vals) > 0:
                        split_stats[split_name] = {
                            "n_total": len(split_graphs),
                            "n_valid": int(masks.sum()),
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                            "mean": float(vals.mean()),
                            "std": float(vals.std(unbiased=False)),
                            "unit": target_info.get('unit', None),
                            "scaling": target_info.get('scaling', 'raw')
                        }
            logger.info(f"{dataset_name} regression targets stats by split: {split_stats}")
        else:
            # Classification/multilabel: keep positive ratio summary
            split_stats = {}
            for split_name in ["train", "valid", "test"]:
                split_graphs = [g for g in graphs if hasattr(g, 'split') and g.split == split_name]
                if split_graphs:
                    labels = torch.cat([g.y for g in split_graphs])
                    masks = torch.cat([g.y_mask for g in split_graphs])
                    valid_labels = labels[masks]
                    if len(valid_labels) > 0:
                        pos_ratio = float((valid_labels == 1.0).float().mean())
                        split_stats[split_name] = {
                            "n_total": len(split_graphs),
                            "n_valid": int(masks.sum()),
                            "pos_ratio": pos_ratio
                        }
            logger.info(f"{dataset_name} label distribution by split: {split_stats}")
    except Exception as e:
        logger.warning(f"Failed to compute label statistics: {e}")

    # Save
    ds_out_dir = output_root / dataset_name
    logger.info(f"Saving {len(graphs)} graphs to {ds_out_dir} (batch size {batch_size_save})")
    logger.info(f"Processing summary - Total: {len(smiles_all)}, Converted: {len(graphs)}, "
               f"Skipped: {skipped} (Timeouts: {timeout_count}, Errors: {error_count})")
    num_files = _save_batches(graphs, ds_out_dir, batch_size_save)
    # Save dataset-level target metadata alongside batches
    try:
        import json as _json
        (ds_out_dir).mkdir(parents=True, exist_ok=True)
        with open(ds_out_dir / 'target_info.json', 'w', encoding='utf-8') as f:
            _json.dump(target_info, f, indent=2)
    except Exception as _e:
        logger.warning(f"Failed to save target_info.json: {_e}")
    result = {
        "dataset": dataset_name,
        "total": len(smiles_all),
        "converted": len(graphs),
        "skipped": skipped,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "batches": num_files,
        "output_dir": str(ds_out_dir),
        "target_info": target_info
    }
    # Attach SIDER task filtering summary if available
    if dataset_name == "SIDER" and kept_task_names is not None:
        result["tasks_total"] = len(kept_task_names) + len(removed_task_names or [])
        result["tasks_kept"] = len(kept_task_names)
        result["tasks_removed"] = len(removed_task_names or [])
        result["kept_tasks"] = kept_task_names
    return result


def main():
    # Set multiprocessing start method to 'spawn' for better isolation
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Disable PyTorch's multiprocessing sharing strategy to avoid tensor serialization issues
    try:
        import torch.multiprocessing as torch_mp
        torch_mp.set_sharing_strategy('file_system')
    except:
        pass
    
    args = parse_args()
    output_root = Path(args.output_root)

    logger = setup_logger("convert_deepchem", log_dir=output_root, level="INFO")
    logger.info(f"Output root: {output_root}")

    # Prepare output root
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        logger.info("Output root exists and is not empty. Use --overwrite to continue or choose another directory.")
    output_root.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using CUDA for parts of the pipeline where applicable")
    else:
        logger.info("Using CPU device")

    # Load configuration (reusing main preprocessing pipeline logic)
    import yaml
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
        logger.info(f"Hypergraph type: {config.get('hypergraph_type', 'default')}")
    except Exception as e:
        logger.error(f"Failed to load config from {args.config}: {e}")
        raise

    # Datasets to process
    # Robust name normalization via explicit mapping (avoid capitalize pitfalls)
    name_map = {
        'bbbp': 'BBBP', 'BBBP': 'BBBP',
        'bace': 'BACE', 'BACE': 'BACE',
        'sider': 'SIDER', 'SIDER': 'SIDER',
        'esol': 'ESOL', 'ESOL': 'ESOL', 'delaney': 'ESOL', 'Delaney': 'ESOL',
        'tox21': 'Tox21', 'TOX21': 'Tox21', 'Tox21': 'Tox21',
        'clintox': 'ClinTox', 'CLINTOX': 'ClinTox', 'ClinTox': 'ClinTox',
        'hiv': 'HIV', 'HIV': 'HIV',
        'freesolv': 'FreeSolv', 'FreeSolv': 'FreeSolv',
        'lipo': 'Lipophilicity', 'Lipophilicity': 'Lipophilicity', 'LIPO': 'Lipophilicity',
        'toxcast': 'ToxCast', 'TOXCAST': 'ToxCast', 'ToxCast': 'ToxCast'
    }
    normalized = []
    for raw in args.datasets:
        key = raw if raw in name_map else raw.lower()
        if key in name_map:
            normalized.append(name_map[key])
        else:
            logger.warning(f"Skipping unsupported dataset: {raw}")
    target_datasets = normalized

    summary: Dict[str, Dict] = {}
    for name in target_datasets:
        try:
            info = convert_one_dataset(
                dataset_name=name,
                output_root=output_root,
                batch_size_save=args.batch_size_save,
                max_samples=args.max_samples,
                device=device,
                config=config,
                logger=logger,
            )
            logger.info(f"{name}: converted {info['converted']}/{info['total']} "
                       f"(skipped {info['skipped']}: {info.get('timeout_count', 0)} timeouts, "
                       f"{info.get('error_count', 0)} errors) -> {info['batches']} files")
            summary[name] = info
        except Exception as e:
            logger.error(f"Failed to convert {name}: {e}")
            summary[name] = {"dataset": name, "error": str(e)}

    # Save summary JSON
    try:
        import json
        summary_path = output_root / "conversion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to save summary: {e}")


if __name__ == "__main__":
    # Protect multiprocessing code
    mp.freeze_support()  # For Windows compatibility
    main()


