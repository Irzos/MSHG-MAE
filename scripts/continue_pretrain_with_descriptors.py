#!/usr/bin/env python
"""
Continue pretraining with counterfactual Δ-descriptor supervision (teacher-free).

Storyline: From a strong reconstruction/edge self-supervised checkpoint, we add a
counterfactual Δ-property target aligned with the model's semantic masking. The Δ* target
is computed from RDKit atomic/fragment-additive contributions (e.g., Crippen logP/MR, TPSA):
Δ* = sum of per-atom contributions over the masked subgraph S. We then train a descriptor
head to match Δ̂ = s(x) - s(x_cf) (same head on pooled graph embeddings), using Huber loss,
while keeping the original reconstruction/edge losses. No teacher model or real molecule
generation is needed.

Usage example:
  python scripts/continue_pretrain_with_descriptors.py \
      --data_dir /path/to/bonds_only_dataset \
      --config hydra/version2/configs/max_full.yaml \
      --checkpoint /path/to/best_model_or_final_checkpoint.pth \
      --output_dir experiments \
      --experiment_name mae_plus_delta \
      --steps 10000 \
      --delta_names "MolLogP MolMR TPSA"
"""

import argparse
import json
from pathlib import Path
import sys
import yaml
import torch
import torch.nn.functional as F
import logging
import math
import hashlib
import sqlite3
from typing import List, Tuple, Dict, Any
from collections import OrderedDict

# Append project root
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hypergraph_mae import EnhancedHyperGraphMAE
from src.data.data_loader import MolecularHypergraphDataset, split_dataset, create_data_loaders, set_collate_hyperedge_dim
from src.utils.logging_utils import ExperimentLogger
from src.training.trainer import HyperGraphMAETrainer
from src.utils.memory_utils import optimize_memory_allocation


def parse_args():
    p = argparse.ArgumentParser(description="Continue pretraining with descriptor head")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Pretrained checkpoint to init backbone weights")
    p.add_argument("--output_dir", type=str, default="experiments")
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--steps", type=int, default=10000, help="Number of steps to run for fine-tune phase")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    # Incremental hypergraph loader options
    p.add_argument("--use_incremental_loader", action="store_true",
                   help="Force using hydra/version2 IncrementalHypergraphLoader even if hypergraph_types=['bond']")
    p.add_argument("--hypergraph_types", type=str, default=None,
                   help="Comma/space-separated hyperedge types, e.g. 'bond ring functional_group hydrogen_bond'.\n"
                        "If provided, overrides config.training.hypergraph_types and enables incremental loader.")
    p.add_argument("--incremental_cache_dir", type=str, default="hydra/version2/cache",
                   help="Cache directory for IncrementalHypergraphLoader")
    # Δ-descriptor supervision options (teacher-free)
    p.add_argument("--delta_names", type=str, default="MolLogP MolMR TPSA EState LabuteASA MolWt",
                   help=(
                       "Space/comma-separated RDKit descriptor names for Δ supervision. "
                       "Supported per-atom additive targets: MolLogP, MolMR, TPSA, EState, LabuteASA, MolWt, Gasteiger"
                   ))
    p.add_argument("--delta_weight_max", type=float, default=0.25,
                   help="Max weight for Δ loss; applied after warmup")
    p.add_argument("--delta_warmup_ratio", type=float, default=0.25,
                   help="Warmup ratio of total steps for Δ loss weight")
    p.add_argument("--delta_huber_beta", type=float, default=1.0,
                   help="Huber beta for Δ loss (SmoothL1 beta)")
    # Absolute anchor subset: choose which dims (subset of --delta_names) participate in L_abs
    p.add_argument(
        "--abs_anchor_names",
        type=str,
        default=None,
        help=(
            "Space/comma-separated descriptor names used for absolute anchor L_abs. "
            "If omitted, defaults to {MolLogP, MolMR, TPSA, LabuteASA, EState}. "
            "The actual supervised dims are the intersection with --delta_names. "
            "Use --abs_anchor_weight 0 to disable absolute anchoring entirely."
        ),
    )
    # Accept optional value; if provided without arg, use default 0.08; set to 0 to disable
    p.add_argument("--abs_anchor_weight", type=float, nargs='?', const=0.08, default=0.08,
                   help=(
                       "Weight α for absolute descriptor anchor loss L_abs = MSE(s_base, y_abs_RDKit). "
                       "Provide a value (e.g., --abs_anchor_weight 0.08); if flag is present without a value, "
                       "defaults to 0.08; set to 0 to disable."
                   ))
    # Checkpoint interval
    p.add_argument("--ckpt_every", type=int, default=1000,
                   help="Save checkpoint every N steps (default: 1000)")
    # Debugging options
    p.add_argument("--debug_nans", action="store_true",
                   help="Enable extensive NaN/Inf checks and gradient diagnostics; slows training")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_descriptor_in_config(cfg: dict, args) -> dict:
    cfg = cfg.copy()
    # Ensure training loss components (keep reconstruction/edge only; Δ-loss will be added externally)
    train = cfg.setdefault('training', {})
    train['loss_components'] = ['reconstruction', 'edge']
    # Mask ratio schedule: lower difficulty for more stable Δ supervision
    train['mask_ratio_max'] = 0.4
    train['mask_ratio_min'] = 0.25

    # Keep/enable TCC for reconstruction/edge balancing
    tcc = train.setdefault('tcc', {})
    tcc.setdefault('enabled', True)

    # Descriptor head settings for Δ prediction (graph-level head, but used for Δ̂)
    desc_cfg = cfg.setdefault('descriptor_head', {})
    desc_cfg['enabled'] = True
    # Split by comma/space and normalize names
    raw = (args.delta_names or "").replace(',', ' ').split()
    names = [n.strip() for n in raw if n.strip()]
    # Normalize aliases
    norm_map = {
        'MR': 'MolMR',
        'ExactMolWt': 'MolWt',
        'EStateIndices': 'EState',
        'EStateIndex': 'EState',
        'GasteigerCharge': 'Gasteiger',
        'PartialCharge': 'Gasteiger',
        'GasteigerPartialCharge': 'Gasteiger',
        'GasteigerPC': 'Gasteiger'
    }
    names = [norm_map.get(n, n) for n in names] if names else ['MolLogP', 'MolMR', 'TPSA', 'EState', 'LabuteASA', 'MolWt']
    desc_cfg['names'] = names
    desc_cfg.setdefault('hidden_dim', 128)
    # Keep normalization for descriptor head internal targets off (we supervise Δ separately)
    desc_cfg['normalize'] = False
    desc_cfg.setdefault('momentum', 0.9)
    # No warmup for direct descriptor head inside model (we don't use that loss here)
    desc_cfg.setdefault('warmup_freeze_steps', 0)

    # Optionally override hypergraph types from CLI to trigger incremental loading
    if args.hypergraph_types:
        raw = args.hypergraph_types.replace(',', ' ').split()
        types = [t.strip() for t in raw if t.strip()]
        if types:
            cfg['hypergraph_types'] = types

    # Scheduler defaults: WSD with shorter stable plateau (0.50)
    sched = train.setdefault('scheduler', {})
    if not sched:
        sched['type'] = 'WSD'
    if sched.get('type', 'WSD') == 'WSD':
        sched['stable_ratio'] = 0.50

    return cfg


def mask_ratio_at(step: int, max_steps: int, hi: float = 0.7, lo: float = 0.3) -> float:
    total = max(1, int(max_steps))
    s = max(0, min(int(step), total))
    t = s / total
    frac = 0.5 * (1.0 - math.cos(math.pi * t))
    ratio = hi - frac * (hi - lo)
    return float(max(0.0, min(1.0, ratio)))


def _z_enhanced_from_model(model: EnhancedHyperGraphMAE,
                           x: torch.Tensor,
                           he_idx: torch.Tensor,
                           he_attr: torch.Tensor,
                           node_mask: torch.Tensor,
                           edge_mask: torch.Tensor,
                           smiles,
                           global_step: int,
                           total_steps: int,
                           use_amp: bool = False,
                           autocast_dtype: torch.dtype = None) -> torch.Tensor:
    """Get z_enhanced with a safe fallback if eval_mode is unsupported.

    Primary path: call model(..., eval_mode=True) to get (recon_x, edge_pred, z_enhanced).
    Fallback: manually apply masking + light PE + encoder + hypergraph_attention.
    """
    try:
        if use_amp and autocast_dtype is not None and torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True):
                _, _, z = model(
                    x, he_idx, he_attr,
                    node_mask, edge_mask,
                    global_step=global_step, max_steps=total_steps,
                    smiles=smiles, eval_mode=True
                )
        else:
            _, _, z = model(
                x, he_idx, he_attr,
                node_mask, edge_mask,
                global_step=global_step, max_steps=total_steps,
                smiles=smiles, eval_mode=True
            )
        return z
    except TypeError:
        # Manual path (keep it minimal and consistent with model.forward)
        x = x.float()
        if he_attr is not None:
            he_attr = he_attr.float()
        if he_idx.numel() > 0:
            he_idx = he_idx.long()
        # Apply node/edge masks and run encoder+attention under autocast for dtype coherence
        if use_amp and autocast_dtype is not None and torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True):
                x_masked = x.clone()
                if node_mask is not None and node_mask.any():
                    x_masked[node_mask] = 0
                he_attr_masked = he_attr
                if he_attr is not None and edge_mask is not None and edge_mask.numel() > 0 and edge_mask.any():
                    he_attr_masked = he_attr.clone()
                    masked_indices = edge_mask.nonzero(as_tuple=True)[0]
                    if masked_indices.numel() > 0:
                        he_attr_masked[masked_indices] = 0
                x_pe = model._add_light_positional_encoding(x_masked, he_idx)
                z_latent = model.encoder(x_pe, he_idx, he_attr_masked, None)
                z_enh = model.hypergraph_attention(z_latent, he_idx, he_attr_masked, None)
        else:
            x_masked = x.clone()
            if node_mask is not None and node_mask.any():
                x_masked[node_mask] = 0
            he_attr_masked = he_attr
            if he_attr is not None and edge_mask is not None and edge_mask.numel() > 0 and edge_mask.any():
                he_attr_masked = he_attr.clone()
                masked_indices = edge_mask.nonzero(as_tuple=True)[0]
                if masked_indices.numel() > 0:
                    he_attr_masked[masked_indices] = 0
            x_pe = model._add_light_positional_encoding(x_masked, he_idx)
            z_latent = model.encoder(x_pe, he_idx, he_attr_masked, None)
            z_enh = model.hypergraph_attention(z_latent, he_idx, he_attr_masked, None)
        return z_enh


_SMILES_PROP_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_SMILES_PROP_CACHE_CAP = 20000
_SQLITE_CONN = None


def _sqlite_path(cache_dir: Path) -> Path:
    return cache_dir / "smiles_propertycache.sqlite"

def _ensure_sqlite(cache_dir: Path):
    global _SQLITE_CONN
    if cache_dir is None:
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    db_path = _sqlite_path(cache_dir)
    if _SQLITE_CONN is None:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        conn.execute(
            "CREATE TABLE IF NOT EXISTS props (hash TEXT PRIMARY KEY, payload TEXT)"
        )
        _SQLITE_CONN = conn
    return _SQLITE_CONN


def _cache_put(smiles: str, payload: Dict[str, Any], cache_dir: Path = None):
    # memory LRU
    if smiles in _SMILES_PROP_CACHE:
        _SMILES_PROP_CACHE.pop(smiles, None)
    _SMILES_PROP_CACHE[smiles] = payload
    if len(_SMILES_PROP_CACHE) > _SMILES_PROP_CACHE_CAP:
        _SMILES_PROP_CACHE.popitem(last=False)
    # sqlite disk cache (single file, robust)
    try:
        conn = _ensure_sqlite(cache_dir)
        if conn is not None:
            h = hashlib.sha1(smiles.encode('utf-8')).hexdigest()
            conn.execute("INSERT OR REPLACE INTO props(hash, payload) VALUES(?, ?)", (h, json.dumps(payload)))
            conn.commit()
    except Exception:
        pass


def _cache_get(smiles: str, cache_dir: Path = None) -> Dict[str, Any]:
    # memory
    if smiles in _SMILES_PROP_CACHE:
        val = _SMILES_PROP_CACHE.pop(smiles)
        _SMILES_PROP_CACHE[smiles] = val
        return val
    # sqlite disk cache
    try:
        conn = _ensure_sqlite(cache_dir)
        if conn is not None:
            h = hashlib.sha1(smiles.encode('utf-8')).hexdigest()
            cur = conn.execute("SELECT payload FROM props WHERE hash=?", (h,))
            row = cur.fetchone()
            if row and row[0]:
                payload = json.loads(row[0])
                # populate memory LRU
                _cache_put(smiles, payload, cache_dir=None)
                return payload
    except Exception:
        return None
    return None


def _compute_batch_delta_targets(smiles_list: List[str], batch_vec: torch.Tensor,
                                 node_mask: torch.Tensor, names: List[str],
                                 device: torch.device, cache_dir: Path = None,
                                 rdkit_idx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute Δ* targets per-graph by summing atomic contributions over masked nodes.

    Returns:
        delta_star: [G, D] tensor (zeros where invalid)
        valid_mask: [G] boolean tensor indicating which graphs have valid Δ*
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors as rdm
        from rdkit.Chem import EState as rdEState
        try:
            from rdkit.Chem import rdPartialCharges as rdPC
        except Exception:
            rdPC = None
    except Exception as e:
        raise ImportError(f"RDKit is required for Δ supervision: {e}")

    num_graphs = int(batch_vec.max().item() + 1) if batch_vec.numel() > 0 else len(smiles_list)
    D = len(names)
    # Use NaN to mark missing descriptor dims per-graph; caller will mask them
    delta_star = torch.full((num_graphs, D), float('nan'), dtype=torch.float32, device=device)
    valid = torch.zeros((num_graphs,), dtype=torch.bool, device=device)
    # Diagnostics counters
    n_no_smiles = 0
    n_atoms_mismatch = 0
    n_no_mask_nodes = 0
    dim_avail_counts = torch.zeros((D,), dtype=torch.long)
    n_probed = 0  # graphs with smiles and atoms matched and mask non-empty

    # Move masks to CPU for indexing, but keep indices list
    batch_cpu = batch_vec.detach().cpu()
    mask_cpu = node_mask.detach().cpu()
    rki_cpu = rdkit_idx.detach().cpu() if isinstance(rdkit_idx, torch.Tensor) else None

    # Precompute group indices per graph
    graph_indices: List[torch.Tensor] = []
    for g in range(num_graphs):
        graph_indices.append((batch_cpu == g).nonzero(as_tuple=True)[0])

    examples = []  # sample diagnostics
    for g, smi in enumerate(smiles_list):
        if smi is None:
            n_no_smiles += 1
            continue
        try:
            # try cache first
            payload = _cache_get(smi, cache_dir)
            if (payload is None) or ('n_atoms_noH' not in payload):
                mol_noH = Chem.MolFromSmiles(smi)
                if mol_noH is None:
                    # cache skip
                    _cache_put(smi, {"skip": True}, cache_dir)
                    continue
                # Also build explicit-H molecule
                try:
                    mol_H = Chem.AddHs(mol_noH)
                except Exception:
                    mol_H = None
                # Crippen contributions (per atom)
                crippen_noH = rdm._CalcCrippenContribs(mol_noH)
                crippen_H = rdm._CalcCrippenContribs(mol_H) if mol_H is not None else None
                # TPSA per-atom (RDKit gives heavy-only). Build both variants.
                try:
                    tpsa_noH = rdm._CalcTPSAContribs(mol_noH)
                except Exception:
                    tpsa_noH = None
                if mol_H is not None and tpsa_noH is not None:
                    # Hydrogens appended at end in RDKit; build vector of len n_atoms_H with zeros for H
                    n_noH = mol_noH.GetNumAtoms()
                    n_H = mol_H.GetNumAtoms()
                    tpsa_H = [0.0] * n_H
                    for i in range(min(n_noH, len(tpsa_noH))):
                        tpsa_H[i] = float(tpsa_noH[i])
                else:
                    tpsa_H = None
                # EState indices (RDKit supports both)
                estate_noH = None
                estate_H = None
                try:
                    estate_noH = list(rdEState.EStateIndices(mol_noH))
                except Exception:
                    try:
                        estate_noH = list(rdm._CalcEStateIndices(mol_noH))
                    except Exception:
                        estate_noH = None
                if mol_H is not None:
                    try:
                        estate_H = list(rdEState.EStateIndices(mol_H))
                    except Exception:
                        try:
                            estate_H = list(rdm._CalcEStateIndices(mol_H))
                        except Exception:
                            estate_H = None
                # LabuteASA per-atom (robust across RDKit versions)
                def _labute_contribs(mol, includeHs=False):
                    try:
                        tot, per_atom = rdm.CalcLabuteASA(mol, includeHs=includeHs)
                        return [float(v) for v in per_atom]
                    except Exception:
                        try:
                            vec = rdm._CalcLabuteASAContribs(mol)
                            return [float(v) for v in vec]
                        except Exception:
                            return [0.0] * mol.GetNumAtoms()

                labute_noH = _labute_contribs(mol_noH, includeHs=False)
                labute_H = _labute_contribs(mol_H, includeHs=True) if mol_H is not None else None
                # MolWt per-atom
                try:
                    pt = Chem.GetPeriodicTable()
                    mw_noH = [float(pt.GetAtomicWeight(a.GetAtomicNum())) for a in mol_noH.GetAtoms()]
                except Exception:
                    mw_noH = None
                if mol_H is not None:
                    try:
                        pt = Chem.GetPeriodicTable()
                        mw_H = [float(pt.GetAtomicWeight(a.GetAtomicNum())) for a in mol_H.GetAtoms()]
                    except Exception:
                        mw_H = None
                # Gasteiger partial charges
                def _gasteiger_list(mol):
                    if mol is None:
                        return None
                    try:
                        if rdPC is not None:
                            rdPC.ComputeGasteigerCharges(mol)
                        else:
                            from rdkit.Chem import AllChem as _AllChem
                            _AllChem.ComputeGasteigerCharges(mol)
                        vals = []
                        for a in mol.GetAtoms():
                            try:
                                if a.HasProp('_GasteigerCharge'):
                                    vals.append(float(a.GetProp('_GasteigerCharge')))
                                else:
                                    vals.append(float('nan'))
                            except Exception:
                                try:
                                    vals.append(float(a.GetDoubleProp('_GasteigerCharge')))
                                except Exception:
                                    vals.append(float('nan'))
                        return vals
                    except Exception:
                        return None

                gasteiger_noH = _gasteiger_list(mol_noH)
                gasteiger_H = _gasteiger_list(mol_H) if mol_H is not None else None

                payload = {
                    "n_atoms_noH": int(mol_noH.GetNumAtoms()),
                    "n_atoms_H": int(mol_H.GetNumAtoms()) if mol_H is not None else None,
                    # Crippen (MolLogP, MolMR)
                    "MolLogP_noH": [float(c[0]) for c in crippen_noH],
                    "MolMR_noH": [float(c[1]) for c in crippen_noH],
                    "MolLogP_H": ([float(c[0]) for c in crippen_H] if crippen_H is not None else None),
                    "MolMR_H": ([float(c[1]) for c in crippen_H] if crippen_H is not None else None),
                    # TPSA
                    "TPSA_noH": ([float(v) for v in tpsa_noH] if tpsa_noH is not None else None),
                    "TPSA_H": (tpsa_H if tpsa_H is not None else None),
                    # EState
                    "EState_noH": estate_noH,
                    "EState_H": estate_H,
                    # LabuteASA
                    "LabuteASA_noH": labute_noH,
                    "LabuteASA_H": labute_H,
                    # MolWt
                    "MolWt_noH": mw_noH,
                    "MolWt_H": mw_H,
                    # Gasteiger partial charges
                    "Gasteiger_noH": gasteiger_noH,
                    "Gasteiger_H": gasteiger_H
                }
                _cache_put(smi, payload, cache_dir)
            # payload loaded
            # Patch missing LabuteASA fields if absent in cache (older RDKit builds)
            try:
                need_fix_labute = (
                    ('LabuteASA_noH' not in payload) or (payload.get('LabuteASA_noH') is None) or
                    ('LabuteASA_H' not in payload) or (payload.get('LabuteASA_H') is None)
                )
            except Exception:
                need_fix_labute = False
            if need_fix_labute:
                try:
                    mol_noH = Chem.MolFromSmiles(smi)
                    mol_H = Chem.AddHs(mol_noH) if mol_noH is not None else None
                    def _labute_contribs(mol, includeHs=False):
                        try:
                            tot, per_atom = rdm.CalcLabuteASA(mol, includeHs=includeHs)
                            return [float(v) for v in per_atom]
                        except Exception:
                            try:
                                vec = rdm._CalcLabuteASAContribs(mol)
                                return [float(v) for v in vec]
                            except Exception:
                                return [0.0] * (mol.GetNumAtoms() if mol is not None else 0)
                    payload['LabuteASA_noH'] = _labute_contribs(mol_noH, includeHs=False) if mol_noH is not None else None
                    payload['LabuteASA_H'] = _labute_contribs(mol_H, includeHs=True) if mol_H is not None else None
                    _cache_put(smi, payload, cache_dir)
                except Exception:
                    pass
            if payload.get("skip", False):
                continue
            # Choose noH/H variant based on node count match
            n_atoms_noH = payload.get("n_atoms_noH", None)
            n_atoms_H = payload.get("n_atoms_H", None)

            # Build per-atom vector for requested names
            # Map contributions by descriptor
            # Note: length must match number of nodes in this graph
            n_local = int(graph_indices[g].numel())
            use_H = False
            if n_atoms_noH == n_local:
                use_H = False
            elif (n_atoms_H is not None) and (n_atoms_H == n_local):
                use_H = True
            else:
                valid[g] = False
                n_atoms_mismatch += 1
                if len(examples) < 3:
                    examples.append({'n_local': int(n_local), 'n_atoms_noH': int(n_atoms_noH or -1), 'n_atoms_H': int(n_atoms_H or -1)})
                continue
            contrib_mat = torch.zeros((n_local, D), dtype=torch.float32)
            # Track per-dim availability for this graph
            dim_avail = torch.zeros((D,), dtype=torch.bool)
            requested_missing = False
            tpsa_requested = any(n == 'TPSA' for n in names)
            for j, name in enumerate(names):
                key = None
                if name == 'MolLogP':
                    key = 'MolLogP_H' if use_H else 'MolLogP_noH'
                elif name in ('MolMR', 'MR'):
                    key = 'MolMR_H' if use_H else 'MolMR_noH'
                elif name == 'TPSA':
                    key = 'TPSA_H' if use_H else 'TPSA_noH'
                elif name == 'EState':
                    key = 'EState_H' if use_H else 'EState_noH'
                elif name == 'LabuteASA':
                    key = 'LabuteASA_H' if use_H else 'LabuteASA_noH'
                elif name in ('MolWt', 'ExactMolWt'):
                    key = 'MolWt_H' if use_H else 'MolWt_noH'
                elif name == 'Gasteiger':
                    key = 'Gasteiger_H' if use_H else 'Gasteiger_noH'
                if key is not None:
                    vals = payload.get(key, None)
                else:
                    vals = None
                if vals is None:
                    pass
                else:
                    vals_t = torch.tensor(vals, dtype=torch.float32)
                    # If rdkit_idx is provided, remap by index; else assume identity by length match
                    if rki_cpu is not None:
                        local_idx = rki_cpu[graph_indices[g]]  # shape [n_local]
                        # Guard: indices must be within bounds of vals
                        if local_idx.numel() > 0 and int(local_idx.max().item()) < vals_t.numel():
                            mapped = vals_t[local_idx]
                            contrib_mat[:, j] = mapped
                            dim_avail[j] = True
                        else:
                            # out of bounds -> mark unavailable
                            pass
                    else:
                        # No mapping; fall back to length match
                        if len(vals) == n_local:
                            contrib_mat[:, j] = vals_t
                            dim_avail[j] = True

            # Do not skip if some dims missing; compute on available dims only

            # Sum contributions over masked atoms of this graph
            local_mask = mask_cpu[graph_indices[g]]
            if local_mask.any():
                n_probed += 1
                s = contrib_mat[local_mask].sum(dim=0)
                # Assign available dims; keep others as NaN
                s = s.to(device)
                row = delta_star[g]
                row[dim_avail.to(device)] = s[dim_avail.to(device)]
                delta_star[g] = row
                # valid if at least one dim is available
                valid[g] = bool(dim_avail.any().item())
                # accumulate per-dim availability (per-graph)
                dim_avail_counts += dim_avail.long()
            else:
                # If the scheduler produced no masked atoms for this graph, skip supervision
                valid[g] = False
                n_no_mask_nodes += 1
        except Exception:
            # Skip invalid molecules; keep default zeros and valid=False
            continue

    diag = {
        'n_graphs': int(num_graphs),
        'n_valid': int(valid.sum().item()),
        'n_no_smiles': int(n_no_smiles),
        'n_atoms_mismatch': int(n_atoms_mismatch),
        'n_no_mask_nodes': int(n_no_mask_nodes),
        'n_probed': int(n_probed),
        'dim_avail_counts': [int(x) for x in dim_avail_counts.tolist()],
        'dim_names': list(names),
        'examples': examples
    }
    return delta_star, valid, diag


def _compute_batch_abs_targets(smiles_list: List[str], batch_vec: torch.Tensor,
                               names: List[str], device: torch.device,
                               cache_dir: Path = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute absolute descriptor targets per-graph using cached RDKit per-atom contributions.

    Returns:
        abs_targets: [G, D]
        valid_mask: [G]
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors as rdm
        from rdkit.Chem import EState as rdEState
        try:
            from rdkit.Chem import rdPartialCharges as rdPC
        except Exception:
            rdPC = None
    except Exception as e:
        raise ImportError(f"RDKit is required for absolute supervision: {e}")

    num_graphs = int(batch_vec.max().item() + 1) if (isinstance(batch_vec, torch.Tensor) and batch_vec.numel() > 0) else len(smiles_list)
    D = len(names)
    abs_t = torch.full((num_graphs, D), float('nan'), dtype=torch.float32, device=device)
    vmask = torch.zeros((num_graphs,), dtype=torch.bool, device=device)

    for g, smi in enumerate(smiles_list):
        if smi is None:
            continue
        try:
            payload = _cache_get(smi, cache_dir)
            if (payload is None) or ('n_atoms_noH' not in payload):
                # Build and cache once
                mol_noH = Chem.MolFromSmiles(smi)
                if mol_noH is None:
                    _cache_put(smi, {"skip": True}, cache_dir)
                    continue
                try:
                    mol_H = Chem.AddHs(mol_noH)
                except Exception:
                    mol_H = None
                crippen_noH = rdm._CalcCrippenContribs(mol_noH)
                crippen_H = rdm._CalcCrippenContribs(mol_H) if mol_H is not None else None
                try:
                    tpsa_noH = rdm._CalcTPSAContribs(mol_noH)
                except Exception:
                    tpsa_noH = None
                if mol_H is not None and tpsa_noH is not None:
                    n_noH = mol_noH.GetNumAtoms()
                    n_H = mol_H.GetNumAtoms()
                    tpsa_H = [0.0] * n_H
                    for i in range(min(n_noH, len(tpsa_noH))):
                        tpsa_H[i] = float(tpsa_noH[i])
                else:
                    tpsa_H = None
                estate_noH = None
                estate_H = None
                try:
                    estate_noH = list(rdEState.EStateIndices(mol_noH))
                except Exception:
                    try:
                        estate_noH = list(rdm._CalcEStateIndices(mol_noH))
                    except Exception:
                        estate_noH = None
                if mol_H is not None:
                    try:
                        estate_H = list(rdEState.EStateIndices(mol_H))
                    except Exception:
                        try:
                            estate_H = list(rdm._CalcEStateIndices(mol_H))
                        except Exception:
                            estate_H = None
                def _labute_contribs(mol, includeHs=False):
                    try:
                        tot, per_atom = rdm.CalcLabuteASA(mol, includeHs=includeHs)
                        return [float(v) for v in per_atom]
                    except Exception:
                        try:
                            vec = rdm._CalcLabuteASAContribs(mol)
                            return [float(v) for v in vec]
                        except Exception:
                            return [0.0] * mol.GetNumAtoms()

                labute_noH = _labute_contribs(mol_noH, includeHs=False)
                labute_H = _labute_contribs(mol_H, includeHs=True) if mol_H is not None else None
                try:
                    pt = Chem.GetPeriodicTable()
                    mw_noH = [float(pt.GetAtomicWeight(a.GetAtomicNum())) for a in mol_noH.GetAtoms()]
                except Exception:
                    mw_noH = None
                if mol_H is not None:
                    try:
                        pt = Chem.GetPeriodicTable()
                        mw_H = [float(pt.GetAtomicWeight(a.GetAtomicNum())) for a in mol_H.GetAtoms()]
                    except Exception:
                        mw_H = None
                # Gasteiger partial charges
                def _gasteiger_list(mol):
                    if mol is None:
                        return None
                    try:
                        if rdPC is not None:
                            rdPC.ComputeGasteigerCharges(mol)
                        else:
                            from rdkit.Chem import AllChem as _AllChem
                            _AllChem.ComputeGasteigerCharges(mol)
                        vals = []
                        for a in mol.GetAtoms():
                            try:
                                if a.HasProp('_GasteigerCharge'):
                                    vals.append(float(a.GetProp('_GasteigerCharge')))
                                else:
                                    vals.append(float('nan'))
                            except Exception:
                                try:
                                    vals.append(float(a.GetDoubleProp('_GasteigerCharge')))
                                except Exception:
                                    vals.append(float('nan'))
                        return vals
                    except Exception:
                        return None

                gasteiger_noH = _gasteiger_list(mol_noH)
                gasteiger_H = _gasteiger_list(mol_H) if mol_H is not None else None

                payload = {
                    "n_atoms_noH": int(mol_noH.GetNumAtoms()),
                    "n_atoms_H": int(mol_H.GetNumAtoms()) if mol_H is not None else None,
                    "MolLogP_noH": [float(c[0]) for c in crippen_noH],
                    "MolMR_noH": [float(c[1]) for c in crippen_noH],
                    "MolLogP_H": ([float(c[0]) for c in crippen_H] if crippen_H is not None else None),
                    "MolMR_H": ([float(c[1]) for c in crippen_H] if crippen_H is not None else None),
                    "TPSA_noH": ([float(v) for v in tpsa_noH] if tpsa_noH is not None else None),
                    "TPSA_H": (tpsa_H if tpsa_H is not None else None),
                    "EState_noH": estate_noH,
                    "EState_H": estate_H,
                    "LabuteASA_noH": labute_noH,
                    "LabuteASA_H": labute_H,
                    "MolWt_noH": mw_noH,
                    "MolWt_H": mw_H,
                    "Gasteiger_noH": gasteiger_noH,
                    "Gasteiger_H": gasteiger_H
                }
                _cache_put(smi, payload, cache_dir)
            if payload.get("skip", False):
                continue
            n_atoms_noH = payload.get("n_atoms_noH", None)
            n_atoms_H = payload.get("n_atoms_H", None)
            # Prefer matching H/noH variant by available lists; if both available, use_H=True if it exists
            # Since abs target is sum over atoms, variant only matters for alignment with dataset's node count
            # but numerically both should be close for most descriptors.
            # We'll choose H if available, else noH.
            use_H = True if (n_atoms_H is not None) else False
            row = []
            valid_any = False
            for name in names:
                key = None
                if name == 'MolLogP':
                    key = 'MolLogP_H' if use_H else 'MolLogP_noH'
                elif name in ('MolMR', 'MR'):
                    key = 'MolMR_H' if use_H else 'MolMR_noH'
                elif name == 'TPSA':
                    key = 'TPSA_H' if use_H else 'TPSA_noH'
                elif name == 'EState':
                    key = 'EState_H' if use_H else 'EState_noH'
                elif name == 'LabuteASA':
                    key = 'LabuteASA_H' if use_H else 'LabuteASA_noH'
                elif name in ('MolWt', 'ExactMolWt'):
                    key = 'MolWt_H' if use_H else 'MolWt_noH'
                elif name == 'Gasteiger':
                    key = 'Gasteiger_H' if use_H else 'Gasteiger_noH'
                vals = payload.get(key, None) if key is not None else None
                if vals is None:
                    row.append(float('nan'))
                else:
                    try:
                        s = float(np.sum(vals)) if isinstance(vals, list) else float(vals)
                    except Exception:
                        try:
                            import numpy as _np
                            s = float(_np.sum(vals))
                        except Exception:
                            s = float('nan')
                    row.append(s)
                    if not math.isnan(s):
                        valid_any = True
            abs_t[g] = torch.tensor(row, dtype=torch.float32, device=device)
            vmask[g] = bool(valid_any)
        except Exception:
            continue
    return abs_t, vmask
def _descriptor_forward_stable(model: EnhancedHyperGraphMAE, g_emb: torch.Tensor) -> torch.Tensor | None:
    head = getattr(model, 'descriptor_head', None)
    if head is None:
        return None
    was_training = head.training
    head.train(False)
    try:
        return head(g_emb)
    finally:
        head.train(was_training)


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    # Align with train.py numerical settings
    try:
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimize_memory_allocation()
    except Exception:
        pass

    # Load and modify config
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.setdefault('training', {}).setdefault('batch_size', args.batch_size)
        cfg['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg.setdefault('training', {}).setdefault('learning_rate', args.lr)
        cfg['training']['learning_rate'] = args.lr
    cfg = ensure_descriptor_in_config(cfg, args)

    # Prepare experiment
    out_dir = Path(args.output_dir)
    if args.experiment_name is None:
        args.experiment_name = "mae_continue_desc"
    exp = ExperimentLogger(args.experiment_name, out_dir, cfg, minimal_output=False)
    logger = exp.logger
    logger.info("Continuing pretraining with descriptor head")
    logger.info(f"Device: {device}")
    if args.debug_nans:
        logger.warning("Debug-NAN mode enabled: enabling anomaly detection and gradient scans")
        try:
            torch.autograd.set_detect_anomaly(True)
        except Exception:
            pass

    # Load dataset (bonds-only or incremental decided by config/CLI)
    target_types = cfg.get('hypergraph_types', ['bond'])
    use_incremental = args.use_incremental_loader or not (len(target_types) == 1 and target_types[0] == 'bond')
    if use_incremental:
        from src.data.incremental_hypergraph_loader import IncrementalHypergraphLoader
        loader = IncrementalHypergraphLoader(
            bonds_data_dir=args.data_dir,
            config=cfg,
            cache_dir=args.incremental_cache_dir
        )
        ds = loader.load_dataset()
        logger.info(f"Incremental loader enabled with types={cfg.get('hypergraph_types')}")
    else:
        ds = MolecularHypergraphDataset(Path(args.data_dir))
        logger.info("Using bonds-only dataset (MolecularHypergraphDataset)")

    stats = ds.get_stats()
    logger.info(f"Dataset: n={stats['num_graphs']}, feat_dim={stats['feature_dim']}")

    # Fix collate hyperedge dim from sample or config
    try:
        sample = ds[0]
        if hasattr(sample, 'hyperedge_attr') and sample.hyperedge_attr is not None and sample.hyperedge_attr.numel() > 0:
            edge_dim = int(sample.hyperedge_attr.size(1))
        else:
            edge_dim = int(cfg.get('features', {}).get('hyperedge_dim', 1))
    except Exception:
        edge_dim = int(cfg.get('features', {}).get('hyperedge_dim', 1))
    cfg.setdefault('features', {})
    cfg['features']['hyperedge_dim'] = edge_dim
    set_collate_hyperedge_dim(edge_dim)
    logger.info(f"Using hyperedge_dim={edge_dim}")

    # Ensure checkpoints are saved inside this experiment directory
    cfg.setdefault('paths', {})
    cfg['paths']['checkpoint_dir'] = str(exp.experiment_dir / 'checkpoints')

    # Split and loaders
    train_ds, val_ds = split_dataset(ds, train_ratio=1 - cfg['training'].get('val_ratio', 0.2), seed=cfg.get('seed', 42))
    train_loader, val_loader = create_data_loaders(
        train_ds, val_ds,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg.get('data', {}).get('dataloader', {}).get('num_workers', 4),
        pin_memory=cfg.get('data', {}).get('dataloader', {}).get('pin_memory', torch.cuda.is_available())
    )

    # Build model with descriptor head enabled
    model_cfg = cfg['model']
    model = EnhancedHyperGraphMAE(
        in_dim=stats['feature_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        latent_dim=model_cfg['latent_dim'],
        proj_dim=model_cfg['proj_dim'],
        heads=model_cfg['heads'],
        num_layers=model_cfg['num_layers'],
        mask_ratio=model_cfg.get('mask_ratio', 0.7),
        config=cfg
    )
    model.to(device)

    # Replace descriptor head with a linear head for Δ/absolute supervision (simpler, better inductive bias)
    try:
        out_dim = len(cfg.get('descriptor_head', {}).get('names', []))
        if out_dim > 0:
            linear_head = torch.nn.Linear(model_cfg['proj_dim'], out_dim).to(device)
            model.descriptor_head = linear_head
    except Exception:
        pass

    # Load pretrained backbone weights with compatibility filtering
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    loaded_state = ckpt.get('model_state_dict', ckpt)
    cur_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in loaded_state.items():
        # Skip TCC controller and descriptor head to avoid loss_component/name-size drift
        if k.startswith('tcc_controller.') or k.startswith('descriptor_head.'):
            skipped.append((k, 'excluded_by_name'))
            continue
        # Only load keys that exist and match shape
        if k in cur_state and cur_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append((k, 'missing_or_shape_mismatch'))
    incompat = model.load_state_dict(filtered, strict=False)
    mk = getattr(incompat, 'missing_keys', [])
    uk = getattr(incompat, 'unexpected_keys', [])
    logger.info(f"Checkpoint filtered load: kept={len(filtered)}, skipped={len(skipped)}, missing={len(mk)}, unexpected={len(uk)}")

    # SIMPLE DELTA CONTINUE TRAINING LOOP (KISS)
    # Optimizer from model config
    lr = float(cfg.get('training', {}).get('learning_rate', 2e-4))
    wd = float(cfg.get('training', {}).get('weight_decay', 1e-4))
    optim_cfg = model.configure_optimizers(lr=lr, weight_decay=wd)
    optimizer = optim_cfg['optimizer']

    use_amp = bool(cfg.get('training', {}).get('use_amp', True) and torch.cuda.is_available())
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler(enabled=(use_amp and not bf16)) if hasattr(torch.amp, 'GradScaler') else None
    autocast_dtype = torch.bfloat16 if (use_amp and bf16) else (torch.float16 if use_amp else None)

    # Step settings
    total_steps = int(args.steps)
    train_iter = iter(train_loader)

    # Learning rate scheduler (respect config when possible; default to WSD)
    scheduler = None
    sched_cfg = cfg.get('training', {}).get('scheduler', {}) or {}
    sched_type = sched_cfg.get('type', 'WSD')
    if sched_type == 'WSD':
        warmup_ratio = float(sched_cfg.get('warmup_ratio', 0.02))
        stable_ratio = float(sched_cfg.get('stable_ratio', 0.70))
        start_factor = float(sched_cfg.get('start_factor', 0.01))
        min_lr = float(sched_cfg.get('min_lr', 1e-6))
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        stable_steps = max(0, int(total_steps * stable_ratio))
        decay_steps = max(1, total_steps - warmup_steps - stable_steps)
        # Adjust rounding remainder
        rem = total_steps - (warmup_steps + stable_steps + decay_steps)
        if rem != 0:
            decay_steps = max(1, decay_steps + rem)
        from torch.optim.lr_scheduler import LinearLR, ConstantLR, CosineAnnealingLR, SequentialLR
        sched_warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
        sched_stable = ConstantLR(optimizer, factor=1.0, total_iters=stable_steps)
        sched_decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, [sched_warmup, sched_stable, sched_decay], milestones=[warmup_steps, warmup_steps + stable_steps])
        logger.info(f"Scheduler=WSD warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}, min_lr={min_lr}")
    elif sched_type == 'WarmupCosine':
        warmup_ratio = float(sched_cfg.get('warmup_ratio', 0.1))
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda)
        logger.info(f"Scheduler=WarmupCosine warmup={warmup_steps}")
    elif sched_type == 'OneCycleLR':
        pct_start = float(sched_cfg.get('pct_start', 0.1))
        div_factor = float(sched_cfg.get('div_factor', 25.0))
        final_div_factor = float(sched_cfg.get('final_div_factor', 1000.0))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        logger.info(f"Scheduler=OneCycleLR total_steps={total_steps}")

    # Δ-loss settings
    delta_names = cfg['descriptor_head']['names']
    # Build absolute-anchor dimension mask (subset of delta_names)
    raw_anchor = getattr(args, 'abs_anchor_names', None)
    if raw_anchor is None:
        anchor_list = ['MolLogP', 'MolMR', 'TPSA', 'LabuteASA', 'EState']
    else:
        anchor_list = [n.strip() for n in raw_anchor.replace(',', ' ').split() if n.strip()]
    # Normalize aliases to match delta_names normalization
    _norm_map = {
        'MR': 'MolMR',
        'ExactMolWt': 'MolWt',
        'EStateIndices': 'EState',
        'EStateIndex': 'EState',
        'GasteigerCharge': 'Gasteiger',
        'PartialCharge': 'Gasteiger',
        'GasteigerPartialCharge': 'Gasteiger',
        'GasteigerPC': 'Gasteiger'
    }
    anchor_set = set(_norm_map.get(n, n) for n in anchor_list)
    # Intersection with available delta_names
    dims_mask = torch.tensor([(n in anchor_set) for n in delta_names], dtype=torch.bool, device=device)
    max_lambda = float(args.delta_weight_max)
    warmup_ratio = float(args.delta_warmup_ratio)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    huber_beta = float(args.delta_huber_beta)
    huber_loss = torch.nn.SmoothL1Loss(beta=huber_beta)

    # Save interval (CLI has highest priority; default 1000)
    ckpt_every = max(1, int(getattr(args, 'ckpt_every', 1000)))
    log_every = int(cfg.get('training', {}).get('log_every_n_steps', max(1, total_steps // 20)))

    # Training
    model.train()
    global_step = 0
    import time
    from collections import deque
    step_times = deque(maxlen=50)
    
    # Initialize step timer
    step_tic = time.perf_counter()
    # Align with trainer: no anomaly/debug toggles by default
    # Validation settings
    val_cfg = cfg.get('validation', {}) or {}
    val_interval = int(val_cfg.get('interval_steps', max(1, total_steps // 10)))
    val_quick = int(val_cfg.get('quick_batches', 30)) if 'quick_batches' in val_cfg else None
    best_val_total = float('inf')
    best_step = 0
    # Property cache dir for RDKit contributions (use incremental_cache_dir as base)
    prop_cache_dir = Path(args.incremental_cache_dir) / 'smiles_propertycache'
    while global_step < total_steps:
        # Record start time for this step
        step_tic = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device, non_blocking=True)
        # Ensure dtypes
        batch.x = batch.x.float()
        if hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
            batch.hyperedge_attr = batch.hyperedge_attr.float()
        if hasattr(batch, 'hyperedge_index') and batch.hyperedge_index is not None:
            batch.hyperedge_index = batch.hyperedge_index.long()

        # Generate semantic masks (consistent with scheduler)
        mr_hi = float(cfg.get('training', {}).get('mask_ratio_max', getattr(model, 'mask_ratio', 0.7)))
        mr_lo = float(cfg.get('training', {}).get('mask_ratio_min', getattr(model, 'mask_ratio', 0.7)))
        ratio = mask_ratio_at(global_step, total_steps, hi=mr_hi, lo=mr_lo)
        smiles = getattr(batch, 'smiles', None)
        node_mask, edge_mask = model._generate_intelligent_masks(
            batch.x, batch.hyperedge_index, batch.hyperedge_attr,
            global_step=global_step, max_steps=total_steps, smiles=smiles, mask_ratio=ratio
        )
        # Mask stats for logging
        try:
            nodes_total = int(batch.x.size(0))
            nodes_masked = int(node_mask.sum().item()) if node_mask is not None else 0
            nm_ratio = (nodes_masked / max(1, nodes_total))
        except Exception:
            nodes_total, nodes_masked, nm_ratio = 0, 0, 0.0
        try:
            if edge_mask is not None:
                edges_total = int(edge_mask.numel())
                edges_masked = int(edge_mask.sum().item())
            else:
                edges_total, edges_masked = 0, 0
            em_ratio = (edges_masked / max(1, edges_total))
        except Exception:
            edges_total, edges_masked, em_ratio = 0, 0, 0.0
        # Graph count
        try:
            if hasattr(batch, 'num_graphs'):
                num_graphs = int(batch.num_graphs)
            elif hasattr(batch, 'batch') and torch.is_tensor(batch.batch):
                num_graphs = int(batch.batch.max().item() + 1)
            else:
                num_graphs = 1
        except Exception:
            num_graphs = 1

        # Base and counterfactual graph embeddings for Δ̂
        # Ensure Δ-loss can backpropagate: avoid no_grad so descriptor_head and the Δ path keep gradients
        # Base (no masking)
        zeros_nm = torch.zeros_like(node_mask, dtype=torch.bool)
        zeros_em = torch.zeros_like(edge_mask, dtype=torch.bool)
        z_base = _z_enhanced_from_model(
            model,
            batch.x, batch.hyperedge_index, batch.hyperedge_attr,
            zeros_nm, zeros_em, smiles, global_step, total_steps,
            use_amp=use_amp, autocast_dtype=autocast_dtype
        )
        # We'll get z_cf from the training forward (return_z=True) to avoid double forward
        g_base = model._graph_pool_mean(z_base, getattr(batch, 'batch', None))
        s_base = _descriptor_forward_stable(model, g_base)

        # Prepare Δ containers (computed after we obtain s_cf)
        delta_loss_val = torch.tensor(0.0, device=device)
        delta_raw_mae_val = torch.tensor(0.0, device=device)
        delta_rel_mae_val = torch.tensor(0.0, device=device)
        lam = max_lambda * (float(global_step) / warmup_steps) if global_step < warmup_steps else max_lambda
        delta_diag = None

        # Model's self-supervised loss (reconstruction/edge via TCC) and z_cf in one forward
        if use_amp and autocast_dtype is not None:
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True):
                outputs = model(
                    batch.x, batch.hyperedge_index, batch.hyperedge_attr,
                    node_mask, edge_mask,
                    global_step=global_step, max_steps=total_steps,
                    smiles=smiles, batch=getattr(batch, 'batch', None),
                    return_z=True
                )
        else:
            outputs = model(
                batch.x, batch.hyperedge_index, batch.hyperedge_attr,
                node_mask, edge_mask,
                global_step=global_step, max_steps=total_steps,
                smiles=smiles, batch=getattr(batch, 'batch', None),
                return_z=True
            )
        # Parse outputs with optional appended z
        if isinstance(outputs, (tuple, list)):
            model_loss = outputs[0]
            # Last element may be z_cf if return_z=True
            z_cf_from_model = outputs[-1] if torch.is_tensor(outputs[-1]) else None
        else:
            model_loss = outputs
            z_cf_from_model = None
        # Use returned z_cf if available to avoid extra forward
        if z_cf_from_model is not None:
            z_cf = z_cf_from_model
        else:
            # Safety fallback: compute z_cf if not returned (rare)
            z_cf = _z_enhanced_from_model(
                model,
                batch.x, batch.hyperedge_index, batch.hyperedge_attr,
                node_mask, edge_mask, smiles, global_step, total_steps,
                use_amp=use_amp, autocast_dtype=autocast_dtype
            )
        # Now compute s_cf
        g_cf = model._graph_pool_mean(z_cf, getattr(batch, 'batch', None))
        s_cf = _descriptor_forward_stable(model, g_cf)
        # Compute Δ̂ and Δ* (teacher-free) AFTER we have both s_base and s_cf
        abs_loss_val = torch.tensor(0.0, device=device)
        abs_raw_mae_val = torch.tensor(0.0, device=device)
        abs_alpha = float(args.abs_anchor_weight)
        if s_base is not None and s_cf is not None:
            d_hat = s_base - s_cf
            try:
                d_star, valid_mask, delta_diag = _compute_batch_delta_targets(
                    smiles, getattr(batch, 'batch', None), node_mask, delta_names, device,
                    cache_dir=prop_cache_dir, rdkit_idx=getattr(batch, 'rdkit_idx', None)
                )
                if valid_mask.any():
                    vm = valid_mask
                    ds = d_star[vm]  # [G_v, D]
                    dh = d_hat[vm]   # [G_v, D]

                    # Scale normalization: divide both targets and predictions by sqrt(masked_count)
                    # to reduce dominance of subgraph size while preserving signal magnitude
                    try:
                        bvec = getattr(batch, 'batch', None)
                        if bvec is not None and torch.is_tensor(bvec):
                            num_g = int(bvec.max().item() + 1) if bvec.numel() > 0 else ds.size(0)
                            masked_per_graph = torch.bincount(bvec, weights=node_mask.float(), minlength=num_g).to(device)
                            sqrt_k = torch.sqrt(masked_per_graph.clamp_min(1.0))  # [G]
                            scale = sqrt_k[vm].unsqueeze(1)  # [G_v, 1]
                        else:
                            # Fallback single-graph
                            k = float(node_mask.sum().item())
                            scale = torch.full((ds.size(0), 1), math.sqrt(max(1.0, k)), device=device)
                        ds = ds / scale
                        dh = dh / scale
                    except Exception:
                        pass
                    finite = torch.isfinite(ds)
                    if finite.any():
                        count = finite.sum(dim=0)
                        use_dim = count > 0
                        if use_dim.any():
                            ds_zeronan = torch.where(finite, ds, torch.zeros_like(ds))
                            mu_star = ds_zeronan.sum(dim=0) / count.clamp_min(1)
                            centered = ds - mu_star
                            var_num = torch.where(finite, centered ** 2, torch.zeros_like(ds)).sum(dim=0)
                            std_star = torch.sqrt(var_num / count.clamp_min(1))
                            std_eff = std_star.clamp_min(1e-3)
                            ds_n = (ds - mu_star) / std_eff
                            dh_n = (dh - mu_star) / std_eff
                            ds_n = torch.nan_to_num(ds_n, nan=0.0, posinf=0.0, neginf=0.0)
                            dh_n = torch.nan_to_num(dh_n, nan=0.0, posinf=0.0, neginf=0.0)
                            valid_pos = finite & use_dim.unsqueeze(0)
                            if valid_pos.any():
                                diff = torch.abs(dh_n - ds_n)
                                beta = huber_beta
                                l = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
                                delta_loss_val = l[valid_pos].mean()
                                raw_mask = torch.isfinite(ds) & use_dim.unsqueeze(0)
                                if raw_mask.any():
                                    delta_raw_mae_val = torch.abs(dh[raw_mask] - ds[raw_mask]).mean()
                                    denom = torch.clamp(torch.abs(ds), min=1e-2)
                                    delta_rel_mae_val = (torch.abs(dh - ds) / denom)[raw_mask].mean()
                                    # Pearson correlation statistics removed (KISS).
                                else:
                                    delta_rel_mae_val = torch.tensor(0.0, device=device)
            except Exception as e:
                logger.debug(f"Δ* computation skipped for this batch: {e}")

            # Absolute anchor: supervise s_base toward RDKit absolute descriptors (small weight)
            try:
                # Compute absolute targets for all delta dims; then mask to anchor subset
                abs_targets, abs_valid = _compute_batch_abs_targets(
                    smiles, getattr(batch, 'batch', None), delta_names, device, cache_dir=prop_cache_dir
                )
                if abs_valid.any():
                    vm2 = abs_valid
                    sb = s_base[vm2]
                    ya = abs_targets[vm2]
                    # Only anchor on selected dims
                    finite2d = torch.isfinite(ya) & dims_mask.unsqueeze(0)
                    if finite2d.any():
                        count = finite2d.sum(dim=0)
                        use_dim2 = count > 0
                        if use_dim2.any():
                            ya_zn = torch.where(finite2d, ya, torch.zeros_like(ya))
                            mu = ya_zn.sum(dim=0) / count.clamp_min(1)
                            var_num = torch.where(finite2d, (ya - mu) ** 2, torch.zeros_like(ya)).sum(dim=0)
                            std = torch.sqrt(var_num / count.clamp_min(1)).clamp_min(1e-3)
                            ya_n = (ya - mu) / std
                            sb_n = (sb - mu) / std
                            ya_n = torch.nan_to_num(ya_n)
                            sb_n = torch.nan_to_num(sb_n)
                            valid2 = finite2d & use_dim2.unsqueeze(0)
                            if valid2.any():
                                abs_loss_val = F.mse_loss(sb_n[valid2], ya_n[valid2])
                                abs_raw_mae_val = torch.abs(sb[valid2] - ya[valid2]).mean()
            except Exception:
                pass
        # Always keep total_loss with grad to allow backward
        total_loss = model_loss + lam * delta_loss_val + abs_alpha * abs_loss_val

        # Backward & step
        optimizer.zero_grad(set_to_none=True)
        grad_norm = None
        nonfinite_grad_report = None

        def _scan_grads(_model) -> str:
            lines = []
            bad = 0
            total_tensors = 0
            for name, p in _model.named_parameters():
                if p is None or p.grad is None:
                    continue
                g = p.grad
                total_tensors += 1
                if not torch.isfinite(g).all():
                    bad += 1
                    numel = g.numel()
                    badcnt = int((~torch.isfinite(g)).sum().item())
                    try:
                        gnan = int(torch.isnan(g).sum().item())
                        ginf = int((~torch.isfinite(g) & ~torch.isnan(g)).sum().item())
                    except Exception:
                        gnan, ginf = 0, 0
                    try:
                        gmin = float(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).min().item())
                        gmax = float(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).max().item())
                    except Exception:
                        gmin, gmax = float('nan'), float('nan')
                    lines.append(f"grad_nonfinite name={name} bad={badcnt}/{numel} nan={gnan} inf={ginf} range=[{gmin:.3e},{gmax:.3e}]")
                    if len(lines) >= 50:
                        break
            header = f"Non-finite gradients: {bad} of {total_tensors} tensors"
            return "\n".join([header] + lines) if bad > 0 else ""
        if scaler is not None and use_amp and autocast_dtype is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if args.debug_nans:
                nonfinite_grad_report = _scan_grads(model)
            try:
                gnt = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('training', {}).get('gradient_clip_norm', 1.0))
                grad_norm = float(gnt) if hasattr(gnt, 'item') else float(gnt)
            except Exception:
                grad_norm = None
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.debug_nans:
                nonfinite_grad_report = _scan_grads(model)
            try:
                gnt = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('training', {}).get('gradient_clip_norm', 1.0))
                grad_norm = float(gnt) if hasattr(gnt, 'item') else float(gnt)
            except Exception:
                grad_norm = None
            optimizer.step()

        if args.debug_nans and nonfinite_grad_report:
            # Enrich with step context and TCC internals if present, then abort to surface early
            logger.error(nonfinite_grad_report)
            try:
                cur_lr = optimizer.param_groups[0]['lr']
                logger.error(f"Context: step={global_step}, lr={cur_lr:.2e}, mr={ratio:.3f}, nodes_masked={nodes_masked}, edges_masked={edges_masked}")
            except Exception:
                pass
            try:
                if isinstance(outputs, (tuple, list)) and len(outputs) >= 6 and isinstance(outputs[5], dict):
                    tcc_info = outputs[5]
                    rl = tcc_info.get('raw_losses', {})
                    nl = tcc_info.get('normalized_losses', {})
                    es = tcc_info.get('ema_scales', {})
                    logger.error(f"TCC raw_losses={rl}")
                    logger.error(f"TCC normalized_losses={nl}")
                    logger.error(f"TCC ema_scales={es}")
            except Exception:
                pass
            raise RuntimeError("Detected non-finite gradients; see logs above for details")

        # Step scheduler if present (per-step schedulers)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                scheduler.step()
            except Exception:
                pass

        # Measure and record elapsed time for this step
        try:
            step_elapsed = time.perf_counter() - step_tic
            step_times.append(step_elapsed)
        except Exception:
            pass

        global_step += 1

        if global_step % log_every == 0 or global_step <= 5:
            try:
                ml = float(model_loss.detach().item())
                dl = float(delta_loss_val.detach().item())
                dl_raw = float(delta_raw_mae_val.detach().item()) if hasattr(delta_raw_mae_val, 'item') else 0.0
            except Exception:
                ml = float('nan'); dl = float('nan'); dl_raw = float('nan')
            try:
                dl_rel = float(delta_rel_mae_val.detach().item())
                abs_l = float(abs_loss_val.detach().item()) if hasattr(abs_loss_val, 'item') else 0.0
                abs_raw = float(abs_raw_mae_val.detach().item()) if hasattr(abs_raw_mae_val, 'item') else 0.0
            except Exception:
                dl_rel = float('nan'); abs_l = float('nan'); abs_raw = float('nan')
            # Extract TCC raw total loss (raw_total) and component losses if available
            model_raw_total = None
            recon_v = None
            edge_v = None
            tcc_weights = None
            tcc_contrib = None
            tcc_ema = None
            try:
                if isinstance(outputs, (tuple, list)):
                    if len(outputs) >= 3:
                        recon_v = float(outputs[1].detach().item()) if hasattr(outputs[1], 'item') else None
                        edge_v = float(outputs[2].detach().item()) if hasattr(outputs[2], 'item') else None
                    if len(outputs) >= 6 and isinstance(outputs[5], dict):
                        tcc_info = outputs[5]
                        if isinstance(tcc_info, dict):
                            model_raw_total = tcc_info.get('raw_total_loss', None)
                            tcc_weights = tcc_info.get('weights', None)
                            tcc_contrib = tcc_info.get('contributions', None)
                            tcc_ema = tcc_info.get('ema_contributions', None)
            except Exception:
                pass
            cur_lr = optimizer.param_groups[0]['lr']
            # timing & throughput
            try:
                step_time = step_times[-1] if len(step_times) > 0 else None
                avg_time = sum(step_times)/len(step_times) if len(step_times)>0 else None
            except Exception:
                step_time, avg_time = None, None
            # Δ diagnostics
            delta_diag_parts = []
            try:
                if 'delta_diag' not in locals():
                    delta_diag = None
                if isinstance(delta_diag, dict):
                    n_probed = delta_diag.get('n_probed', 0)
                    n_valid = delta_diag.get('n_valid', 0)
                    n_no_smiles = delta_diag.get('n_no_smiles', 0)
                    n_atoms_mismatch = delta_diag.get('n_atoms_mismatch', 0)
                    n_no_mask_nodes = delta_diag.get('n_no_mask_nodes', 0)
                    dim_counts = delta_diag.get('dim_avail_counts', [])
                    dim_names = delta_diag.get('dim_names', [])
                    examples = delta_diag.get('examples', [])
                    if n_probed > 0:
                        delta_diag_parts.append(f"Δvalid={n_valid}/{n_probed}")
                    if (n_no_smiles + n_atoms_mismatch + n_no_mask_nodes) > 0:
                        delta_diag_parts.append(
                            f"Δmiss[smiles]={n_no_smiles},[atoms]={n_atoms_mismatch},[mask0]={n_no_mask_nodes}"
                        )
                    # Removed Δdims percentage logging to reduce verbosity
                    # Show one mismatch example
                    if examples:
                        ex = examples[0]
                        delta_diag_parts.append(
                            f"Δex(n={ex.get('n_local')},noH={ex.get('n_atoms_noH')},H={ex.get('n_atoms_H')})"
                        )
            except Exception:
                pass
            # Assemble available log fragments (trainer-style detail)
            parts = [
                f"Step {global_step}/{total_steps}",
                f"lr={cur_lr:.2e}",
                f"G={num_graphs}",
                f"mr={ratio:.3f}",
                f"mask(nodes={nodes_masked}/{nodes_total}={nm_ratio:.1%}, edges={edges_masked}/{edges_total}={em_ratio:.1%})",
                f"loss={ml + lam*dl:.4f}",
                f"model={ml:.4f}"
            ]
            if step_time is not None:
                try:
                    parts.append(f"t={step_time:.3f}s")
                    if avg_time is not None:
                        parts.append(f"t_avg={avg_time:.3f}s")
                    if step_time>0 and num_graphs>0:
                        parts.append(f"g/s={num_graphs/step_time:.1f}")
                except Exception:
                    pass
            if model_raw_total is not None:
                try:
                    parts.append(f"model_raw={float(model_raw_total):.4f}")
                except Exception:
                    pass
            if recon_v is not None and edge_v is not None:
                parts.append(f"recon={recon_v:.4f}")
                parts.append(f"edge={edge_v:.4f}")
            # TCC weights/contributions (condensed)
            if isinstance(tcc_weights, dict):
                try:
                    w_str = ",".join([f"{k[:3]}:{float(v):.2f}" for k, v in tcc_weights.items()])
                    parts.append(f"w=[{w_str}]")
                except Exception:
                    pass
            if isinstance(tcc_contrib, dict):
                try:
                    c_str = ",".join([f"{k[:3]}:{float(v):.2f}" for k, v in tcc_contrib.items()])
                    parts.append(f"c=[{c_str}]")
                except Exception:
                    pass
            if grad_norm is not None:
                parts.append(f"grad={grad_norm:.3f}")
            # Attach Δ diagnostics
            if delta_diag_parts:
                parts.append(" ".join(delta_diag_parts))
            parts.append(f"delta={dl:.4f}")
            parts.append(f"delta_raw_mae={dl_raw:.4f}")
            parts.append(f"delta_rel_mae={dl_rel:.4f}")
            parts.append(f"λ={lam:.3f}")
            try:
                parts.append(f"λ·delta={(lam*dl):.4f}")
            except Exception:
                pass
            try:
                parts.append(f"abs={abs_l:.4f}")
                parts.append(f"abs_raw_mae={abs_raw:.4f}")
                parts.append(f"α={abs_alpha:.3f}")
            except Exception:
                pass
            logger.info(" | ".join(parts))
            if args.debug_nans and isinstance(outputs, (tuple, list)) and len(outputs) >= 6 and isinstance(outputs[5], dict):
                tcc_info = outputs[5]
                try:
                    rl = tcc_info.get('raw_losses', {})
                    nl = tcc_info.get('normalized_losses', {})
                    es = tcc_info.get('ema_scales', {})
                    logger.info(f"TCC[raw]={rl} | TCC[norm]={nl} | TCC[scales]={es}")
                except Exception:
                    pass

        if global_step % ckpt_every == 0:
            ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"checkpoint_step_{global_step}.pth"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg
            }, str(path))
            logger.info(f"Saved checkpoint: {path}")

        # Lightweight validation
        if (global_step % val_interval == 0) and (val_loader is not None):
            model.eval()
            with torch.no_grad():
                val_model_total = 0.0
                val_delta_total = 0.0
                n_batches = 0
                for b_idx, vbatch in enumerate(val_loader):
                    if val_quick is not None and n_batches >= val_quick:
                        break
                    vbatch = vbatch.to(device, non_blocking=True)
                    vbatch.x = vbatch.x.float()
                    if hasattr(vbatch, 'hyperedge_attr') and vbatch.hyperedge_attr is not None:
                        vbatch.hyperedge_attr = vbatch.hyperedge_attr.float()
                    if hasattr(vbatch, 'hyperedge_index') and vbatch.hyperedge_index is not None:
                        vbatch.hyperedge_index = vbatch.hyperedge_index.long()
                    # Same mask scheduler
                    vr = mask_ratio_at(global_step, total_steps, hi=mr_hi, lo=mr_lo)
                    v_smiles = getattr(vbatch, 'smiles', None)
                    v_node_mask, v_edge_mask = model._generate_intelligent_masks(
                        vbatch.x, vbatch.hyperedge_index, vbatch.hyperedge_attr,
                        global_step=global_step, max_steps=total_steps, smiles=v_smiles, mask_ratio=vr
                    )
                    # Self-supervised loss (model)
                    if use_amp and autocast_dtype is not None:
                        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, cache_enabled=True):
                            v_outputs = model(
                                vbatch.x, vbatch.hyperedge_index, vbatch.hyperedge_attr,
                                v_node_mask, v_edge_mask,
                                global_step=global_step, max_steps=total_steps,
                                smiles=v_smiles, batch=getattr(vbatch, 'batch', None)
                            )
                    else:
                        v_outputs = model(
                            vbatch.x, vbatch.hyperedge_index, vbatch.hyperedge_attr,
                            v_node_mask, v_edge_mask,
                            global_step=global_step, max_steps=total_steps,
                            smiles=v_smiles, batch=getattr(vbatch, 'batch', None)
                        )
                    v_model_loss = v_outputs[0] if isinstance(v_outputs, (tuple, list)) else v_outputs
                    val_model_total += float(v_model_loss.detach().item())
                    # Δ validation (teacher-free)
                    z_base_v = _z_enhanced_from_model(
                        model,
                        vbatch.x, vbatch.hyperedge_index, vbatch.hyperedge_attr,
                        torch.zeros_like(v_node_mask, dtype=torch.bool), torch.zeros_like(v_edge_mask, dtype=torch.bool),
                        v_smiles, global_step, total_steps,
                        use_amp=use_amp, autocast_dtype=autocast_dtype
                    )
                    z_cf_v = _z_enhanced_from_model(
                        model,
                        vbatch.x, vbatch.hyperedge_index, vbatch.hyperedge_attr,
                        v_node_mask, v_edge_mask,
                        v_smiles, global_step, total_steps,
                        use_amp=use_amp, autocast_dtype=autocast_dtype
                    )
                    g_base_v = model._graph_pool_mean(z_base_v, getattr(vbatch, 'batch', None))
                    g_cf_v = model._graph_pool_mean(z_cf_v, getattr(vbatch, 'batch', None))
                    if model.descriptor_head is not None:
                        s_base_v = _descriptor_forward_stable(model, g_base_v)
                        s_cf_v = _descriptor_forward_stable(model, g_cf_v)
                        d_hat_v = s_base_v - s_cf_v
                        try:
                            d_star_v, vmask, _ = _compute_batch_delta_targets(
                                v_smiles, getattr(vbatch, 'batch', None), v_node_mask,
                                delta_names, device, cache_dir=prop_cache_dir,
                                rdkit_idx=getattr(vbatch, 'rdkit_idx', None)
                            )
                            if vmask.any():
                                ds = d_star_v[vmask]  # [G_v, D]
                                dh = d_hat_v[vmask]   # [G_v, D]
                                # Apply sqrt(k) scaling normalization during validation
                                try:
                                    bvec = getattr(vbatch, 'batch', None)
                                    if bvec is not None and torch.is_tensor(bvec):
                                        num_g = int(bvec.max().item() + 1) if bvec.numel() > 0 else ds.size(0)
                                        masked_per_graph = torch.bincount(bvec, weights=v_node_mask.float(), minlength=num_g).to(device)
                                        sqrt_k = torch.sqrt(masked_per_graph.clamp_min(1.0))
                                        scale = sqrt_k[vmask].unsqueeze(1)
                                    else:
                                        k = float(v_node_mask.sum().item())
                                        scale = torch.full((ds.size(0), 1), math.sqrt(max(1.0, k)), device=device)
                                    ds = ds / scale
                                    dh = dh / scale
                                except Exception:
                                    pass
                                finite = torch.isfinite(ds)
                                if finite.any():
                                    count = finite.sum(dim=0)
                                    use_dim = count > 0
                                    if use_dim.any():
                                        ds_zeronan = torch.where(finite, ds, torch.zeros_like(ds))
                                        mu = ds_zeronan.sum(dim=0) / count.clamp_min(1)
                                        centered = ds - mu
                                        var_num = torch.where(finite, centered ** 2, torch.zeros_like(ds)).sum(dim=0)
                                        std = torch.sqrt(var_num / count.clamp_min(1)).clamp_min(1e-3)
                                        ds_n = (ds - mu) / std
                                        dh_n = (dh - mu) / std
                                        ds_n = torch.nan_to_num(ds_n, nan=0.0, posinf=0.0, neginf=0.0)
                                        dh_n = torch.nan_to_num(dh_n, nan=0.0, posinf=0.0, neginf=0.0)
                                        valid2d = finite & use_dim.unsqueeze(0)
                                        if valid2d.any():
                                            ds_sel = ds_n[valid2d]
                                            dh_sel = dh_n[valid2d]
                                            v_delta = huber_loss(dh_sel, ds_sel).item()
                                            val_delta_total += v_delta
                                            rel = torch.abs(dh - ds) / torch.clamp(torch.abs(ds), min=1e-2)
                                            v_rel = rel[valid2d].mean().item()
                                # else: skip batch
                        except Exception:
                            pass
                    n_batches += 1
                if n_batches > 0:
                    val_model_avg = val_model_total / n_batches
                    val_delta_avg = val_delta_total / max(1, n_batches)
                    val_total = val_model_avg + lam * val_delta_avg
                    logger.info(f"Validation @step {global_step}: total={val_total:.4f} (model={val_model_avg:.4f}, delta={val_delta_avg:.4f}, λ={lam:.3f})")
                    # Track best and save
                    if val_total < best_val_total:
                        best_val_total = val_total
                        best_step = global_step
                        best_path = Path(cfg['paths']['checkpoint_dir']) / 'best_val_total.pth'
                        best_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': cfg,
                            'val_total': best_val_total
                        }, str(best_path))
                        logger.info(f"Saved best (val_total) checkpoint: {best_path}")
            model.train()

    # Save final checkpoint
    ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"{args.experiment_name}_delta_final.pth"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg
    }, str(final_path))
    logger.info(f"Δ-descriptor fine-tune checkpoint saved to {final_path}")

    # Save simple summary
    artifacts = {
        "final_checkpoint": str(final_path),
        "training_completed": True,
        "total_steps": global_step
    }
    with open(exp.experiment_dir / "artifacts_delta.json", 'w') as f:
        json.dump(artifacts, f, indent=2)


if __name__ == "__main__":
    main()
