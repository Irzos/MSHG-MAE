#!/usr/bin/env python3
"""
Precompute unique drug/protein embeddings to avoid repeated encoding.

Pipeline:
- Read a CSV with columns for SMILES and protein sequence
- Deduplicate to two sets: unique smiles and unique protein sequences
- Compute embeddings once per unique item
- Save as Parquet files:
  * drug_emb.parquet: columns ['smiles', 'emb_0', ..., 'emb_{D-1}']
  * prot_emb.parquet: columns ['protein', 'emb_0', ..., 'emb_{H-1}']

KISS: batching only for proteins (transformers), drugs per-SMILES
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import sys
import pandas as pd
import numpy as np
import torch

# Make project src importable
sys.path.append(str(Path(__file__).parent.parent))

from dti_e2e_predict import (
    load_hg_model_robust,
    build_identity_global_stats,
)
from src.data.hypergraph_construction import smiles_to_hypergraph


def parse_args():
    p = argparse.ArgumentParser(description='Precompute unique drug/protein embeddings')
    p.add_argument('--csv', type=str, required=True, help='Input CSV with at least SMILES and protein columns')
    p.add_argument('--smiles_col', type=str, default='smiles')
    p.add_argument('--sequence_col', type=str, default='protein')
    p.add_argument('--hg_ckpt', type=str, required=True)
    p.add_argument('--hg_config', type=str, required=True)
    p.add_argument('--protbert_model', type=str, required=True)
    p.add_argument('--device', type=str, default=None, help='cuda or cpu (default: auto)')
    p.add_argument('--out_drug', type=str, required=True, help='Output Parquet for drug embeddings')
    p.add_argument('--out_prot', type=str, required=True, help='Output Parquet for protein embeddings')
    # Sharding for parallel preprocessing
    p.add_argument('--num_shards', type=int, default=1, help='Number of shards for parallel run (>=1)')
    p.add_argument('--shard_index', type=int, default=0, help='Current shard index [0..num_shards-1]')
    p.add_argument('--batch_size', type=int, default=16, help='Protein batch size')
    p.add_argument('--max_length', type=int, default=1024, help='Protein tokenizer max_length')
    p.add_argument('--pool', type=str, default='mean', choices=['mean','max','sum'])
    p.add_argument('--no_norm', action='store_true', help='Disable L2 normalization on drug embedding')
    p.add_argument('--progress', action='store_true', help='Show progress bars')
    p.add_argument('--timeout_seconds', type=int, default=None, help='Override hypergraph construction timeout (seconds)')
    return p.parse_args()


@torch.no_grad()
def batch_encode_proteins(seqs: list[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    """Return (N,H) float32 embeddings with mean pooling."""
    H = None
    out_list = []
    rng = range(0, len(seqs), batch_size)
    iterator = rng
    if args.progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(rng, desc='ProtBert encode', total=(len(seqs)+batch_size-1)//batch_size)
        except Exception:
            pass
    for i in iterator:
        batch = [s.strip().upper().replace('U','X').replace('Z','X').replace('O','X') for s in seqs[i:i+batch_size]]
        spaced = [' '.join(list(s)) for s in batch]
        enc = tokenizer(spaced, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state  # [B,L,H]
        mask = enc.get('attention_mask', torch.ones_like(last_hidden[...,0]))  # [B,L]
        mask = mask.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        emb = summed/denom  # [B,H]
        if H is None:
            H = int(emb.shape[1])
        out_list.append(emb.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(out_list, axis=0)


def main(args):
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CSV and deduplicate keys
    df = pd.read_csv(args.csv, usecols=[args.smiles_col, args.sequence_col])
    smiles_unique = df[args.smiles_col].dropna().astype(str).drop_duplicates().tolist()
    prot_unique = df[args.sequence_col].dropna().astype(str).drop_duplicates().tolist()

    # Apply content-based sharding for balanced work across processes
    if args.num_shards and args.num_shards > 1:
        def belongs(s: str) -> bool:
            h = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
            return (h % args.num_shards) == int(args.shard_index)
        smiles_unique = [s for s in smiles_unique if belongs(s)]
        prot_unique = [s for s in prot_unique if belongs(s)]

    # Load models
    hg_model, hg_config = load_hg_model_robust(args.hg_ckpt, args.hg_config, device)
    if getattr(args, 'timeout_seconds', None) is not None:
        try:
            hg_config['timeout_seconds'] = int(args.timeout_seconds)
        except Exception:
            pass
    from transformers import AutoTokenizer, AutoModel
    mpath = Path(args.protbert_model)
    if not mpath.exists():
        raise FileNotFoundError(f"ProtBert path not found: {mpath}. Use a local directory containing tokenizer_config.json and model weights.")
    tokenizer = AutoTokenizer.from_pretrained(str(mpath), do_lower_case=False, local_files_only=True)
    prot_model = AutoModel.from_pretrained(str(mpath), local_files_only=True).to(device).eval()

    # Prepare identity global stats to avoid per-molecule std warnings and keep consistency
    id_stats = build_identity_global_stats()

    # Drug embeddings (per-unique)
    d_embs = []
    iterator = enumerate(smiles_unique)
    if args.progress:
        try:
            from tqdm import tqdm
            iterator = enumerate(tqdm(smiles_unique, desc='Drug encode'))
        except Exception:
            pass
    for idx, smi in iterator:
        try:
            smi_s = str(smi)
            # Build hypergraph with identity stats
            data = smiles_to_hypergraph(smiles=smi_s, mol_id=str(idx), config=hg_config, global_stats=id_stats, device=device)
            if data is None:
                d_embs.append(None)
                continue
            hg_model.eval().to(device)
            with torch.no_grad():
                x = data.x.to(device)
                he_idx = data.hyperedge_index.to(device)
                he_attr = getattr(data, 'hyperedge_attr', None)
                if he_attr is not None:
                    he_attr = he_attr.to(device)
                z = hg_model.get_embedding(x, he_idx, he_attr)
                if args.pool == 'mean':
                    g = z.mean(dim=0)
                elif args.pool == 'max':
                    g, _ = z.max(dim=0)
                elif args.pool == 'sum':
                    g = z.sum(dim=0)
                else:
                    g = z.mean(dim=0)
                if not args.no_norm:
                    g = torch.nn.functional.normalize(g.unsqueeze(0), p=2, dim=1).squeeze(0)
            d_embs.append(g.detach().cpu().numpy().astype(np.float32))
        except Exception:
            d_embs.append(None)
    # Filter failures
    keep_pairs = [(s, v) for s, v in zip(smiles_unique, d_embs) if v is not None]
    if keep_pairs:
        d_keys, d_mat = zip(*keep_pairs)
        D = int(d_mat[0].shape[0])
        drug_df = pd.DataFrame({'smiles': list(d_keys)})
        for j in range(D):
            drug_df[f'emb_{j}'] = [float(vec[j]) for vec in d_mat]
        # Adjust output path with shard suffix to avoid collisions
        out_drug = args.out_drug
        if args.num_shards and args.num_shards > 1:
            p = Path(out_drug)
            out_drug = str(p.with_name(f"{p.stem}.shard{args.shard_index}-of-{args.num_shards}{p.suffix}"))
        Path(Path(out_drug).parent).mkdir(parents=True, exist_ok=True)
        drug_df.to_parquet(out_drug, index=False)
    else:
        raise RuntimeError('No drug embeddings were computed successfully')

    # Protein embeddings (batched)
    p_mat = batch_encode_proteins(prot_unique, tokenizer, prot_model, device, args.batch_size, args.max_length)
    H = int(p_mat.shape[1])
    prot_df = pd.DataFrame({'protein': prot_unique})
    for j in range(H):
        prot_df[f'emb_{j}'] = p_mat[:, j].astype(np.float32)
    out_prot = args.out_prot
    if args.num_shards and args.num_shards > 1:
        p = Path(out_prot)
        out_prot = str(p.with_name(f"{p.stem}.shard{args.shard_index}-of-{args.num_shards}{p.suffix}"))
    Path(Path(out_prot).parent).mkdir(parents=True, exist_ok=True)
    prot_df.to_parquet(out_prot, index=False)

    print({'drug_emb': out_drug, 'prot_emb': out_prot, 'drug_count': len(keep_pairs), 'prot_count': len(prot_unique), 'shard': int(args.shard_index), 'num_shards': int(args.num_shards)})


if __name__ == '__main__':
    args = parse_args()
    main(args)
