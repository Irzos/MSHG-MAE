![Python Version](https://img.shields.io/badge/Python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900)
# MSHG-MAE: Multi-Scale Hypergraph Masked Autoencoder
![Uploading 0237cac1ebec018d0d2171f5dfc43f51.png…]()

This directory contains a compact implementation of a **Multi-Scale Hypergraph Masked Autoencoder with Δ-Property Alignment for Novel Molecular Representation Learning**.  
It includes the core model, data processing pipeline, and training / analysis scripts.

## Project structure

- `src/`
  - `models/`: `EnhancedHyperGraphMAE` backbone, encoder/decoder, and hypergraph convolution layers.
  - `data/`: SMILES → hypergraph construction (`hypergraph_construction.py`), feature engineering, and dataset loading (`data_loader.py`, etc.).
  - `training/`: `HyperGraphMAETrainer`, losses, dynamic loss weighting, masking strategies, and callbacks.
  - `utils/`: logging, memory monitoring, visualization, and evaluation metrics.
- `scripts/`
  - `train.py`: main training script (step-based self-supervised reconstruction + edge prediction).
  - `continue_pretrain_with_descriptors.py`: second-stage pretraining with Δ‑descriptor supervision on top of a self-supervised checkpoint.
  - `convert_deepchem_to_hypergraphs.py`: convert DeepChem MolNet datasets into hypergraph batches (`batch_*.pt`).
  - `preprocess_data.py` / `preprocess_semantic_blocks.py`: preprocessing and semantic block helpers.
  - `precompute_embeddings.py` : precompute molecule / protein embeddings for downstream tasks.
  - `functional_group_clustering_experiment.py`: functional-group–centric clustering and analysis.
- `config/`
  - `default_config.yaml`: example MAX configuration (multi-scale conv + attention + semantic masking + TCC).

## Installation

It is recommended to use a fresh Conda environment (Python 3.9 or 3.10), then install requirements:

```bash
conda create -n hgmae python=3.10
conda activate hgmae

# Install remaining Python dependencies
pip install -r requirements.txt
```

Example tested environment:
- Python 3.10
- PyTorch 2.8.0 (CUDA 12.8 build)
- torch-geometric 2.6.1
- torch-scatter 2.1.2
- RDKit (recent conda build, e.g. 2023+)

Most users should install PyTorch and torch-geometric following their official instructions, then use `requirements.txt` for the rest. Optional plotting / analysis libraries are commented out and can be enabled only when needed.

## Data preparation (hypergraph construction)

### 1. From DeepChem MolNet datasets (optional)

Use `scripts/convert_deepchem_to_hypergraphs.py` to build hypergraphs from MolNet datasets:

```bash
python scripts/convert_deepchem_to_hypergraphs.py \
  --output_root data/hypergraphs_molnet \
  --config config/default_config.yaml \
  --dataset_names BBBP ESOL
```

The script saves `batch_*.pt` files per dataset. Each file is a list of `torch_geometric.data.Data` objects with at least:
- `x`, `hyperedge_index`, `hyperedge_attr`
- `y`, `y_mask` (for supervised tasks)
- `smiles` and other metadata required by the training pipeline.

### 2. From custom SMILES data

Use `scripts/preprocess_data.py` / `scripts/preprocess_semantic_blocks.py` together with
`src/data/hypergraph_construction.py` to convert SMILES into hypergraphs with:
- node features `x`
- hyperedge connectivity `hyperedge_index`
- hyperedge features `hyperedge_attr`
- `smiles` and optional semantic annotations.

The main training script expects `--data_dir` to contain multiple `batch_*.pt` files.

## Training: self-supervised HyperGraph-MAE

Basic training command:

```bash
python scripts/train.py \
  --data_dir path/to/hypergraph_batches \
  --config config/default_config.yaml \
  --output_dir experiments \
  --experiment_name hgmae_max_full \
  --max_steps 50000
```

Key arguments:
- `--data_dir`: directory containing `batch_*.pt`.
- `--config`: YAML config file (you can copy and modify `config/default_config.yaml`).
- `--max_steps`: total number of training steps; if omitted, `training.max_steps` in the config must be set.

During training the script:
- infers and propagates `features.hyperedge_dim` to both collate functions and the model;
- instantiates `EnhancedHyperGraphMAE` with multi-scale hypergraph conv and attention;
- runs step-based training in `HyperGraphMAETrainer` with mixed precision and cosine mask-ratio scheduling;
- writes logs and checkpoints under `output_dir/experiment_name`.

## Stage‑2: Δ‑descriptor continued pretraining

To add RDKit-based Δ‑descriptor supervision (e.g., MolLogP / TPSA) on top of a self-supervised checkpoint:

```bash
python scripts/continue_pretrain_with_descriptors.py \
  --data_dir path/to/hypergraph_batches \
  --config config/default_config.yaml \
  --checkpoint path/to/pretrained_mae.pth \
  --output_dir experiments \
  --experiment_name hgmae_plus_delta \
  --steps 10000 \
  --delta_names "MolLogP MolMR TPSA"
```

This script:
- loads the pre-trained MAE backbone weights;
- enables and trains the `descriptor_head`, computing Δ* targets from RDKit atomic/fragment contributions;
- optimizes the original reconstruction/edge losses together with the Δ‑loss (and optional absolute anchor loss), producing more property-aware molecular fingerprints.

## Other useful scripts
- `scripts/precompute_embeddings.py`: jointly precompute protein + molecule embeddings (e.g., with ProtBert).
- `scripts/functional_group_clustering_experiment.py`: functional-group / semantic-block clustering and statistical analysis.

## Core ideas

The project centers on:
- `EnhancedHyperGraphMAE` with multi-scale hypergraph convolution and attention-based aggregation;
- semantic masking + self-supervised reconstruction and hyperedge prediction;
- TCC-based dynamic loss balancing and a robust step-based training loop;
- an Δ‑descriptor continued pretraining stage ( Δ-Property Alignment) that injects RDKit-derived property signal.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
