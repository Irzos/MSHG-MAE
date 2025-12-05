#!/usr/bin/env python
"""
Main training script for HyperGraph-MAE.
"""

import argparse
import torch
import yaml
from pathlib import Path
import os
import sys
import logging
import random
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.hypergraph_mae import EnhancedHyperGraphMAE
from src.data.data_loader import (
    MolecularHypergraphDataset,
    create_data_loaders,
    split_dataset,
    set_collate_hyperedge_dim,
)
from src.training.trainer import HyperGraphMAETrainer
from src.utils.logging_utils import setup_logger, ExperimentLogger
from src.utils.memory_utils import optimize_memory_allocation, log_memory_usage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HyperGraph-MAE model")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing preprocessed data")
    parser.add_argument("--max_graphs", type=int, default=None,
                        help="Maximum number of graphs to load")

    # Model arguments
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Training arguments
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Output control
    parser.add_argument("--minimal_output", action="store_true",
                        help="Enable minimal output mode (less files)")

    return parser.parse_args()


def _merge_dict(dst: dict, src: dict) -> dict:
    """Deep-merge src into dst (in-place) with simple semantics.
    Dict keys are merged recursively; other types are overwritten by src.
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def _dotlist_to_nested(d: dict) -> dict:
    """Convert dot.keys dict to nested dict: {'a.b.c': 1} -> {'a': {'b': {'c': 1}}} """
    result = {}
    for key, value in (d or {}).items():
        parts = key.split('.') if isinstance(key, str) else [key]
        cur = result
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return result


def load_config(config_path: str, args) -> dict:
    """Load and update configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    config['seed'] = args.seed

    # First ensure dataloader defaults exist to avoid KeyError
    config.setdefault('data', {}).setdefault('dataloader', {})
    config['data']['dataloader'].setdefault('pin_memory', torch.cuda.is_available())
    # Then override num_workers from CLI
    config['data']['dataloader']['num_workers'] = args.num_workers

    # Merge overrides from environment (Hydra runner compatibility)
    # Expect YAML string in HYPERGRAPH_MAE_OVERRIDES; support both nested and dot-keys
    env_yaml = os.environ.get("HYPERGRAPH_MAE_OVERRIDES", "").strip()
    if env_yaml:
        try:
            overrides_raw = yaml.safe_load(env_yaml) or {}
            # If all keys look like dot-notation, convert to nested
            if isinstance(overrides_raw, dict):
                has_dot_keys = any(isinstance(k, str) and '.' in k for k in overrides_raw.keys())
                overrides = _dotlist_to_nested(overrides_raw) if has_dot_keys else overrides_raw
                config = _merge_dict(config, overrides)
        except Exception as e:
            print(f"Warning: failed to parse HYPERGRAPH_MAE_OVERRIDES: {e}")

    # Validate training section and max_steps (Single Source of Truth)
    if 'training' not in config or not isinstance(config['training'], dict):
        raise ValueError("Invalid config: missing 'training' section")
    if 'max_steps' not in config['training'] or config['training']['max_steps'] is None:
        # If legacy 'epochs' exists, prefer explicit failure with guidance
        if 'epochs' in config['training']:
            raise ValueError(
                "Config requires training.max_steps (step-based training). "
                "Found legacy 'training.epochs'. Please set 'training.max_steps' or pass --max_steps."
            )
        else:
            raise ValueError(
                "Config requires training.max_steps (step-based training). "
                "Please set it in the config or pass --max_steps."
            )

    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior if needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main(args=None, return_history=False, optuna_trial=None):
    """
    Main training function.
    
    Args:
        args: Training arguments
        return_history: Whether to return training history
        optuna_trial: Optional Optuna trial for pruning
    """
    if args is None:
        args = parse_args()

    # Set up experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"hypergraph_mae_{timestamp}"

    # Load configuration
    config = load_config(args.config, args)

    # Set random seed
    set_seed(config['seed'])

    # Set up experiment logger
    output_dir = Path(args.output_dir)
    experiment_logger = ExperimentLogger(
        args.experiment_name,
        output_dir,
        config,
        minimal_output=getattr(args, 'minimal_output', False)
    )

    # Ensure checkpoints are saved inside this experiment's directory
    # so they appear alongside other run artifacts
    config.setdefault('paths', {})
    config['paths']['checkpoint_dir'] = str(experiment_logger.experiment_dir / 'checkpoints')

    # Set up logger
    logger = experiment_logger.logger
    logger.info("Starting HyperGraph-MAE training")
    logger.info(f"Experiment: {args.experiment_name}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Optimize memory allocation
    if device.type == 'cuda':
        optimize_memory_allocation()

    # Set matrix multiplication precision for better performance
    torch.set_float32_matmul_precision('medium')
    logger.info("Set float32 matrix multiplication precision to 'medium' for improved performance")
    
    # Enable TF32 acceleration (A100/H100; compatible fallback)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 acceleration enabled for CUDA matmul and cuDNN operations")
    else:
        logger.info("CUDA not available, TF32 acceleration skipped")

    # Log initial memory state (commented out to avoid early resource usage)
    # log_memory_usage("Initial: ")

    # Load dataset with incremental hypergraph support
    logger.info(f"Loading dataset from {args.data_dir}")
    
    # Check if we need incremental hypergraph construction
    target_types = config.get('hypergraph_types', ['bond'])
    if len(target_types) == 1 and target_types[0] == 'bond':
        # Only bonds needed, use original loader
        logger.info("Using bonds-only data (original loader)")
        dataset = MolecularHypergraphDataset(
            data_dir=args.data_dir,
            max_graphs=args.max_graphs
        )
    else:
        # Need incremental hypergraph construction
        logger.info(f"Using incremental hypergraph construction for types: {target_types}")
        from src.data.incremental_hypergraph_loader import IncrementalHypergraphLoader
        loader = IncrementalHypergraphLoader(
            bonds_data_dir=args.data_dir,
            config=config,
            cache_dir="hydra/version2/cache"
        )
        dataset = loader.load_dataset()
        
        # Apply max_graphs limit if specified
        if args.max_graphs is not None and len(dataset.graphs) > args.max_graphs:
            logger.info(f"Limiting dataset to {args.max_graphs} graphs")
            dataset.graphs = dataset.graphs[:args.max_graphs]

    # Get dataset statistics
    stats = dataset.get_stats()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Number of graphs: {stats['num_graphs']}")
    logger.info(f"  Average nodes: {stats['avg_nodes']:.2f} ± {stats['std_nodes']:.2f}")
    logger.info(f"  Average edges: {stats['avg_edges']:.2f} ± {stats['std_edges']:.2f}")
    logger.info(f"  Feature dimension: {stats['feature_dim']}")

    # Split dataset
    train_dataset, val_dataset = split_dataset(
        dataset,
        train_ratio=1 - config['training'].get('val_ratio', 0.2),
        seed=config['seed']
    )

    # Detect and fix a global hyperedge feature dimension for collate and model
    try:
        sample_graph = dataset[0]
        if hasattr(sample_graph, 'hyperedge_attr') and sample_graph.hyperedge_attr is not None and sample_graph.hyperedge_attr.numel() > 0:
            actual_edge_dim = int(sample_graph.hyperedge_attr.size(1))
        else:
            actual_edge_dim = int(config.get('features', {}).get('hyperedge_dim', 1))
    except Exception:
        actual_edge_dim = int(config.get('features', {}).get('hyperedge_dim', 1))

    config.setdefault('features', {})
    config['features']['hyperedge_dim'] = actual_edge_dim
    set_collate_hyperedge_dim(actual_edge_dim)
    logger.info(f"Using hyperedge_dim={actual_edge_dim} for collate and model")

    # Set PyTorch multiprocessing sharing strategy to reduce file descriptor pressure
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy('file_system')
        logger.info("Set PyTorch multiprocessing sharing strategy to 'file_system'")
    except RuntimeError as e:
        logger.warning(f"Failed to set sharing strategy: {e}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['dataloader']['num_workers'],
        pin_memory=config['data']['dataloader']['pin_memory']
    )

    # Initialize model
    logger.info("Initializing model")
    
    # Get original model config
    model_config = config['model'].copy()  # Make a copy to avoid modifying original
    
    # If not resuming from checkpoint, validate model dimensions
    if not args.checkpoint:
        feature_dim = stats['feature_dim']
        logger.info(f"Dataset feature dimension: {feature_dim}")
        
        # Keep model architecture dimensions as configured - don't reduce them
        # Only ensure proj_dim is divisible by heads for attention mechanism
        original_heads = model_config['heads']
        original_proj_dim = model_config['proj_dim']
        original_hidden_dim = model_config['hidden_dim']
        
        # Ensure proj_dim is divisible by heads (required for multi-head attention)
        if original_proj_dim % original_heads != 0:
            # Round up to nearest multiple of heads
            proj_dim = ((original_proj_dim + original_heads - 1) // original_heads) * original_heads
            model_config['proj_dim'] = proj_dim
            logger.info(f"Adjusted proj_dim from {original_proj_dim} to {proj_dim} to be divisible by {original_heads} heads")
        
        # Keep hidden_dim as configured - don't reduce to feature_dim
        # The model should have capacity to learn complex representations
        logger.info(f"Keeping model architecture: hidden_dim={original_hidden_dim}, proj_dim={model_config['proj_dim']}, heads={original_heads}")
        
        # Add feature dimension adaptation in the model's input layer instead
        model_config['feature_dim'] = feature_dim  # Pass to model for input adaptation

    # Create updated config and ensure it is used everywhere
    updated_config = config.copy()
    updated_config['model'] = model_config
    config = updated_config  # Overwrite for consistency (trainer & checkpoints)
    # Resave config so ExperimentLogger has the final version
    experiment_logger.save_config(config)

    model = EnhancedHyperGraphMAE(
        in_dim=stats['feature_dim'],
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        proj_dim=model_config['proj_dim'],
        heads=model_config['heads'],
        num_layers=model_config['num_layers'],
        mask_ratio=model_config['mask_ratio'],
        config=config
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize trainer
    trainer = HyperGraphMAETrainer(model, config, device, optuna_trial=optuna_trial)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train model
    logger.info("Starting training")
    
    # Disable anomaly detection for production training (only enable for debugging)
    # torch.autograd.set_detect_anomaly(True)
    
    try:
        history = trainer.train(
            train_loader,
            val_loader,
            max_steps=config['training']['max_steps']
        )

        # Save final checkpoint to checkpoint_dir (alongside best_model.pth)
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        if getattr(args, 'minimal_output', False):
            # Simplified mode: use <experiment_name>_checkpoint.pth
            final_checkpoint_path = checkpoint_dir / f"{args.experiment_name}_checkpoint.pth"
        else:
            # Full mode: save as final_checkpoint.pth
            final_checkpoint_path = checkpoint_dir / "final_checkpoint.pth"
        trainer.save_checkpoint(str(final_checkpoint_path))
        logger.info(f"Final checkpoint saved to {final_checkpoint_path}")

        # Generate artifacts.json as a simple standardized training artifact contract
        if not getattr(args, 'minimal_output', False):
            import json
            artifacts = {
                "best_checkpoint": str(Path(config['paths']['checkpoint_dir']) / "best_model.pth"),
                "final_checkpoint": str(Path(config['paths']['checkpoint_dir']) / "final_checkpoint.pth"),
                "best_metric": "val_loss",
                "best_metric_value": getattr(trainer, "best_val_loss", None),
                "best_step": getattr(trainer, "best_step", None),
                "seed": config['seed'],
                "timestamp": datetime.now().isoformat(),
                "experiment_dir": str(experiment_logger.experiment_dir),
                "training_completed": True,
                "total_steps": trainer.global_step
            }
            
            artifacts_path = experiment_logger.experiment_dir / "artifacts.json"
            with open(artifacts_path, 'w') as f:
                json.dump(artifacts, f, indent=2)
            logger.info(f"Training artifacts saved to {artifacts_path}")
            

        # Log concise experiment summary
        if not getattr(args, 'minimal_output', False):
            for phase, metrics in history.items():
                if isinstance(metrics, list) and len(metrics) > 0:
                    experiment_logger.log_metrics(
                        {phase: metrics[-1]},
                        phase='final',
                        step=len(metrics)
                    )
            experiment_logger.save_summary()

        logger.info("Training completed successfully")
        
        if return_history:
            # Ensure history contains total_steps (consistent with error paths)
            history.setdefault('total_steps', trainer.global_step)
            return history

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save interrupted checkpoint
        interrupted_path = experiment_logger.experiment_dir / "interrupted_checkpoint.pth"
        trainer.save_checkpoint(str(interrupted_path))
        logger.info(f"Checkpoint saved to {interrupted_path}")
        
        if return_history:
            # Ensure history contains total_steps
            trainer.history.setdefault('total_steps', trainer.global_step)
            return trainer.history

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if return_history:
            # If training has already started, return existing history instead of an empty one
            if 'trainer' in locals() and hasattr(trainer, 'history') and trainer.history:
                # Ensure returned history contains total_steps
                trainer.history.setdefault('total_steps', getattr(trainer, 'global_step', 0))
                return trainer.history
            else:
                return {'train_loss': [], 'val_loss': [], 'total_steps': 0}
        raise

    finally:
        # Log final memory state (commented out to avoid resource issues)
        # log_memory_usage("Final: ")

        # Save experiment notes
        if 'trainer' in locals() and hasattr(trainer, 'global_step'):
            experiment_logger.log_text(
                f"Training completed at {datetime.now()}\n"
                f"Final step: {trainer.global_step}\n"
                f"Best validation loss: {getattr(trainer, 'best_val_loss', float('inf')):.4f}\n"
            )
        else:
            experiment_logger.log_text(
                f"Training aborted at {datetime.now()} before trainer initialization\n"
            )


if __name__ == "__main__":
    main()
