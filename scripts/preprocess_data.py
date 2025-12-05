#!/usr/bin/env python
"""
Data preprocessing script for HyperGraph-MAE.
Converts molecular SMILES to hypergraph representations.
"""

import argparse
import pandas as pd
import torch
import yaml
from pathlib import Path
import sys
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.hypergraph_construction import smiles_to_hypergraph
from src.data.molecule_features import get_global_feature_statistics
from src.data.molecule_standardizer import MoleculeStandardizer
from src.utils.logging_utils import setup_logger
from src.utils.memory_utils import cleanup_memory
from rdkit import Chem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess molecular data for HyperGraph-MAE")

    # Input/output arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input CSV file with SMILES")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--smiles_column", type=str, default="smiles",
                        help="Name of SMILES column in CSV")
    parser.add_argument("--id_column", type=str, default="id",
                        help="Name of ID column in CSV")

    # Processing arguments
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Configuration file")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for saving")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--max_molecules", type=int, default=None,
                        help="Maximum number of molecules to process")

    # Feature extraction arguments
    parser.add_argument("--compute_stats", action="store_true",
                        help="Compute global feature statistics")
    parser.add_argument("--stats_sample_size", type=int, default=10000,
                        help="Number of molecules to sample for statistics")

    # Standardization arguments
    parser.add_argument("--no_standardize", action="store_true",
                        help="Skip molecule standardization (not recommended)")
    parser.add_argument("--keep_metals", action="store_true",
                        help="Keep metal-containing molecules")
    parser.add_argument("--max_atoms", type=int, default=200,
                        help="Maximum number of atoms per molecule")

    return parser.parse_args()


def load_molecules(input_file: str, smiles_column: str, id_column: str,
                   max_molecules: int = None, standardize: bool = True, 
                   keep_metals: bool = False, max_atoms: int = 200) -> pd.DataFrame:
    """Load and standardize molecules from CSV file."""
    logger = logging.getLogger(__name__)

    # Read CSV
    logger.info(f"Loading molecules from {input_file}")
    df = pd.read_csv(input_file)

    # Check columns
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in CSV")

    # Create ID column if not present
    if id_column not in df.columns:
        logger.info(f"Creating ID column '{id_column}'")
        df[id_column] = [f"mol_{i}" for i in range(len(df))]

    # Filter valid SMILES
    initial_count = len(df)
    df = df[df[smiles_column].notna()]
    logger.info(f"Initial SMILES count: {len(df)}")

    # Standardize molecules if requested
    if standardize:
        logger.info("Starting molecule standardization...")
        # Use CLI parameters for standardization
        standardizer = MoleculeStandardizer(remove_metals=not keep_metals, max_atoms=max_atoms)

        # Get pre-standardization statistics
        pre_stats = standardizer.get_statistics(df[smiles_column].tolist())
        logger.info(f"Pre-standardization stats: {pre_stats}")

        # Standardize SMILES with progress bar
        smiles_list = df[smiles_column].tolist()
        logger.info(f"Standardizing {len(smiles_list)} molecules...")
        
        standardized_smiles = []
        failed_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Standardizing molecules")):
            std_smiles = standardizer.standardize_smiles(smiles)
            if std_smiles is not None:
                standardized_smiles.append(std_smiles)
            else:
                standardized_smiles.append(None)
                failed_indices.append(i)

        # Update dataframe with standardized SMILES
        df[smiles_column] = standardized_smiles

        # Remove failed standardizations
        df = df[df[smiles_column].notna()]

        # Remove duplicates after standardization
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=[smiles_column])
        post_dedup = len(df)

        # Standardization completed message
        logger.info(f"Standardization completed: {len(df)} valid molecules")
        logger.info(f"Removed {len(failed_indices)} failed standardizations")
        logger.info(f"Removed {pre_dedup - post_dedup} duplicates after standardization")
    else:
        # Basic validation without standardization (with progress bar)
        logger.info("Performing basic SMILES validation...")
        valid_indices = []
        for i, smiles in enumerate(tqdm(df[smiles_column], desc="Validating SMILES")):
            if Chem.MolFromSmiles(smiles) is not None:
                valid_indices.append(i)
        
        df = df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Basic validation: {len(df)} valid molecules")

    final_count = len(df)
    logger.info(f"Final molecule count: {final_count} (filtered {initial_count - final_count})")

    # Limit number of molecules if specified
    if max_molecules is not None and final_count > max_molecules:
        df = df.head(max_molecules)
        logger.info(f"Limited to {max_molecules} molecules")

    return df


def process_molecule(row: pd.Series, config: dict, global_stats: dict) -> dict:
    """Process a single molecule."""
    smiles = row['smiles']
    mol_id = row['id']

    try:
        # Convert to hypergraph
        device = torch.device('cpu')  # Process on CPU
        data = smiles_to_hypergraph(smiles, mol_id, config, global_stats, device)

        if data is not None:
            return {'id': mol_id, 'data': data, 'error': None}
        else:
            return {'id': mol_id, 'data': None, 'error': 'Conversion failed'}

    except Exception as e:
        return {'id': mol_id, 'data': None, 'error': str(e)}


def process_molecule_with_timeout(args):
    """Process molecule with proper timeout handling for multiprocessing."""
    row, config, global_stats = args

    def target_function(queue, row, config, global_stats):
        """Target function that runs in separate process."""
        try:
            result = process_molecule(row, config, global_stats)
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', str(e)))

    # Use multiprocessing.Process with timeout instead of signal
    import multiprocessing as mp
    from multiprocessing import Queue
    import time

    queue = Queue()
    process = mp.Process(target=target_function, args=(queue, row, config, global_stats))
    process.start()

    # Wait with timeout (30 seconds)
    process.join(timeout=30)

    if process.is_alive():
        # Process timed out, force kill
        process.terminate()
        process.join(timeout=5)  # Give 5 seconds to clean up
        if process.is_alive():
            process.kill()  # Force kill if still alive

        mol_id = row.get('id', 'unknown')
        return {'id': mol_id, 'data': None, 'error': 'Processing timeout (30s)'}

    # Get result from queue
    try:
        if not queue.empty():
            status, result = queue.get_nowait()
            if status == 'success':
                return result
            else:
                mol_id = row.get('id', 'unknown')
                return {'id': mol_id, 'data': None, 'error': result}
    except:
        pass

    # Process finished but no result (likely crashed)
    mol_id = row.get('id', 'unknown')
    return {'id': mol_id, 'data': None, 'error': 'Process crashed'}


def process_batch_parallel(batch_df: pd.DataFrame, config: dict,
                           global_stats: dict, num_workers: int) -> list:
    """Process a batch of molecules in parallel with proper resource management."""
    # Prepare arguments for each worker
    args_list = [(row, config, global_stats) for _, row in batch_df.iterrows()]

    results = []

    # Process in smaller chunks to avoid resource exhaustion
    chunk_size = min(50, len(args_list))  # Process max 50 molecules at once

    for i in range(0, len(args_list), chunk_size):
        chunk = args_list[i:i + chunk_size]

        # Use ThreadPoolExecutor instead of Process Pool for better resource control
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        with ThreadPoolExecutor(max_workers=min(num_workers, len(chunk))) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(process_molecule_with_timeout, args): args
                for args in chunk
            }

            # Collect results with timeout per task
            for future in as_completed(future_to_args, timeout=60):  # 60s total timeout per chunk
                try:
                    result = future.result(timeout=35)  # 35s per individual task
                    results.append(result)
                except Exception as e:
                    # Handle failed tasks
                    args = future_to_args[future]
                    mol_id = args[0].get('id', 'unknown')
                    results.append({
                        'id': mol_id,
                        'data': None,
                        'error': f'Task failed: {str(e)}'
                    })

        # Force garbage collection between chunks
        import gc
        gc.collect()

    return results


def compute_statistics(df: pd.DataFrame, smiles_column: str,
                       sample_size: int) -> dict:
    """Compute global feature statistics."""
    logger = logging.getLogger(__name__)
    logger.info("Computing global feature statistics")

    # Sample molecules
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    # Convert to RDKit molecules with progress bar
    molecules = []
    logger.info(f"Converting {len(sample_df)} molecules for statistics computation...")
    for smiles in tqdm(sample_df[smiles_column], desc="Converting molecules"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)

    # Compute statistics
    stats = get_global_feature_statistics(molecules)
    logger.info(f"Computed statistics from {len(molecules)} molecules")

    return stats


def save_batch(batch_data: list, output_dir: Path, batch_idx: int):
    """Save a batch of processed data (ID is already embedded in each Data object)."""
    # Separate successful and failed data
    successful_data = []
    failed_ids = []

    for item in batch_data:
        if item['data'] is not None:
            successful_data.append(item['data'])  # Data object contains id and smiles
        else:
            failed_ids.append(item['id'])

    # Save successful data (each Data object contains id and smiles)
    if successful_data:
        output_file = output_dir / f"batch_{batch_idx:04d}.pt"
        torch.save(successful_data, output_file)

    # Save failed IDs for reference
    if failed_ids:
        failed_file = output_dir / f"batch_{batch_idx:04d}_failed_ids.txt"
        with open(failed_file, 'w') as f:
            for failed_id in failed_ids:
                f.write(f"{failed_id}\n")

    # Log errors with details
    errors = [item for item in batch_data if item['error'] is not None]
    if errors:
        error_file = output_dir / f"errors_{batch_idx:04d}.txt"
        with open(error_file, 'w') as f:
            for item in errors:
                f.write(f"{item['id']}: {item['error']}\n")

    return len(successful_data), len(failed_ids)


def main():
    """Main preprocessing function."""
    args = parse_args()

    # Setup logging
    logger = setup_logger("preprocess", level="INFO")
    logger.info("Starting molecular data preprocessing")

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load molecules with standardization
    standardize_molecules = not args.no_standardize
    df = load_molecules(
        args.input_file,
        args.smiles_column,
        args.id_column,
        args.max_molecules,
        standardize=standardize_molecules,
        keep_metals=args.keep_metals,  # Pass CLI parameter
        max_atoms=args.max_atoms       # Pass CLI parameter  
    )

    # Rename columns for processing
    df = df.rename(columns={
        args.smiles_column: 'smiles',
        args.id_column: 'id'
    })

    # Compute or load global statistics
    stats_file = output_dir / "global_stats.pkl"

    if args.compute_stats or not stats_file.exists():
        global_stats = compute_statistics(
            df, 'smiles', args.stats_sample_size
        )

        # Save statistics
        with open(stats_file, 'wb') as f:
            pickle.dump(global_stats, f)
        logger.info(f"Saved global statistics to {stats_file}")
    else:
        # Load existing statistics
        with open(stats_file, 'rb') as f:
            global_stats = pickle.load(f)
        logger.info(f"Loaded global statistics from {stats_file}")

    # Determine number of workers
    if args.num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = args.num_workers
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Process molecules in batches with overall progress tracking
    total_processed = 0
    total_errors = 0
    batch_idx = 0
    
    logger.info(f"Processing {len(df)} molecules in batches of {args.batch_size}")

    with tqdm(total=len(df), desc="Overall progress", unit="molecules") as pbar:
        for start_idx in range(0, len(df), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # Process batch
            if num_workers > 1:
                results = process_batch_parallel(
                    batch_df, config, global_stats, num_workers
                )
            else:
                # Sequential processing with batch-specific progress bar  
                results = []
                batch_desc = f"Batch {batch_idx+1}/{(len(df) + args.batch_size - 1) // args.batch_size}"
                for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=batch_desc, leave=False):
                    result = process_molecule(row, config, global_stats)
                    results.append(result)

            # Save batch
            processed, errors = save_batch(results, output_dir, batch_idx)
            total_processed += processed
            total_errors += errors
            batch_idx += 1

            # Update overall progress
            pbar.update(len(batch_df))
            pbar.set_postfix({
                'processed': total_processed, 
                'errors': total_errors,
                'success_rate': f"{total_processed/(total_processed+total_errors)*100:.1f}%" if (total_processed + total_errors) > 0 else "0%"
            })

            # Clean up memory
            cleanup_memory()

    # Collect failed IDs for summary (with progress bar)
    all_failed_ids = []
    logger.info("Collecting failed molecule IDs...")
    for i in tqdm(range(batch_idx), desc="Collecting failed IDs"):
        failed_file = output_dir / f"batch_{i:04d}_failed_ids.txt"
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                batch_failed = [line.strip() for line in f if line.strip()]
                all_failed_ids.extend(batch_failed)

    # Save metadata
    metadata = {
        'total_molecules': len(df),
        'processed_molecules': total_processed,
        'failed_molecules': total_errors,
        'num_batches': batch_idx,
        'failed_ids': all_failed_ids,
        'success_rate': total_processed / len(df) if len(df) > 0 else 0,
        'config': config,
        'global_stats': global_stats,
        'note': 'Each Data object contains original ID and SMILES'
    }

    metadata_file = output_dir / "metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    # Summary
    logger.info(f"Preprocessing completed")
    logger.info(f"Total molecules: {len(df)}")
    logger.info(f"Successfully processed: {total_processed}")
    logger.info(f"Failed: {total_errors}")
    logger.info(f"Output saved to: {output_dir}")

    # Create summary file
    summary_file = output_dir / "preprocessing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Preprocessing Summary\n")
        f.write(f"====================\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Total molecules: {len(df)}\n")
        f.write(f"Successfully processed: {total_processed}\n")
        f.write(f"Failed: {total_errors}\n")
        f.write(f"Success rate: {total_processed / len(df) * 100:.2f}%\n")
        f.write(f"Number of batches: {batch_idx}\n")
        f.write(f"Output directory: {output_dir}\n")


if __name__ == "__main__":
    main()
