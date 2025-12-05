#!/usr/bin/env python3
"""
Offline semantic-block preprocessing.

Precompute semantic blocks from SMILES to avoid online RDKit calls during training.

Usage:
    python scripts/preprocess_semantic_blocks.py --input dataset.csv --output blocks.pkl
    python scripts/preprocess_semantic_blocks.py --smiles-list smiles.txt --output blocks.pkl
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

import pandas as pd
from tqdm import tqdm
from rdkit import Chem

# Add project path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.molecular_semantics import MolecularSemanticAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticBlockPreprocessor:
    """Offline semantic-block preprocessor."""
    
    def __init__(self, cache_dir: Path = None):
        """Initialize the preprocessor.

        Args:
            cache_dir: Cache directory path
        """
        self.analyzer = MolecularSemanticAnalyzer()
        self.cache_dir = cache_dir or Path("cache/semantic_blocks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'processing_time': 0.0,
            'avg_time_per_molecule': 0.0
        }
    
    def preprocess_smiles_list(self, smiles_list: List[str], 
                             output_path: Path,
                             batch_size: int = 1000) -> Dict[str, Any]:
        """Preprocess a list of SMILES.

        Args:
            smiles_list: List of SMILES strings
            output_path: Output file path
            batch_size: Batch size

        Returns:
            Stats dictionary
        """
        logger.info(f"Starting preprocessing for {len(smiles_list)} molecules")
        start_time = time.time()
        
        semantic_blocks_cache = {}
        failed_smiles = []
        
        # Batch processing
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Preprocessing semantic blocks"):
            batch = smiles_list[i:i + batch_size]
            
            for smiles in batch:
                self.stats['total_processed'] += 1
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        logger.warning(f"Invalid SMILES: {smiles}")
                        failed_smiles.append(smiles)
                        self.stats['failed'] += 1
                        continue
                    
                    # Analyze molecule semantics
                    mol_info = self.analyzer.analyze_molecule(mol, smiles)
                    
                    # Extract key fields for training
                    semantic_data = {
                        'smiles': smiles,
                        'semantic_blocks': mol_info.get('semantic_blocks', {}),
                        'functional_groups': mol_info.get('functional_groups', []),
                        'ring_systems': mol_info.get('ring_systems', []),
                        'atom_annotations': mol_info.get('atom_annotations', {}),
                        'num_atoms': mol.GetNumAtoms(),
                        'num_bonds': mol.GetNumBonds()
                    }
                    
                    semantic_blocks_cache[smiles] = semantic_data
                    self.stats['successful'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process molecule {smiles}: {e}")
                    failed_smiles.append(smiles)
                    self.stats['failed'] += 1
        
        self.stats['processing_time'] = time.time() - start_time
        self.stats['avg_time_per_molecule'] = self.stats['processing_time'] / len(smiles_list)
        
        # Save results
        self._save_results(semantic_blocks_cache, output_path, failed_smiles)
        
        logger.info("Preprocessing completed")
        logger.info(f"Success: {self.stats['successful']}, Failed: {self.stats['failed']}")
        logger.info(f"Total time: {self.stats['processing_time']:.2f} s")
        logger.info(f"Average per molecule: {self.stats['avg_time_per_molecule']*1000:.2f} ms")
        
        return self.stats
    
    def _save_results(self, semantic_blocks_cache: Dict[str, Any], 
                     output_path: Path, failed_smiles: List[str]):
        """Save preprocessing results."""
        
        # Main cache file
        with open(output_path, 'wb') as f:
            pickle.dump(semantic_blocks_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Semantic block cache saved to: {output_path}")
        logger.info(f"Cache size: {len(semantic_blocks_cache)} molecules")
        
        # Failed list
        if failed_smiles:
            failed_path = output_path.with_suffix('.failed.txt')
            with open(failed_path, 'w') as f:
                for smiles in failed_smiles:
                    f.write(f"{smiles}\n")
            logger.info(f"Failed SMILES list saved to: {failed_path}")
        
        # Stats report
        stats_path = output_path.with_suffix('.stats.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics report saved to: {stats_path}")
    
    def load_smiles_from_csv(self, csv_path: Path, smiles_column: str = 'smiles') -> List[str]:
        """Load SMILES from a CSV file."""
        df = pd.read_csv(csv_path)
        if smiles_column not in df.columns:
            raise ValueError(f"Column not found in CSV: {smiles_column}")
        
        smiles_list = df[smiles_column].dropna().unique().tolist()
        logger.info(f"Loaded {len(smiles_list)} unique SMILES from {csv_path}")
        return smiles_list
    
    def load_smiles_from_txt(self, txt_path: Path) -> List[str]:
        """Load SMILES from a text file."""
        with open(txt_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        # Deduplicate
        smiles_list = list(set(smiles_list))
        logger.info(f"Loaded {len(smiles_list)} unique SMILES from {txt_path}")
        return smiles_list


def main():
    parser = argparse.ArgumentParser(description="Offline semantic-block preprocessing")
    parser.add_argument('--input', '-i', type=Path, required=True,
                       help='Input file path (CSV or TXT)')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output pickle file path')
    parser.add_argument('--smiles-column', default='smiles',
                       help='SMILES column name in CSV (default: smiles)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size (default: 1000)')
    parser.add_argument('--cache-dir', type=Path, default=None,
                       help='Cache directory (default: cache/semantic_blocks)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = SemanticBlockPreprocessor(cache_dir=args.cache_dir)
    
    # Load SMILES
    if args.input.suffix.lower() == '.csv':
        smiles_list = preprocessor.load_smiles_from_csv(args.input, args.smiles_column)
    elif args.input.suffix.lower() in ['.txt', '.smi']:
        smiles_list = preprocessor.load_smiles_from_txt(args.input)
    else:
        raise ValueError(f"Unsupported file extension: {args.input.suffix}")
    
    # Run preprocessing
    stats = preprocessor.preprocess_smiles_list(
        smiles_list, args.output, args.batch_size
    )
    
    print("\nPreprocessing completed!")
    print(f"Output file: {args.output}")
    print(f"Success rate: {stats['successful']/(stats['total_processed'] or 1):.1%}")


if __name__ == '__main__':
    main()
