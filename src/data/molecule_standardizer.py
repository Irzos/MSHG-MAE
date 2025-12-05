#!/usr/bin/env python
"""
Enhanced molecular preprocessing with standardization.
This module handles molecule cleaning, salt removal, and standardization.
"""

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit import rdBase
import logging
from typing import Optional, Tuple

# Handle different RDKit versions for standardization
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles as standardize_smiles
    RDKIT_STANDARDIZE_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths for different RDKit versions
        from rdkit.Chem import rdMolStandardize
        from rdkit.Chem.rdMolStandardize import StandardizeSmiles as standardize_smiles
        RDKIT_STANDARDIZE_AVAILABLE = True
    except ImportError:
        try:
            # Fallback: try direct import from Chem
            from rdkit.Chem.rdMolStandardize import rdMolStandardize
            # Try to import or create fallback standardize_smiles
            try:
                from rdkit.Chem.rdMolStandardize import StandardizeSmiles as standardize_smiles
            except ImportError:
                def standardize_smiles(smiles):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            return None
                        return Chem.MolToSmiles(mol, canonical=True)
                    except:
                        return None
            RDKIT_STANDARDIZE_AVAILABLE = True
        except ImportError:
            # Final fallback: disable standardization features
            rdMolStandardize = None
            RDKIT_STANDARDIZE_AVAILABLE = False
            
            def standardize_smiles(smiles):
                """Fallback standardize function when MolStandardize is not available"""
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return None
                    return Chem.MolToSmiles(mol, canonical=True)
                except:
                    return None

logger = logging.getLogger(__name__)


class MoleculeStandardizer:
    """
    Comprehensive molecule standardization pipeline.
    
    Features:
    - Salt removal
    - Charge neutralization  
    - Tautomer normalization
    - Stereochemistry handling
    - Metal filtering (configurable)
    """
    
    def __init__(self, remove_metals: bool = True, max_atoms: int = 200, 
                 metal_atomic_numbers: Optional[list] = None):
        """
        Initialize standardizer.
        
        Args:
            remove_metals: Whether to filter out metal-containing molecules
            max_atoms: Maximum number of atoms allowed
            metal_atomic_numbers: List of atomic numbers to filter as metals
        """
        self.remove_metals = remove_metals
        self.max_atoms = max_atoms
        
        # Default to transition metals only (more conservative than original list)
        if metal_atomic_numbers is None:
            self.metal_atomic_numbers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # 3d transition metals
                                       39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # 4d transition metals  
                                       71, 72, 73, 74, 75, 76, 77, 78, 79, 80]  # 5d transition metals
        else:
            self.metal_atomic_numbers = metal_atomic_numbers
        
        # Initialize RDKit standardization tools
        try:
            self.salt_remover = SaltRemover.SaltRemover()
            if RDKIT_STANDARDIZE_AVAILABLE and rdMolStandardize is not None:
                self.normalizer = rdMolStandardize.Normalizer()
                self.uncharger = rdMolStandardize.Uncharger()  
                self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
                logger.info("Molecular standardizer initialized with full standardization support")
            else:
                self.normalizer = None
                self.uncharger = None
                self.tautomer_enumerator = None
                logger.warning("MolStandardize not available - using basic standardization only")
        except Exception as e:
            logger.error(f"Failed to initialize standardizer: {e}")
            raise
    
    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Comprehensive SMILES standardization pipeline.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Standardized SMILES string or None if failed
        """
        # Suppress RDKit warnings during standardization
        rdBase.DisableLog('rdApp.warning')
        rdBase.DisableLog('rdApp.error')
        
        try:
            # Step 1: Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Step 2: Basic validation
            if not self._validate_molecule(mol):
                return None
            
            # Step 3: Remove salts and counterions
            mol = self.salt_remover.StripMol(mol)
            if mol is None:
                return None
            
            # Step 4: Normalize functional groups (if available)
            if self.normalizer is not None:
                mol = self.normalizer.normalize(mol)
                if mol is None:
                    return None
            
            # Step 5: Remove charges where possible (if available)
            if self.uncharger is not None:
                mol = self.uncharger.uncharge(mol)
                if mol is None:
                    return None
            
            # Step 6: Canonical tautomer (if available)
            if self.tautomer_enumerator is not None:
                mol = self.tautomer_enumerator.Canonicalize(mol)
                if mol is None:
                    return None
            
            # Step 7: Generate canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            
            return canonical_smiles
            
        except Exception as e:
            logger.debug(f"Standardization failed for {smiles}: {e}")
            return None
        finally:
            # Re-enable RDKit logging
            rdBase.EnableLog('rdApp.warning')
            rdBase.EnableLog('rdApp.error')
    
    def _validate_molecule(self, mol: Chem.Mol) -> bool:
        """
        Validate molecule before processing.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            True if molecule is valid for processing
        """
        try:
            # Check if molecule exists
            if mol is None:
                return False
            
            # Check molecule size
            if mol.GetNumAtoms() == 0 or mol.GetNumAtoms() > self.max_atoms:
                return False
            
            # Filter metals if requested (now configurable)
            if self.remove_metals:
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() in self.metal_atomic_numbers:
                        return False
            
            # Check for valid chemical structure
            try:
                Chem.SanitizeMol(mol)
            except:
                return False
            
            return True
            
        except Exception:
            return False
    
    def batch_standardize(self, smiles_list: list) -> Tuple[list, list]:
        """
        Batch standardization of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (standardized_smiles, failed_indices)
        """
        standardized = []
        failed_indices = []
        
        for i, smiles in enumerate(smiles_list):
            std_smiles = self.standardize_smiles(smiles)
            if std_smiles is not None:
                standardized.append(std_smiles)
            else:
                standardized.append(None)
                failed_indices.append(i)
        
        logger.info(f"Standardization: {len(smiles_list) - len(failed_indices)}/{len(smiles_list)} succeeded")
        
        return standardized, failed_indices
    
    def get_statistics(self, smiles_list: list) -> dict:
        """
        Get standardization statistics.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total': len(smiles_list),
            'valid': 0,
            'invalid': 0,
            'metal_containing': 0,
            'too_large': 0,
            'parsing_failed': 0
        }
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    stats['parsing_failed'] += 1
                    stats['invalid'] += 1
                    continue
                
                # Check size
                if mol.GetNumAtoms() > self.max_atoms:
                    stats['too_large'] += 1
                    stats['invalid'] += 1
                    continue
                
                # Check metals
                has_metal = any(atom.GetAtomicNum() > 56 for atom in mol.GetAtoms())
                if has_metal:
                    stats['metal_containing'] += 1
                    if self.remove_metals:
                        stats['invalid'] += 1
                        continue
                
                stats['valid'] += 1
                
            except Exception:
                stats['parsing_failed'] += 1
                stats['invalid'] += 1
        
        stats['success_rate'] = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
        
        return stats


def standardize_smiles_simple(smiles: str) -> Optional[str]:
    """
    Simple SMILES standardization function for quick use.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized SMILES or None if failed
    """
    standardizer = MoleculeStandardizer()
    return standardizer.standardize_smiles(smiles)


if __name__ == "__main__":
    # Test the standardizer
    test_smiles = [
        "CC(=O)O",  # Acetic acid
        "CC(=O)[O-].[Na+]",  # Sodium acetate (with salt)
        "c1ccccc1",  # Benzene
        "CC(=O)O.CC(=O)O",  # Acetic acid dimer
        "invalid_smiles",  # Invalid
    ]
    
    standardizer = MoleculeStandardizer()
    
    print("Testing molecule standardization:")
    for smiles in test_smiles:
        result = standardizer.standardize_smiles(smiles)
        print(f"{smiles:20s} -> {result}")
    
    # Get statistics
    stats = standardizer.get_statistics(test_smiles)
    print(f"\nStatistics: {stats}")