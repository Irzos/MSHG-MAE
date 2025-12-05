"""
Functional group identification and feature extraction module.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FunctionalGroupIdentifier:
    """Identify and extract features from functional groups in molecules."""

    def __init__(self, config: Dict):
        """
        Initialize functional group identifier.

        Args:
            config: Configuration dictionary containing SMARTS patterns
        """
        self.config = config
        self.smarts_patterns = self._load_smarts_patterns(config)
        self.fg_types = list(self.smarts_patterns.keys())

    def _load_smarts_patterns(self, config: Dict) -> Dict[str, str]:
        """Load SMARTS patterns from configuration."""
        default_patterns = {
            'carboxyl': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'nitro': '[NX3](=O)=O',
            'cyano': '[CX2]#N',
            'sulfonyl': '[SX4](=[OX1])(=[OX1])([OX2H,OX1H0])',
            'phosphate': '[PX4](=[OX1])([OX2H0])([OX2H0])([OX2H0])',
            'amide': '[NX3][CX3](=[OX1])',
            'ether': '[OD2]([#6])[#6]',
            'ester': '[#6][CX3](=O)[OX2][#6]',
            'thiol': '[SX2H]',
            'ketone': '[#6][CX3](=O)[#6]',
            'aldehyde': '[CX3H1](=O)[#6]',
            'halogen': '[F,Cl,Br,I]',
            # Specific aromatic systems instead of general [a] pattern
            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
            'imidazole': 'c1c[nH]cn1',
            'pyrimidine': 'c1cncnc1',
            'purine': 'c1nc2[nH]cnc2n1'
            # NOTE: Removed generic 'aromatic': '[a]' pattern to avoid over-matching
            # Specific aromatic patterns above provide better precision
        }

        # Override with config if provided
        if 'functional_groups' in config:
            default_patterns.update(config['functional_groups'])
            
        # Allow disabling problematic patterns via config
        disabled_patterns = config.get('disabled_fg_patterns', [])
        for pattern in disabled_patterns:
            if pattern in default_patterns:
                del default_patterns[pattern]
                logger.info(f"Disabled functional group pattern: {pattern}")

        return default_patterns

    def identify(self, mol: Chem.Mol) -> List[Tuple[List[int], str]]:
        """
        Identify functional groups in a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            List of tuples containing (atom_indices, functional_group_type)
        """
        functional_groups = []

        for fg_type, smarts in self.smarts_patterns.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    logger.warning(f"Invalid SMARTS pattern for {fg_type}: {smarts}")
                    continue

                matches = mol.GetSubstructMatches(patt)

                for match in matches:
                    # Expand match to include neighboring atoms for context
                    expanded_match = self._expand_match(mol, match)
                    functional_groups.append((expanded_match, fg_type))

            except Exception as e:
                logger.error(f"Error identifying {fg_type}: {str(e)}")

        return functional_groups

    def _expand_match(self, mol: Chem.Mol, match: Tuple[int]) -> List[int]:
        """
        Expand a functional group match to include neighboring atoms.

        Args:
            mol: RDKit molecule object
            match: Tuple of atom indices in the match

        Returns:
            List of expanded atom indices
        """
        expanded = set(match)

        # Add neighboring atoms
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                expanded.add(neighbor.GetIdx())

        return list(expanded)

    def extract_features(self, atoms: List[int], mol: Chem.Mol, fg_type: str) -> torch.Tensor:
        """
        Extract features from a functional group (no dimension limit).

        Args:
            atoms: List of atom indices in the functional group
            mol: RDKit molecule object
            fg_type: Type of functional group

        Returns:
            Full feature tensor for the functional group (no truncation)
        """
        num_atoms = len(atoms)
        atoms_obj = [mol.GetAtomWithIdx(idx) for idx in atoms]

        # Extract continuous features
        cont_features = [
            float(num_atoms),  # Number of atoms
            1.0 if any(atom.GetIsAromatic() for atom in atoms_obj) else 0.0,  # Contains aromatic
            sum(atom.GetAtomicNum() for atom in atoms_obj) / num_atoms,  # Average atomic number
            sum(atom.GetDegree() for atom in atoms_obj) / num_atoms,  # Average degree
            sum(atom.GetFormalCharge() for atom in atoms_obj) / num_atoms,  # Average formal charge
            sum(atom.GetTotalNumHs() for atom in atoms_obj) / num_atoms  # Average H count
        ]

        cont_features = torch.tensor(cont_features, dtype=torch.float32)
        cont_features = F.normalize(cont_features, p=2, dim=0)

        # Create one-hot encoding for functional group type
        fg_type_onehot = torch.zeros(len(self.fg_types))
        if fg_type in self.fg_types:
            idx = self.fg_types.index(fg_type)
            fg_type_onehot[idx] = 1.0

        # Combine features
        features = torch.cat([cont_features, fg_type_onehot])

        return features  # Return full feature vector without padding/truncation


class ConjugatedSystemIdentifier:
    """Identify conjugated systems in molecules."""

    def identify(self, mol: Chem.Mol) -> List[List[int]]:
        """
        Identify conjugated systems in a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            List of atom indices for each conjugated system
        """
        conjugated_systems = []
        visited_bonds = set()

        for bond in mol.GetBonds():
            if bond.GetIdx() in visited_bonds or not bond.GetIsConjugated():
                continue

            # Start a new conjugated system
            system = set()
            stack = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            visited_bonds.add(bond.GetIdx())

            # Traverse the conjugated system
            while stack:
                atom_idx = stack.pop()
                atom = mol.GetAtomWithIdx(atom_idx)

                for neighbor_bond in atom.GetBonds():
                    if (neighbor_bond.GetIsConjugated() and
                            neighbor_bond.GetIdx() not in visited_bonds):
                        visited_bonds.add(neighbor_bond.GetIdx())
                        system.update([neighbor_bond.GetBeginAtomIdx(),
                                       neighbor_bond.GetEndAtomIdx()])
                        stack.extend([neighbor_bond.GetBeginAtomIdx(),
                                      neighbor_bond.GetEndAtomIdx()])

            if len(system) >= 3:  # Only include systems with 3+ atoms
                conjugated_systems.append(list(system))

        return conjugated_systems

    def extract_features(self, system: List[int], mol: Chem.Mol) -> torch.Tensor:
        """
        Extract features from a conjugated system (no dimension limit).

        Args:
            system: List of atom indices in the conjugated system
            mol: RDKit molecule object

        Returns:
            Full feature tensor for the conjugated system
        """
        # Comprehensive conjugated system features
        features = torch.tensor([
            1.0,  # Conjugated system indicator
            len(system) / 50.0,  # Normalized system size
            sum(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in system) / len(system),  # Aromaticity ratio
            sum(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in system) / len(system) / 20.0,  # Avg atomic number
            sum(mol.GetAtomWithIdx(idx).GetDegree() for idx in system) / len(system) / 6.0,  # Avg degree
            sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in system) / len(system),  # Avg formal charge
        ], dtype=torch.float32)

        return features


def identify_all_functional_groups(mol: Chem.Mol, config: Dict) -> List[Tuple[List[int], str, torch.Tensor]]:
    """
    Identify all functional groups and extract their features with deduplication.

    Args:
        mol: RDKit molecule object
        config: Configuration dictionary

    Returns:
        List of tuples containing (atom_indices, fg_type, features) with duplicates removed
    """
    results = []
    seen_atom_sets = set()  # Track unique atom sets for deduplication

    # Identify standard functional groups
    fg_identifier = FunctionalGroupIdentifier(config)
    functional_groups = fg_identifier.identify(mol)

    for atoms, fg_type in functional_groups:
        atom_set = frozenset(atoms)  # Use frozenset for hashable atom set
        
        # Skip if this exact atom set has been seen before
        if atom_set in seen_atom_sets:
            continue
            
        seen_atom_sets.add(atom_set)
        features = fg_identifier.extract_features(atoms, mol, fg_type)
        results.append((atoms, fg_type, features))

    # Identify conjugated systems
    conj_identifier = ConjugatedSystemIdentifier()
    conjugated_systems = conj_identifier.identify(mol)

    for system in conjugated_systems:
        atom_set = frozenset(system)
        
        # Skip if this exact atom set has been seen before
        if atom_set in seen_atom_sets:
            continue
            
        seen_atom_sets.add(atom_set)
        features = conj_identifier.extract_features(system, mol)
        results.append((system, 'conjugated_system', features))

    return results


def deduplicate_hyperedges(hyperedges: List[Tuple[List[int], str, torch.Tensor]]) -> List[Tuple[List[int], str, torch.Tensor]]:
    """
    Remove duplicate hyperedges based on atom sets.
    
    Args:
        hyperedges: List of hyperedges with (atoms, type, features)
        
    Returns:
        Deduplicated list of hyperedges
    """
    seen_sets = set()
    deduplicated = []
    
    for atoms, edge_type, features in hyperedges:
        atom_set = frozenset(atoms)
        
        if atom_set not in seen_sets:
            seen_sets.add(atom_set)
            deduplicated.append((atoms, edge_type, features))
    
    return deduplicated
