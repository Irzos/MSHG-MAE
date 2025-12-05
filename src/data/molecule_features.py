"""
Enhanced molecular feature extraction module.
Handles extraction of atom and bond features from RDKit molecules with full feature vectors.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AtomFeatureExtractor:
    """Extract features from atoms in a molecule."""

    def __init__(self, config: Dict):
        """
        Initialize atom feature extractor.

        Args:
            config: Configuration dictionary containing feature settings
        """
        self.config = config
        self.continuous_features = config.get('continuous_features',
                                              ['atomic_num', 'formal_charge', 'total_num_hs'])
        self.categorical_features = config.get('categorical_features',
                                               ['degree', 'is_aromatic', 'hybridization',
                                                'chiral_tag', 'is_in_ring'])

        # Define categorical mappings using RDKit enums
        self.degree_cat = list(range(7))  # Degrees 0-6
        self.hybrid_cat = [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        self.chiral_cat = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]

    def extract(self, atom: Chem.Atom, global_stats: Dict) -> torch.Tensor:
        """
        Extract features from an atom.

        Args:
            atom: RDKit atom object
            global_stats: Dictionary containing global statistics for normalization

        Returns:
            Feature tensor for the atom
        """
        features = []

        # Extract continuous features
        cont_features = []
        if 'atomic_num' in self.continuous_features:
            cont_features.append(atom.GetAtomicNum())
        if 'formal_charge' in self.continuous_features:
            cont_features.append(atom.GetFormalCharge())
        if 'total_num_hs' in self.continuous_features:
            cont_features.append(atom.GetTotalNumHs())

        # Normalize continuous features using fixed key names
        if cont_features:
            cont_tensor = torch.tensor(cont_features, dtype=torch.float32)
            cont_means = torch.tensor(global_stats['cont_mean'])
            cont_stds = torch.tensor(global_stats['cont_std'])
            cont_tensor = (cont_tensor - cont_means) / cont_stds
            features.append(cont_tensor)

        # Extract categorical features
        if 'degree' in self.categorical_features:
            degree = atom.GetDegree()
            degree_onehot = torch.zeros(len(self.degree_cat))
            if degree in self.degree_cat:
                degree_onehot[self.degree_cat.index(degree)] = 1.0
            features.append(degree_onehot)

        if 'is_aromatic' in self.categorical_features:
            is_aromatic = atom.GetIsAromatic()
            aromatic_onehot = torch.tensor([1.0, 0.0] if is_aromatic else [0.0, 1.0])
            features.append(aromatic_onehot)

        if 'hybridization' in self.categorical_features:
            hybrid = atom.GetHybridization()
            hybrid_onehot = torch.zeros(len(self.hybrid_cat))
            # Use direct enum comparison instead of string comparison
            for i, h in enumerate(self.hybrid_cat):
                if hybrid == h:
                    hybrid_onehot[i] = 1.0
                    break
            features.append(hybrid_onehot)

        if 'chiral_tag' in self.categorical_features:
            chiral = atom.GetChiralTag()
            chiral_onehot = torch.zeros(len(self.chiral_cat))
            # Use direct enum comparison instead of string comparison
            for i, c in enumerate(self.chiral_cat):
                if chiral == c:
                    chiral_onehot[i] = 1.0
                    break
            features.append(chiral_onehot)

        if 'is_in_ring' in self.categorical_features:
            is_in_ring = atom.IsInRing()
            ring_onehot = torch.tensor([1.0, 0.0] if is_in_ring else [0.0, 1.0])
            features.append(ring_onehot)

        return torch.cat(features)


class BondFeatureExtractor:
    """Extract features from bonds in a molecule."""

    def __init__(self, config: Dict):
        """
        Initialize bond feature extractor.

        Args:
            config: Configuration dictionary containing feature settings
        """
        self.config = config
        self.continuous_features = config.get('continuous_features', ['avg_hybridization'])
        self.categorical_features = config.get('categorical_features',
                                               ['is_conjugated', 'is_aromatic', 'stereo',
                                                'is_in_ring', 'is_rotatable', 'bond_type'])

        # Define categorical mappings using RDKit enums
        self.bond_type_cat = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE, 
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        self.stereo_cat = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY
        ]

    def extract(self, bond: Chem.Bond, mol: Chem.Mol, global_stats: Dict) -> torch.Tensor:
        """
        Extract features from a bond (no dimension truncation).

        Args:
            bond: RDKit bond object
            mol: RDKit molecule object
            global_stats: Dictionary containing global statistics

        Returns:
            Full feature tensor for the bond (no truncation)
        """
        features = []

        # Extract continuous features
        cont_features = []
        if 'avg_hybridization' in self.continuous_features:
            hybrid_begin = bond.GetBeginAtom().GetHybridization()
            hybrid_end = bond.GetEndAtom().GetHybridization()
            avg_hybrid = (int(hybrid_begin) + int(hybrid_end)) / 2.0
            cont_features.append(avg_hybrid)

        # Normalize continuous features
        if cont_features:
            cont_tensor = torch.tensor(cont_features, dtype=torch.float32)
            cont_means = torch.tensor(global_stats['bond_cont_means'])
            cont_stds = torch.tensor(global_stats['bond_cont_stds'])
            cont_tensor = (cont_tensor - cont_means) / cont_stds
            features.append(cont_tensor)

        # Extract categorical features
        if 'is_conjugated' in self.categorical_features:
            conjugated = bond.GetIsConjugated()
            conjugated_onehot = torch.tensor([1.0, 0.0] if conjugated else [0.0, 1.0])
            features.append(conjugated_onehot)

        if 'is_aromatic' in self.categorical_features:
            aromatic = bond.GetIsAromatic()
            aromatic_onehot = torch.tensor([1.0, 0.0] if aromatic else [0.0, 1.0])
            features.append(aromatic_onehot)

        if 'stereo' in self.categorical_features:
            stereo = bond.GetStereo()
            stereo_onehot = torch.zeros(len(self.stereo_cat))
            # Use direct enum comparison instead of string comparison
            for i, s in enumerate(self.stereo_cat):
                if stereo == s:
                    stereo_onehot[i] = 1.0
                    break
            features.append(stereo_onehot)

        if 'is_in_ring' in self.categorical_features:
            in_ring = bond.IsInRing()
            in_ring_onehot = torch.tensor([1.0, 0.0] if in_ring else [0.0, 1.0])
            features.append(in_ring_onehot)

        if 'is_rotatable' in self.categorical_features:
            rotatable = self._is_rotatable(bond)
            rotatable_onehot = torch.tensor([1.0, 0.0] if rotatable else [0.0, 1.0])
            features.append(rotatable_onehot)

        if 'bond_type' in self.categorical_features:
            bond_type = bond.GetBondType()  # Use enum directly
            bond_type_onehot = torch.zeros(len(self.bond_type_cat))
            # Use direct enum comparison instead of string comparison
            for i, bt in enumerate(self.bond_type_cat):
                if bond_type == bt:
                    bond_type_onehot[i] = 1.0
                    break
            features.append(bond_type_onehot)

        # Return full concatenated features without truncation
        result = torch.cat(features)
        return result

    def _is_rotatable(self, bond: Chem.Bond) -> bool:
        """Check if a bond is rotatable using RDKit enum."""
        return (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and
                not bond.IsInRing() and
                bond.GetBeginAtom().GetDegree() > 1 and
                bond.GetEndAtom().GetDegree() > 1)


class RingFeatureExtractor:
    """Extract features from ring structures."""

    def extract(self, ring: List[int], mol: Chem.Mol) -> torch.Tensor:
        """
        Extract features from a ring structure (no dimension limit).

        Args:
            ring: List of atom indices in the ring
            mol: RDKit molecule object

        Returns:
            Full feature tensor for the ring (no truncation)
        """
        ring_size = len(ring)
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]

        features = [
            float(ring_size),  # Ring size
            1.0 if all(atom.GetIsAromatic() for atom in atoms) else 0.0,  # Aromaticity
            1.0 if any(atom.GetAtomicNum() != 6 for atom in atoms) else 0.0,  # Heterocycle
            sum(atom.GetAtomicNum() for atom in atoms) / ring_size,  # Average atomic number
            sum(atom.GetDegree() for atom in atoms) / ring_size,  # Average degree
            sum(atom.GetFormalCharge() for atom in atoms) / ring_size,  # Average formal charge
            sum(atom.GetTotalNumHs() for atom in atoms) / ring_size  # Average H count
        ]

        features = torch.tensor(features, dtype=torch.float32)
        features = F.normalize(features, p=2, dim=0)

        return features  # Return full features without padding/truncation


def get_global_feature_statistics(molecules: List[Chem.Mol]) -> Dict:
    """
    Calculate global statistics for continuous atom features only (single source).
    Avoid "pre-normalized values being re-normalized" contamination.

    Args:
        molecules: List of RDKit molecule objects

    Returns:
        Dictionary containing means and standard deviations for continuous features only
    """
    # Only collect raw continuous atom features (no pre-processed features)
    atom_features = {'atomic_num': [], 'formal_charge': [], 'total_num_hs': []}
    bond_features = {'avg_hybrid': []}

    for mol in molecules:
        if mol is None:
            continue

        # Collect raw atom features (before any normalization)
        for atom in mol.GetAtoms():
            atom_features['atomic_num'].append(float(atom.GetAtomicNum()))
            atom_features['formal_charge'].append(float(atom.GetFormalCharge()))
            atom_features['total_num_hs'].append(float(atom.GetTotalNumHs()))

        # Collect raw bond features
        for bond in mol.GetBonds():
            hybrid_begin = int(bond.GetBeginAtom().GetHybridization())
            hybrid_end = int(bond.GetEndAtom().GetHybridization())
            avg_hybrid = (hybrid_begin + hybrid_end) / 2.0
            bond_features['avg_hybrid'].append(avg_hybrid)

    # Calculate statistics for continuous features only with fixed key names
    stats = {
        'cont_mean': [
            torch.tensor(atom_features['atomic_num']).float().mean().item() if atom_features['atomic_num'] else 6.0,
            torch.tensor(atom_features['formal_charge']).float().mean().item() if atom_features['formal_charge'] else 0.0,
            torch.tensor(atom_features['total_num_hs']).float().mean().item() if atom_features['total_num_hs'] else 1.0
        ],
        'cont_std': [
            max(torch.tensor(atom_features['atomic_num']).float().std().item(), 1e-6) if atom_features['atomic_num'] else 3.0,
            max(torch.tensor(atom_features['formal_charge']).float().std().item(), 1e-6) if atom_features['formal_charge'] else 1.0,
            max(torch.tensor(atom_features['total_num_hs']).float().std().item(), 1e-6) if atom_features['total_num_hs'] else 1.0
        ],
        'bond_cont_means': [torch.tensor(bond_features['avg_hybrid']).float().mean().item() if bond_features['avg_hybrid'] else 2.0],
        'bond_cont_stds': [max(torch.tensor(bond_features['avg_hybrid']).float().std().item(), 1e-6) if bond_features['avg_hybrid'] else 1.0]
    }

    # Add categorical enumeration information with fixed key names
    stats['degree_cat'] = list(range(7))
    stats['hybridization_cats'] = [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    stats['atom_chiral_cats'] = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    stats['bond_stereo_cats'] = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY
    ]
    stats['bond_type_cats'] = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]

    return stats
