"""
Enhanced hydrogen bond identification module using RDKit ChemicalFeatures.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig
from typing import List, Tuple, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class HydrogenBondIdentifier:
    """Enhanced hydrogen bond identification using RDKit ChemicalFeatures standard."""

    # Class-level factory to avoid repeated initialization
    _factory = None
    _factory_initialized = False

    def __init__(self, config: Dict):
        """
        Initialize hydrogen bond identifier with RDKit ChemicalFeatures.

        Args:
            config: Configuration dictionary containing hydrogen bond parameters
        """
        self.config = config.get('hydrogen_bonds', {})
        self.base_dist_thresh = self.config.get('base_distance_threshold', 3.5)
        self.angle_threshold = self.config.get('angle_threshold', 120.0)
        self.donor_acceptor_dist = self.config.get('donor_acceptor_distance', 4.0)
        
        # For small molecules, use more relaxed criteria
        self.use_relaxed_criteria = self.config.get('use_relaxed_criteria', True)
        
        # Initialize RDKit ChemicalFeatures factory (only once per class)
        if not HydrogenBondIdentifier._factory_initialized:
            try:
                fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                HydrogenBondIdentifier._factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
                HydrogenBondIdentifier._factory_initialized = True
                logger.info("Successfully initialized RDKit ChemicalFeatures factory")
            except Exception as e:
                logger.error(f"Failed to initialize ChemicalFeatures factory: {e}")
                HydrogenBondIdentifier._factory = None
                HydrogenBondIdentifier._factory_initialized = True
        
        self.factory = HydrogenBondIdentifier._factory
        
        self.adjustments = self.config.get('adjustments', {
            'protein_like': 0.1,
            'dna_rna_like': 0.2,
            'carbohydrate': 0.15,
            'small_molecule': 0.0
        })

        # Electronegativity values for common elements
        self.electronegativities = {
            1: 2.20,  # H
            6: 2.55,  # C
            7: 3.04,  # N
            8: 3.44,  # O
            9: 3.98,  # F
            15: 2.19,  # P
            16: 2.58,  # S
            17: 3.16,  # Cl
            35: 2.96,  # Br
            53: 2.66  # I
        }

    def identify(self, mol: Chem.Mol) -> List[Tuple[List[int], torch.Tensor]]:
        """
        Identify hydrogen bonds using RDKit ChemicalFeatures standard.

        Args:
            mol: RDKit molecule with 3D coordinates

        Returns:
            List of tuples containing:
                - List[int]: [donor, hydrogen, acceptor] indices
                - torch.Tensor: Full hydrogen bond attributes (no dimension limit)
        """
        h_bonds = []

        # Ensure molecule has 3D coordinates
        if not mol.GetNumConformers():
            logger.warning("Molecule has no 3D coordinates, skipping hydrogen bond detection")
            return h_bonds

        if self.factory is None:
            logger.error("ChemicalFeatures factory not initialized")
            return h_bonds

        # Use RDKit ChemicalFeatures to find HBD and HBA
        try:
            features = self.factory.GetFeaturesForMol(mol)
            
            # Separate donors and acceptors using standard definitions
            donors = []
            acceptors = []
            
            for feat in features:
                if feat.GetFamily() == 'Donor':
                    donors.append(feat)
                elif feat.GetFamily() == 'Acceptor':
                    acceptors.append(feat)
            
            logger.debug(f"Found {len(donors)} donors and {len(acceptors)} acceptors")
            
        except Exception as e:
            logger.error(f"Error getting chemical features: {e}")
            return h_bonds

        # Determine molecule type and adjust thresholds
        mol_type = self._determine_molecule_type(mol)
        dist_thresh = self._adjust_distance_threshold(self.base_dist_thresh, mol_type)
        
        # Use relaxed criteria for small molecules if enabled
        if self.use_relaxed_criteria and mol.GetNumAtoms() < 50:
            dist_thresh = min(dist_thresh * 1.2, 4.0)  # Increase distance threshold by 20%, cap at 4.0
            angle_thresh = max(self.angle_threshold - 20.0, 90.0)  # Decrease angle requirement by 20°, min 90°
            donor_acceptor_thresh = min(self.donor_acceptor_dist * 1.2, 5.0)  # Increase by 20%, cap at 5.0
        else:
            angle_thresh = self.angle_threshold
            donor_acceptor_thresh = self.donor_acceptor_dist

        # Find hydrogen bonds between donors and acceptors
        for donor_feat in donors:
            for acceptor_feat in acceptors:
                # Get atom indices
                donor_atoms = list(donor_feat.GetAtomIds())
                acceptor_atoms = list(acceptor_feat.GetAtomIds())
                
                if not donor_atoms or not acceptor_atoms:
                    continue
                
                # For each donor-acceptor pair
                for donor_idx in donor_atoms:
                    for acceptor_idx in acceptor_atoms:
                        if donor_idx == acceptor_idx:
                            continue
                        
                        # Skip if atoms are directly bonded (too close for H-bond)
                        donor_atom = mol.GetAtomWithIdx(donor_idx)
                        if any(neighbor.GetIdx() == acceptor_idx for neighbor in donor_atom.GetNeighbors()):
                            continue
                        
                        # Find hydrogen atom bonded to donor
                        h_idx = self._find_hydrogen_atom(mol, donor_idx)
                        if h_idx is None:
                            continue
                        
                        # Check if this forms a valid hydrogen bond
                        h_bond_info = self._check_hydrogen_bond(
                            mol, donor_idx, h_idx, acceptor_idx, dist_thresh, 
                            angle_thresh, donor_acceptor_thresh
                        )
                        
                        if h_bond_info is not None:
                            atoms = [donor_idx, h_idx, acceptor_idx]
                            features = self._extract_features(h_bond_info, mol, donor_idx,
                                                              acceptor_idx, dist_thresh)
                            h_bonds.append((atoms, features))

        return h_bonds

    def _find_hydrogen_atom(self, mol: Chem.Mol, donor_idx: int) -> Optional[int]:
        """Find hydrogen atom bonded to a donor atom."""
        donor_atom = mol.GetAtomWithIdx(donor_idx)
        
        for neighbor in donor_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Hydrogen
                return neighbor.GetIdx()
        
        return None

    def _find_donors(self, mol: Chem.Mol) -> List[int]:
        """Find potential hydrogen bond donors (N, O with H)."""
        donors = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in {7, 8} and atom.GetTotalNumHs() > 0:
                donors.append(atom.GetIdx())
        return donors

    def _find_acceptors(self, mol: Chem.Mol) -> List[int]:
        """Find potential hydrogen bond acceptors (N, O or negatively charged atoms)."""
        acceptors = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in {7, 8} or atom.GetFormalCharge() < 0:
                acceptors.append(atom.GetIdx())
        return acceptors

    def _map_hydrogen_atoms(self, mol: Chem.Mol, donors: List[int]) -> Dict[int, int]:
        """Map donor atoms to their hydrogen atoms."""
        h_atom_map = {}

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:  # Hydrogen
                neighbors = atom.GetNeighbors()
                if len(neighbors) > 0:
                    neighbor = neighbors[0]
                    if neighbor.GetIdx() in donors:
                        h_atom_map[neighbor.GetIdx()] = atom.GetIdx()

        return h_atom_map

    def _check_hydrogen_bond(self, mol: Chem.Mol, donor_idx: int, h_idx: int,
                             acceptor_idx: int, dist_thresh: float, angle_thresh: float,
                             donor_acceptor_thresh: float) -> Optional[Dict]:
        """
        Check if atoms form a valid hydrogen bond.

        Returns:
            Dictionary with bond information if valid, None otherwise
        """
        try:
            # Calculate distances using 3D coordinates
            conf = mol.GetConformer()
            h_pos = conf.GetAtomPosition(h_idx)
            acceptor_pos = conf.GetAtomPosition(acceptor_idx)
            donor_pos = conf.GetAtomPosition(donor_idx)
            
            # Calculate distances manually
            h_acceptor_dist = ((h_pos.x - acceptor_pos.x)**2 + 
                              (h_pos.y - acceptor_pos.y)**2 + 
                              (h_pos.z - acceptor_pos.z)**2)**0.5
            donor_acceptor_dist = ((donor_pos.x - acceptor_pos.x)**2 + 
                                  (donor_pos.y - acceptor_pos.y)**2 + 
                                  (donor_pos.z - acceptor_pos.z)**2)**0.5

            # Calculate angle
            angle = self._calculate_angle(mol, donor_idx, h_idx, acceptor_idx)

            # Check criteria using passed thresholds
            if (h_acceptor_dist < dist_thresh and
                    angle > angle_thresh and
                    donor_acceptor_dist < donor_acceptor_thresh):
                return {
                    'h_acceptor_dist': h_acceptor_dist,
                    'donor_acceptor_dist': donor_acceptor_dist,
                    'angle': angle,
                    'strength': 1.0 - (h_acceptor_dist / dist_thresh)
                }

        except Exception as e:
            logger.debug(f"Error checking hydrogen bond: {e}")

        return None

    def _calculate_angle(self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int,
                         atom3_idx: int) -> float:
        """Calculate the angle between three atoms in degrees."""
        try:
            conf = mol.GetConformer()
            point1 = conf.GetAtomPosition(atom1_idx)
            point2 = conf.GetAtomPosition(atom2_idx)
            point3 = conf.GetAtomPosition(atom3_idx)

            # Convert to numpy arrays
            v1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
            v2 = np.array([point3.x - point2.x, point3.y - point2.y, point3.z - point2.z])

            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            # Calculate angle
            dot_product = np.dot(v1_norm, v2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical errors
            angle_rad = np.arccos(dot_product)

            return np.degrees(angle_rad)

        except Exception as e:
            logger.debug(f"Error calculating angle: {e}")
            return 0.0

    def _determine_molecule_type(self, mol: Chem.Mol) -> str:
        """
        Determine the type of molecule for parameter adjustment.

        Returns:
            Molecule type string
        """
        # Count functional groups
        amide_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")
        phosphate_pattern = Chem.MolFromSmarts("[PX4](=[OX1])([OX2H0])([OX2H0])([OX2H0])")

        amide_count = len(mol.GetSubstructMatches(amide_pattern)) if amide_pattern else 0
        phosphate_count = len(mol.GetSubstructMatches(phosphate_pattern)) if phosphate_pattern else 0

        # Count atoms
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        o_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

        # Count rings
        ring_count = len(Chem.GetSymmSSSR(mol))

        # Classify molecule
        if amide_count >= 3:
            return "protein_like"
        elif phosphate_count >= 1 and ring_count >= 3:
            return "dna_rna_like"
        elif o_count > 0 and c_count > 0 and (o_count / c_count) > 0.5:
            return "carbohydrate"
        else:
            return "small_molecule"

    def _adjust_distance_threshold(self, base_thresh: float, mol_type: str) -> float:
        """Adjust hydrogen bond distance threshold based on molecule type."""
        adjustment = self.adjustments.get(mol_type, 0.0)
        return base_thresh + adjustment

    def _extract_features(self, h_bond_info: Dict, mol: Chem.Mol,
                          donor_idx: int, acceptor_idx: int, dist_thresh: float) -> torch.Tensor:
        """Extract full features for a hydrogen bond including configurable thresholds."""
        donor_atom = mol.GetAtomWithIdx(donor_idx)
        acceptor_atom = mol.GetAtomWithIdx(acceptor_idx)

        # Calculate electronegativity difference
        en_donor = self.electronegativities.get(donor_atom.GetAtomicNum(), 2.0)
        en_acceptor = self.electronegativities.get(acceptor_atom.GetAtomicNum(), 2.0)
        electro_diff = abs(en_donor - en_acceptor)

        # Create comprehensive feature vector (no dimension limit)
        features = torch.tensor([
            h_bond_info['strength'],  # Bond strength (0-1)
            h_bond_info['angle'] / 180.0,  # Normalized angle (0-1)
            electro_diff / 2.0,  # Normalized electronegativity difference
            h_bond_info['h_acceptor_dist'] / 5.0,  # Normalized H-acceptor distance
            float(donor_atom.GetAtomicNum()) / 20.0,  # Normalized donor atomic number
            float(acceptor_atom.GetAtomicNum()) / 20.0,  # Normalized acceptor atomic number
            h_bond_info['donor_acceptor_dist'] / 5.0,  # Normalized donor-acceptor distance
            dist_thresh / 5.0,  # Normalized distance threshold (configurable)
            self.angle_threshold / 180.0,  # Normalized angle threshold (configurable)
            float(donor_atom.GetFormalCharge()),  # Donor formal charge
            float(acceptor_atom.GetFormalCharge()),  # Acceptor formal charge
            float(donor_atom.GetDegree()) / 6.0,  # Normalized donor degree
            float(acceptor_atom.GetDegree()) / 6.0,  # Normalized acceptor degree
            1.0 if donor_atom.GetIsAromatic() else 0.0,  # Donor aromaticity
            1.0 if acceptor_atom.GetIsAromatic() else 0.0,  # Acceptor aromaticity
        ], dtype=torch.float32)

        return features  # Return full feature vector without truncation


def create_hydrogen_bond_features(mol: Chem.Mol, config: Dict) -> List[Tuple[List[int], torch.Tensor]]:
    """
    Create hydrogen bond features for a molecule using RDKit ChemicalFeatures.

    Args:
        mol: RDKit molecule with 3D coordinates
        config: Configuration dictionary

    Returns:
        List of hydrogen bond features with full feature vectors
    """
    identifier = HydrogenBondIdentifier(config)
    return identifier.identify(mol)
