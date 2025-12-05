"""
Molecular hypergraph construction utilities - Single Source of Truth.
All dimensions dynamically determined and embedded in Data objects.
"""

import torch
import numpy as np
import logging
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from torch_geometric.data import Data
from typing import Optional, List, Tuple, Dict
import torch.nn.functional as F
from scipy.spatial import cKDTree  # For KNN hyperedge construction
from .functional_groups import identify_all_functional_groups
from .hydrogen_bonds import HydrogenBondIdentifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical feature dimensions for each hyperedge type (fixed per type)
# Dynamic dimension management - no fixed canonical dimensions
# All dimensions determined dynamically from actual feature extraction

def parse_hypergraph_types(config: Dict) -> List[str]:
    """
    Parse hypergraph types from config with backward compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of hyperedge types to construct
        
    Examples:
        # New format (preferred)
        config = {'hypergraph_types': ['bond', 'ring', 'functional_group']}
        
        # Old format (backward compatibility)  
        config = {'hypergraph_type': 'full'}  # -> all 5 types
        config = {'hypergraph_type': 'bonds_rings'}  # -> bond + ring
    """
    # New format takes priority
    if 'hypergraph_types' in config:
        types = config['hypergraph_types']
        if isinstance(types, str):
            # Handle single string case
            types = [types]
        # Validate and normalize
        valid_types = ['bond', 'ring', 'functional_group', 'hydrogen_bond', 'conjugated_system']
        normalized = []
        for t in types:
            if t in valid_types:
                normalized.append(t)
            else:
                logger.warning(f"Unknown hyperedge type '{t}', skipping")
        
        # Special case: empty list means generate standard graph (no hypergraph)
        if len(normalized) == 0:
            return []
        
        # Ensure 'bond' is always included as base (unless explicitly empty)
        if 'bond' not in normalized:
            normalized.insert(0, 'bond')
        
        return normalized
    
    # Backward compatibility for old format
    old_type = config.get('hypergraph_type', 'bonds')
    old_mappings = {
        'full': ['bond', 'ring', 'functional_group', 'hydrogen_bond', 'conjugated_system'],
        'bonds_rings': ['bond', 'ring'],
        'bonds': ['bond'], 
        'graph_baseline': ['bond']
    }
    
    result = old_mappings.get(old_type, ['bond'])
    logger.debug(f"Mapped old hypergraph_type '{old_type}' to hypergraph_types {result}")
    return result

def get_canonical_fg_dim(fg_types: List[str]) -> int:
    """Get canonical functional group feature dimension."""
    return 6 + len(fg_types)  # 6 continuous + len(fg_types) one-hot


def compute_unified_hyperedge_dim(hyperedge_data: List[Dict], config: Dict) -> int:
    """
    Compute unified hyperedge dimension as single source of truth.
    
    Args:
        hyperedge_data: List of hyperedge data with type and features
        config: Configuration dictionary
        
    Returns:
        Unified dimension for all hyperedges
    """
    if not hyperedge_data:
        return 1  # Safe minimum for empty graphs
    
    # Collect all feature dimensions by type
    dims_by_type = {}
    
    for item in hyperedge_data:
        edge_type = item['type']
        actual_dim = item['features'].shape[0] if item['features'].dim() > 0 else 0
        
        if edge_type not in dims_by_type:
            dims_by_type[edge_type] = []
        dims_by_type[edge_type].append(actual_dim)
    
    # Verify consistency within each type
    type_dims = {}
    for edge_type, dims in dims_by_type.items():
        unique_dims = set(dims)
        if len(unique_dims) > 1:
            logger.warning(f"Inconsistent dimensions for {edge_type}: {unique_dims}")
        type_dims[edge_type] = max(dims) if dims else 0
    
    # Compute unified dimension (max across all types)
    unified_dim = max(type_dims.values()) if type_dims else 1
    
    logger.debug(f"Computed unified hyperedge dimension: {unified_dim}")
    logger.debug(f"Type dimensions: {type_dims}")
    
    return unified_dim


def ensure_canonical_feature_dims(hyperedge_data: List[Dict], config: Dict) -> List[Dict]:
    """
    Ensure all hyperedges of the same type have canonical dimensions.
    
    Args:
        hyperedge_data: List of hyperedge data
        config: Configuration dictionary
        
    Returns:
        Hyperedge data with canonical dimensions per type
    """
    # Group by type and find max dimension within each type
    dims_by_type = {}
    
    for item in hyperedge_data:
        edge_type = item['type']
        current_dim = item['features'].shape[0] if item['features'].dim() > 0 else 0
        
        if edge_type not in dims_by_type:
            dims_by_type[edge_type] = current_dim
        else:
            dims_by_type[edge_type] = max(dims_by_type[edge_type], current_dim)
    
    # Pad each type to its maximum observed dimension
    for item in hyperedge_data:
        edge_type = item['type']
        features = item['features']
        expected_dim = dims_by_type[edge_type]
        
        current_dim = features.shape[0] if features.dim() > 0 else 0
        
        if current_dim < expected_dim:
            # Pad to type's maximum dimension
            padding = torch.zeros(expected_dim - current_dim, 
                                dtype=features.dtype, device=features.device)
            item['features'] = torch.cat([features, padding])
    
    return hyperedge_data


def pad_to_unified_dim(hyperedge_data: List[Dict], unified_dim: int) -> List[torch.Tensor]:
    """
    Pad all features to unified dimension for cross-type compatibility.
    
    Args:
        hyperedge_data: List with canonical-dimension features
        unified_dim: Target unified dimension
        
    Returns:
        List of padded feature tensors
    """
    padded_features = []
    
    for item in hyperedge_data:
        features = item['features']
        current_dim = features.shape[0] if features.dim() > 0 else 0
        
        if current_dim < unified_dim:
            padding = torch.zeros(unified_dim - current_dim, 
                                dtype=features.dtype, device=features.device)
            padded_features.append(torch.cat([features, padding]))
        else:
            padded_features.append(features)
    
    return padded_features

# Timeout exception class
class TimeoutException(Exception):
    """Timeout exception class"""
    pass


def get_atom_features(atom: Chem.Atom, cont_mean, cont_std, degree_cat, hybrid_cat, chiral_cat) -> torch.Tensor:
    """Extract atom features with consistent interface"""
    # Continuous features
    cont_features = [
        atom.GetAtomicNum(),  # Atomic number
        atom.GetFormalCharge(),  # Formal charge
        atom.GetTotalNumHs()  # Total number of hydrogens
    ]
    cont_tensor = (torch.tensor(cont_features, dtype=torch.float32) - cont_mean) / cont_std

    # Discrete features with direct enum comparison
    degree = atom.GetDegree()
    degree_onehot = torch.zeros(len(degree_cat))
    if degree < len(degree_cat):
        degree_onehot[degree] = 1.0

    is_aromatic = atom.GetIsAromatic()
    aromatic_onehot = torch.tensor([1.0, 0.0] if is_aromatic else [0.0, 1.0])

    # Direct enum comparison for hybridization
    hybrid = atom.GetHybridization()
    hybrid_onehot = torch.zeros(len(hybrid_cat))
    for i, h in enumerate(hybrid_cat):
        if hybrid == h:
            hybrid_onehot[i] = 1.0
            break

    # Direct enum comparison for chiral tag
    chiral = atom.GetChiralTag()
    chiral_onehot = torch.zeros(len(chiral_cat))
    for i, c in enumerate(chiral_cat):
        if chiral == c:
            chiral_onehot[i] = 1.0
            break

    is_in_ring = atom.IsInRing()
    ring_onehot = torch.tensor([1.0, 0.0] if is_in_ring else [0.0, 1.0])

    disc_features = torch.cat([degree_onehot, aromatic_onehot, hybrid_onehot, chiral_onehot, ring_onehot])
    return torch.cat([cont_tensor, disc_features])


def get_bond_features(bond: Chem.Bond, mol: Chem.Mol, cont_mean, cont_std, stereo_cat, bond_type_cat) -> torch.Tensor:
    """Extract bond features using RDKit enums"""
    # Continuous features
    hybrid_begin = bond.GetBeginAtom().GetHybridization()
    hybrid_end = bond.GetEndAtom().GetHybridization()
    avg_hybrid = (int(hybrid_begin) + int(hybrid_end)) / 2.0
    cont_tensor = (torch.tensor([avg_hybrid], dtype=torch.float32) - cont_mean) / cont_std

    # Discrete features using direct enum comparison
    conjugated = bond.GetIsConjugated()
    conjugated_onehot = torch.tensor([1.0, 0.0] if conjugated else [0.0, 1.0])

    aromatic = bond.GetIsAromatic()
    aromatic_onehot = torch.tensor([1.0, 0.0] if aromatic else [0.0, 1.0])

    # Direct enum comparison for stereo
    stereo = bond.GetStereo()
    stereo_onehot = torch.zeros(len(stereo_cat))
    for i, s in enumerate(stereo_cat):
        if stereo == s:
            stereo_onehot[i] = 1.0
            break

    in_ring = bond.IsInRing()
    in_ring_onehot = torch.tensor([1.0, 0.0] if in_ring else [0.0, 1.0])

    # Check if rotatable
    rotatable = (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and 
                 not bond.IsInRing() and
                 bond.GetBeginAtom().GetDegree() > 1 and 
                 bond.GetEndAtom().GetDegree() > 1)
    rotatable_onehot = torch.tensor([1.0, 0.0] if rotatable else [0.0, 1.0])

    # Direct enum comparison for bond type
    bond_type = bond.GetBondType()
    bond_type_onehot = torch.zeros(len(bond_type_cat))
    for i, bt in enumerate(bond_type_cat):
        if bond_type == bt:
            bond_type_onehot[i] = 1.0
            break

    disc_features = torch.cat([conjugated_onehot, aromatic_onehot, stereo_onehot, 
                               in_ring_onehot, rotatable_onehot, bond_type_onehot])

    return torch.cat([cont_tensor, disc_features])


def get_ring_features(ring, mol: Chem.Mol) -> torch.Tensor:
    """Extract ring structure features"""
    ring_size = len(ring)
    atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
    
    features = [
        float(ring_size),  # Ring size
        1.0 if all(atom.GetIsAromatic() for atom in atoms) else 0.0,  # Is aromatic
        1.0 if any(atom.GetAtomicNum() != 6 for atom in atoms) else 0.0,  # Is heterocycle
        sum(atom.GetAtomicNum() for atom in atoms) / ring_size,  # Average atomic number
        sum(atom.GetDegree() for atom in atoms) / ring_size,  # Average degree
        sum(atom.GetFormalCharge() for atom in atoms) / ring_size,  # Average formal charge
        sum(atom.GetTotalNumHs() for atom in atoms) / ring_size  # Average hydrogen count
    ]
    
    features = torch.tensor(features, dtype=torch.float32)
    return F.normalize(features, p=2, dim=0)


def identify_hydrogen_bonds_simple(mol: Chem.Mol) -> List[Tuple[List[int], torch.Tensor]]:
    """Simple hydrogen bond identification with consistent 3D features"""
    h_bonds = []

    # Ensure molecule has 3D coordinates
    if not mol.GetNumConformers():
        return h_bonds

    # Find donors (N, O with H)
    donors = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {7, 8} and atom.GetTotalNumHs() > 0:
            donors.append(atom.GetIdx())

    # Find acceptors (N, O or negatively charged atoms)
    acceptors = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {7, 8} or atom.GetFormalCharge() < 0:
            acceptors.append(atom.GetIdx())

    # Map hydrogen atoms to donors
    h_atom_map = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen atom
            neighbors = atom.GetNeighbors()
            if len(neighbors) > 0:
                neighbor = neighbors[0]
                if neighbor.GetIdx() in donors:
                    h_atom_map[neighbor.GetIdx()] = atom.GetIdx()

    # Find hydrogen bonds
    for donor in donors:
        if donor not in h_atom_map:
            continue

        h_atom = h_atom_map[donor]

        for acceptor in acceptors:
            if donor == acceptor:
                continue

            try:
                # Calculate distances and angle using 3D coordinates
                conf = mol.GetConformer()
                h_pos = conf.GetAtomPosition(h_atom)
                acceptor_pos = conf.GetAtomPosition(acceptor)
                donor_pos = conf.GetAtomPosition(donor)
                
                h_acceptor_dist = ((h_pos.x - acceptor_pos.x)**2 + 
                                  (h_pos.y - acceptor_pos.y)**2 + 
                                  (h_pos.z - acceptor_pos.z)**2)**0.5
                donor_acceptor_dist = ((donor_pos.x - acceptor_pos.x)**2 + 
                                      (donor_pos.y - acceptor_pos.y)**2 + 
                                      (donor_pos.z - acceptor_pos.z)**2)**0.5
                angle = calculate_angle(mol, donor, h_atom, acceptor)

                # Check hydrogen bond criteria
                if h_acceptor_dist < 3.5 and angle > 120.0 and donor_acceptor_dist < 4.0:
                    # Create 14-dimensional feature vector to match canonical dimension
                    strength = 1.0 - (h_acceptor_dist / 3.5)
                    
                    # Get atomic features for comprehensive vector
                    donor_atom = mol.GetAtomWithIdx(donor)
                    acceptor_atom = mol.GetAtomWithIdx(acceptor)
                    
                    attr = torch.tensor([
                        strength,  # Bond strength
                        angle / 180.0,  # Normalized angle  
                        0.8,  # Electronegativity difference (approximate)
                        h_acceptor_dist / 5.0,  # Normalized H-acceptor distance
                        float(donor_atom.GetAtomicNum()) / 20.0,  # Normalized donor atomic number
                        float(acceptor_atom.GetAtomicNum()) / 20.0,  # Normalized acceptor atomic number
                        donor_acceptor_dist / 5.0,  # Normalized donor-acceptor distance
                        3.5 / 5.0,  # Normalized distance threshold
                        120.0 / 180.0,  # Normalized angle threshold
                        float(donor_atom.GetFormalCharge()),  # Donor formal charge
                        float(acceptor_atom.GetFormalCharge()),  # Acceptor formal charge
                        float(donor_atom.GetDegree()) / 6.0,  # Normalized donor degree
                        float(acceptor_atom.GetDegree()) / 6.0,  # Normalized acceptor degree
                        1.0 if donor_atom.GetIsAromatic() else 0.0  # Donor aromaticity
                    ], dtype=torch.float32)

                    h_bonds.append(([donor, h_atom, acceptor], attr))
            except:
                continue

    return h_bonds


def calculate_angle(mol: Chem.Mol, atom1_idx: int, atom2_idx: int, atom3_idx: int) -> float:
    """Calculate angle between three atoms in degrees"""
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
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        return np.degrees(angle_rad)
    except:
        return 0.0


def smiles_to_hypergraph(smiles: str, mol_id: str, config: dict, global_stats: dict, device: torch.device) -> Optional[Data]:
    """
    Fixed SMILES to hypergraph conversion with 30s timeout.
    """
    # Suppress RDKit warnings during processing
    rdBase.DisableLog('rdApp.warning')
    rdBase.DisableLog('rdApp.error')
    
    # Get timeout from config, default to 30 seconds
    timeout_seconds = config.get('timeout_seconds', 30)
    
    try:
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        # Check for problematic metals (configurable)
        problematic_metals = config.get('filter', {}).get('problematic_metals', 
                                                          ['Co', 'Fe', 'Ni', 'Pt', 'Pd', 'Ru', 'Os', 'Ir', 'Rh'])  # Only transition metals by default
        if problematic_metals:
            metals_pattern_smarts = '[' + ','.join(problematic_metals) + ']'
            problematic_metals_pattern = Chem.MolFromSmarts(metals_pattern_smarts)
            if problematic_metals_pattern and mol.HasSubstructMatch(problematic_metals_pattern):
                logger.warning(f"Skipping molecule with problematic metals: {mol_id}")
                return None

        start_time = time.time()
        
        def check_timeout():
            """Check if processing time exceeds timeout"""
            if time.time() - start_time > timeout_seconds:
                raise TimeoutException(f"Processing timeout ({timeout_seconds}s) exceeded for molecule {mol_id}")

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        check_timeout()

        # Embedding with fallbacks
        embed_success = False
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100) >= 0:
            embed_success = True
        elif AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=50) >= 0:
            embed_success = True
        
        if not embed_success:
            logger.warning(f"Embedding failed for {smiles}, using 2D coordinates")
            AllChem.Compute2DCoords(mol)
        
        check_timeout()

        # Force field optimization with bounded iterations to avoid stalls
        ff_max_iters = int(config.get('ff_max_iters', 200))
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=ff_max_iters)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=ff_max_iters)
            except Exception:
                logger.warning(f"Force field optimization failed for {smiles}")
        
        check_timeout()

        num_atoms = mol.GetNumAtoms()
        
        # Extract atom features with fixed stats key names
        cont_mean = torch.tensor(global_stats['cont_mean'])
        cont_std = torch.tensor(global_stats['cont_std'])
        
        x = torch.stack([get_atom_features(atom, cont_mean, cont_std,
                                           global_stats['degree_cat'],
                                           global_stats['hybridization_cats'],
                                           global_stats['atom_chiral_cats'])
                         for atom in mol.GetAtoms()]).to(device)
        
        check_timeout()

        # Collect all hyperedges with consistent feature dimensions
        hyperedge_data = []
        seen_hyperedges = set()  # Use (atoms_tuple, edge_type) for deduplication
        
        # Parse hypergraph types from config (supports both old and new formats)
        hypergraph_types = parse_hypergraph_types(config)
        
        # Special case: if hypergraph_types is empty, generate standard graph
        if len(hypergraph_types) == 0:
            return construct_standard_graph(smiles, mol, mol_id, x, global_stats, config, device, start_time)

        # Bond hyperedges (always included - needed for all hypergraph types)
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            edge_atoms = tuple(sorted([begin_atom, end_atom]))
            
            hyperedge_key = (edge_atoms, 'bond')
            if hyperedge_key in seen_hyperedges:
                continue
            seen_hyperedges.add(hyperedge_key)
            
            bond_features = get_bond_features(bond, mol,
                                              torch.tensor(global_stats['bond_cont_means']),
                                              torch.tensor(global_stats['bond_cont_stds']),
                                              global_stats['bond_stereo_cats'],
                                              global_stats['bond_type_cats'])
            
            # Store features without padding (will be done in second pass)
            hyperedge_data.append({
                'atoms': list(edge_atoms),
                'features': bond_features,
                'type': 'bond'
            })

        check_timeout()
        
        # Additional hyperedges based on selected types
        check_timeout()
        
        # Ring hyperedges
        if 'ring' in hypergraph_types:
            sssr = Chem.GetSymmSSSR(mol)
            for ring in sssr:
                if len(ring) >= 3:
                    ring_atoms = tuple(sorted(ring))
                    
                    hyperedge_key = (ring_atoms, 'ring')
                    if hyperedge_key in seen_hyperedges:
                        continue
                    seen_hyperedges.add(hyperedge_key)
                    
                    ring_features = get_ring_features(list(ring), mol)
                    
                    # Store features without padding (will be done in second pass)
                    hyperedge_data.append({
                        'atoms': list(ring_atoms),
                        'features': ring_features,
                        'type': 'ring'
                    })
        
        check_timeout()

        # Functional group hyperedges
        if 'functional_group' in hypergraph_types:
            try:
                functional_group_results = identify_all_functional_groups(mol, config or {})
                for fg_atoms, fg_type, fg_features in functional_group_results:
                    check_timeout()
                    if len(fg_atoms) >= 2:
                        fg_atoms_tuple = tuple(fg_atoms)  # Don't sort to preserve structure
                        
                        hyperedge_key = (fg_atoms_tuple, 'functional_group')
                        if hyperedge_key in seen_hyperedges:
                            continue
                        seen_hyperedges.add(hyperedge_key)
                        
                        # Store features without padding (will be done in second pass) 
                        hyperedge_data.append({
                            'atoms': list(fg_atoms_tuple),
                            'features': fg_features,
                            'type': 'functional_group'
                        })
            except Exception as e:
                logger.warning(f"Functional group identification failed: {e}")
                # Fallback to simplified version if module fails
        
        check_timeout()

        # Hydrogen bond hyperedges (skip only this stage on 3D failure or time-budget exhaustion)
        if 'hydrogen_bond' in hypergraph_types:
            # Only attempt if 3D embedding succeeded and we still have some time budget
            try:
                remaining = timeout_seconds - (time.time() - start_time)
                if mol.GetNumConformers() == 0:
                    raise RuntimeError('No 3D conformer; skip hydrogen bonds')
                if remaining <= 3.0:
                    raise RuntimeError('Insufficient time budget for hydrogen bonds; skip')

                h_bond_identifier = HydrogenBondIdentifier(config or {})
                h_bonds = h_bond_identifier.identify(mol)
                for h_bond_atoms, h_bond_attrs in h_bonds:
                    # Do not trigger global timeout here; rely on remaining time guard
                    if len(h_bond_atoms) == 3:
                        h_bond_tuple = tuple(h_bond_atoms)
                        hyperedge_key = (h_bond_tuple, 'hydrogen_bond')
                        if hyperedge_key in seen_hyperedges:
                            continue
                        seen_hyperedges.add(hyperedge_key)
                        hyperedge_data.append({
                            'atoms': list(h_bond_tuple),
                            'features': h_bond_attrs,
                            'type': 'hydrogen_bond'
                        })
            except Exception as e:
                logger.debug(f"Hydrogen bond stage skipped: {e}")
                # soft-skip hydrogen bonds only; keep other types intact
        
        # Avoid strict timeout during H-bond stage to prevent dropping molecule

        # Conjugated system hyperedges
        if 'conjugated_system' in hypergraph_types:
            try:
                from .functional_groups import ConjugatedSystemIdentifier
                conj_identifier = ConjugatedSystemIdentifier()
                conjugated_systems = conj_identifier.identify(mol)
                
                for system in conjugated_systems:
                    check_timeout()
                    if len(system) >= 3:
                        system_tuple = tuple(system)  # Don't sort to preserve structure
                        
                        hyperedge_key = (system_tuple, 'conjugated_system')
                        if hyperedge_key in seen_hyperedges:
                            continue
                        seen_hyperedges.add(hyperedge_key)
                        
                        # Use modular implementation features (6 dimensions)
                        conj_features = conj_identifier.extract_features(system, mol)
                        
                        # Store features without padding (will be done in second pass)
                        hyperedge_data.append({
                            'atoms': list(system_tuple),
                            'features': conj_features,
                            'type': 'conjugated_system'
                        })
            except Exception as e:
                logger.warning(f"Modular conjugated system identification failed: {e}")
                # Fallback to simplified version
                visited_bonds = set()
                for bond in mol.GetBonds():
                    check_timeout()
                    if bond.GetIdx() in visited_bonds or not bond.GetIsConjugated():
                        continue
                        
                    system = set()
                    stack = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                    visited_bonds.add(bond.GetIdx())
                    
                    while stack:
                        check_timeout()
                        atom_idx = stack.pop()
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for b in atom.GetBonds():
                            if b.GetIsConjugated() and b.GetIdx() not in visited_bonds:
                                visited_bonds.add(b.GetIdx())
                                system.update([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
                                stack.extend([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
                    
                    if len(system) >= 3:
                        system_tuple = tuple(system)
                        
                        hyperedge_key = (system_tuple, 'conjugated_system')
                        if hyperedge_key in seen_hyperedges:
                            continue
                        seen_hyperedges.add(hyperedge_key)
                        
                        # Simple 3D features as fallback
                        conj_features = torch.tensor([
                            1.0,  # Conjugated system indicator
                            len(system) / 50.0,  # Normalized system size
                            sum(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in system) / len(system)  # Aromaticity ratio
                        ], dtype=torch.float32)
                        
                        # Store features without padding (will be done in second pass)
                        hyperedge_data.append({
                            'atoms': list(system_tuple),
                            'features': conj_features,
                                'type': 'conjugated_system'
                            })
        
        check_timeout()

        # Create final hypergraph structure - Single Source of Truth
        if hyperedge_data:
            hyperedge_list = [item['atoms'] for item in hyperedge_data]
            
            # Step 1: Ensure canonical dimensions within each type
            hyperedge_data = ensure_canonical_feature_dims(hyperedge_data, config)
            check_timeout()
            
            # Step 2: Compute unified dimension across all types
            unified_dim = compute_unified_hyperedge_dim(hyperedge_data, config)
            check_timeout()
            
            # Step 3: Pad all features to unified dimension
            padded_features = pad_to_unified_dim(hyperedge_data, unified_dim)
            hyperedge_attr = torch.stack(padded_features).to(device)
            check_timeout()
            
            # Create hyperedge_index directly (no need for DHG)
            hyperedge_index = []
            for i, edge in enumerate(hyperedge_list):
                for node in edge:
                    hyperedge_index.append([node, i])
            
            hyperedge_index = torch.tensor(hyperedge_index).t().contiguous().to(device)

            # Per-edge type ids for downstream use (e.g., SHAP labeling)
            type_names_all = [str(item.get('type', 'unknown')) for item in hyperedge_data]
            unique_types = sorted(set(type_names_all)) if type_names_all else ['unknown']
            type_to_id = {t: i for i, t in enumerate(unique_types)}
            hyperedge_type = torch.tensor([type_to_id[t] for t in type_names_all], dtype=torch.long, device=device)

        else:
            # Empty hypergraph - use canonical minimum dimension
            unified_dim = 1  # Canonical empty graph dimension
            hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            hyperedge_attr = torch.empty((0, unified_dim), dtype=torch.float32, device=device)
            hyperedge_type = torch.empty((0,), dtype=torch.long, device=device)

        # Calculate processing time first
        processing_time = time.time() - start_time

        # Create final data object with embedded dimension metadata (no edge_index)
        # NOTE: Masks are handled dynamically during training, not in preprocessing
        data = Data(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            hyperedge_type=hyperedge_type,
            id=mol_id,
            smiles=smiles,
            num_nodes=num_atoms,
            # Single Source of Truth: Embed dimensions in data
            hyperedge_dim=unified_dim,  # Unified hyperedge feature dimension
            num_hyperedges=hyperedge_attr.shape[0],  # Number of hyperedges
            # Metadata for downstream use
            meta_info={
                'hyperedge_types': list(set(item['type'] for item in hyperedge_data)) if hyperedge_data else [],
                'hyperedge_type_map': {int(i): t for t, i in type_to_id.items()} if hyperedge_data else {},
                'dynamic_dims': True,  # Using dynamic dimension calculation
                'processing_time': processing_time
            }
        )

        # Provide a trivial RDKit atom index mapping (node -> atom idx),
        # assuming nodes correspond to atoms in order during preprocessing.
        try:
            data.rdkit_idx = torch.arange(num_atoms, dtype=torch.long, device=device)
        except Exception:
            pass

        return data

    except TimeoutException as e:
        logger.warning(f"Timeout processing {smiles}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Processing {smiles} failed: {str(e)}")
        return None
    finally:
        # Re-enable RDKit logging
        rdBase.EnableLog('rdApp.warning')
        rdBase.EnableLog('rdApp.error')


def construct_standard_graph(smiles, mol, mol_id, x, global_stats, config, device, start_time):
    """
    Construct standard PyTorch Geometric graph (not hypergraph) with edge_index.
    Used when hypergraph_types is empty list.
    """
    try:
        # Create edge_index and edge_attr for standard graph
        edge_list = []
        edge_features = []
        
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_list.append([begin_atom, end_atom])
            edge_list.append([end_atom, begin_atom])
            
            # Get bond features
            bond_features = get_bond_features(bond, mol,
                                            torch.tensor(global_stats['bond_cont_means']),
                                            torch.tensor(global_stats['bond_cont_stds']),
                                            global_stats['bond_stereo_cats'],
                                            global_stats['bond_type_cats'])
            
            # Add same features for both directions
            edge_features.append(bond_features)
            edge_features.append(bond_features)
        
        # Create tensors
        if edge_list:
            edge_index = torch.tensor(edge_list).t().contiguous().to(device)
            edge_attr = torch.stack(edge_features).to(device)
        else:
            # Empty graph case
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 19), dtype=torch.float32, device=device)  # 19 is bond feature dim
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create standard graph data object (with edge_index, not hyperedge_index)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            id=mol_id,
            smiles=smiles,
            num_nodes=mol.GetNumAtoms(),
            # Standard graph metadata
            num_edges=edge_attr.shape[0],
            edge_dim=edge_attr.shape[1] if edge_attr.shape[0] > 0 else 19,
            # Metadata for downstream use
            meta_info={
                'graph_type': 'standard',  # Mark as standard graph, not hypergraph
                'processing_time': processing_time
            }
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to construct standard graph for {smiles}: {str(e)}")
        return None


def construct_knn_hyperedges(mol, k=3):
    """Construct KNN hyperedges as fallback"""
    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 1:
        return []

    actual_k = min(k + 1, num_atoms)
    if actual_k <= 1:
        return []

    try:
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])
        tree = cKDTree(coords)
        hyperedges = []

        for i in range(num_atoms):
            dists, indices = tree.query(coords[i], k=actual_k)
            neighbors = indices[1:] if len(indices) > 1 else []
            if len(neighbors) > 0:
                hyperedges.append([i] + list(neighbors))
        return hyperedges
    except Exception as e:
        logger.warning(f"KNN hyperedge construction failed: {str(e)}")
        return []
