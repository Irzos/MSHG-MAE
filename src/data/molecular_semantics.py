"""
Molecular semantic analysis for advanced masking strategies.

This module provides tools to analyze molecular structure and identify
semantic units like functional groups, rings, chains, and complex fragments
for use with sophisticated masking strategies like Curriculum-Mask.
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from typing import Dict, List, Tuple, Set, Optional, Any
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MolecularSemanticAnalyzer:
    """
    Comprehensive molecular semantic analyzer for identifying
    meaningful chemical fragments and structural patterns.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Extended SMARTS library; include pharmacologically relevant groups
        self.functional_group_patterns = {
            # Basic functional groups
            'carboxyl': '[CX3](=O)[OX2H1]',
            'carboxylate': '[CX3](=O)[OX1-]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'amino_tertiary': '[NX3;H0;!$(NC=O)]',
            'nitro': '[NX3](=O)=O',
            'cyano': '[CX2]#N',
            'aldehyde': '[CX3H1](=O)[#6]',
            'ketone': '[#6][CX3](=O)[#6]',
            'ester': '[#6][CX3](=O)[OX2][#6]',
            'amide': '[NX3][CX3](=[OX1])[#6]',
            'ether': '[OD2]([#6])[#6]',
            'thiol': '[SX2H]',
            'disulfide': '[SX2][SX2]',
            'halogen': '[F,Cl,Br,I]',
            
            # Sulfur-containing groups
            'sulfone': '[SX4](=[OX1])(=[OX1])([#6])[#6]',
            'sulfonate': '[SX4](=[OX1])(=[OX1])([OX1-])',
            'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])',
            'sulfate': '[SX4](=[OX1])(=[OX1])([OX2])([OX2])',
            
            # Phosphorus-containing groups
            'phosphate': '[PX4](=[OX1])([OX2])([OX2])([OX2])',
            'phosphonate': '[PX4](=[OX1])([OX2])([OX2])([#6])',
            'phosphoramide': '[PX4](=[OX1])([NX3])([OX2])([OX2])',
            
            # Aromatic systems (explicit patterns)
            'phenyl': 'c1ccccc1',
            'benzyl': '[#6]c1ccccc1',
            'pyridyl': 'c1ccncc1',
            'pyrimidinyl': 'c1cncnc1',
            'imidazolyl': 'c1c[nH]cn1',
            'triazolyl': 'c1nnc[nH]1',
            'tetrazolyl': 'c1nnn[nH]1',
            'furanyl': 'c1ccoc1',
            'thiopheneyl': 'c1ccsc1',
            'pyrrolyl': 'c1cc[nH]c1',
            
            # Polycyclic aromatics
            'indolyl': 'c1ccc2[nH]ccc2c1',
            'quinolinyl': 'c1ccc2ncccc2c1',
            'isoquinolinyl': 'c1cnc2ccccc2c1',
            'benzofuranyl': 'c1ccc2occc2c1',
            'benzothienyl': 'c1ccc2sccc2c1',
            
            # Pharmacologically important groups
            'guanidine': '[NX3]([#1,#6])[CX3](=[NX3+,NX2+0])[NX3]([#1,#6])',
            'urea': '[NX3][CX3](=[OX1])[NX3]',
            'hydrazone': '[NX3]=[CX3]',
            'oxime': '[NX2]=[CX3][OX2H]',
            'semicarbazone': '[NX3][CX3](=[NX2])[NX3]',
            
            # Carbohydrate-related
            'acetal': '[OX2]([#6])[CX4]([OX2])',
            'ketal': '[OX2]([#6])[CX4]([OX2])([#6])',
            
            # Other important groups
            'anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
            'acid_chloride': '[CX3](=[OX1])[ClX1]',
            'imine': '[CX3]=[NX2]',
            'enamine': '[NX3][CX3]=[CX3]',
            'alkyne': '[CX2]#[CX2]',
            'alkene': '[CX3]=[CX3]',
            'epoxide': '[OX2r3]1[#6r3][#6r3]1',
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for name, pattern in self.functional_group_patterns.items():
            try:
                self.compiled_patterns[name] = Chem.MolFromSmarts(pattern)
            except Exception as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")
        
        # Ring system analysis parameters
        self.min_ring_size = self.config.get('min_ring_size', 3)
        self.max_ring_size = self.config.get('max_ring_size', 12)
        
        # Chain analysis parameters
        self.min_chain_length = self.config.get('min_chain_length', 3)
        self.max_chain_length = self.config.get('max_chain_length', 20)
    
    def analyze_molecule(self, mol: Chem.Mol, smiles: str = None) -> Dict[str, Any]:
        """
        Comprehensive molecular semantic analysis.
        
        Args:
            mol: RDKit molecule object
            smiles: SMILES string (optional, for fallback)
            
        Returns:
            Dictionary containing all semantic analysis results
        """
        if mol is None:
            logger.warning("Invalid molecule provided for semantic analysis")
            return self._empty_analysis()
        
        try:
            # Basic molecule properties
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            # Initialize empty results for robustness
            functional_groups = []
            ring_systems = []
            chain_systems = []
            aromatic_systems = []
            complex_fragments = []
            pharmacophore_features = []
            
            # Try each analysis step independently; isolate failures
            try:
                functional_groups = self._identify_functional_groups(mol)
            except Exception as e:
                logger.warning(f"Functional group analysis failed: {e}")
            
            try:
                ring_systems = self._analyze_ring_systems(mol)
            except Exception as e:
                logger.warning(f"Ring system analysis failed: {e}")
            
            try:
                chain_systems = self._analyze_chain_systems(mol)
            except Exception as e:
                logger.warning(f"Chain analysis failed: {e}")
            
            try:
                aromatic_systems = self._analyze_aromatic_systems(mol)
            except Exception as e:
                logger.warning(f"Aromatic system analysis failed: {e}")
            
            try:
                complex_fragments = self._identify_complex_fragments(mol)
            except Exception as e:
                logger.warning(f"Complex fragment analysis failed: {e}")
            
            try:
                pharmacophore_features = self._extract_pharmacophore_features(mol)
            except Exception as e:
                logger.warning(f"Pharmacophore feature extraction failed: {e}")

            # Assemble comprehensive analysis
            analysis = {
                'basic_properties': {
                    'num_atoms': num_atoms,
                    'num_bonds': num_bonds,
                    'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol)
                },
                'functional_group': functional_groups,
                'ring_systems': ring_systems,
                'chain_systems': chain_systems,
                'aromatic_systems': aromatic_systems,
                'complex_fragments': complex_fragments,
                'pharmacophore_features': pharmacophore_features,
                
                # Derived semantic blocks for masking
                'semantic_blocks': self._create_semantic_blocks(
                    functional_groups, ring_systems, chain_systems, 
                    aromatic_systems, complex_fragments
                ),
                
                # Atom-level annotations
                'atom_annotations': self._create_atom_annotations(
                    mol, functional_groups, ring_systems, aromatic_systems
                ),
                
                # Difficulty assessment for curriculum learning
                'complexity_metrics': self._assess_structural_complexity(
                    mol, functional_groups, ring_systems, aromatic_systems
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in molecular semantic analysis: {e}")
            return self._empty_analysis()
    
    def _identify_functional_groups(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Identify functional groups in the molecule."""
        functional_groups = []
        
        for fg_name, pattern in self.compiled_patterns.items():
            if pattern is None:
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                functional_groups.append({
                    'type': fg_name,
                    'atoms': list(match),
                    'size': len(match),
                    'pattern': self.functional_group_patterns[fg_name]
                })
        
        # Remove overlapping matches (keep larger groups)
        functional_groups = self._remove_overlapping_groups(functional_groups)
        
        return functional_groups
    
    def _analyze_ring_systems(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Analyze ring systems in the molecule."""
        ring_systems = []
        
        # Get ring information
        ring_info = mol.GetRingInfo()
        
        # Individual rings
        for ring in ring_info.AtomRings():
            if self.min_ring_size <= len(ring) <= self.max_ring_size:
                ring_atoms = list(ring)
                ring_bonds = []
                
                # Find bonds in this ring
                for i in range(len(ring_atoms)):
                    for j in range(i + 1, len(ring_atoms)):
                        bond = mol.GetBondBetweenAtoms(ring_atoms[i], ring_atoms[j])
                        if bond is not None:
                            ring_bonds.append(bond.GetIdx())
                
                # Analyze ring properties
                is_aromatic = all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() 
                                for atom_idx in ring_atoms)
                is_hetero = any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 
                              for atom_idx in ring_atoms)
                
                # Safe ring_type classification
                try:
                    ring_type = self._classify_ring_type(mol, ring_atoms)
                except Exception as e:
                    logger.warning(f"Failed to classify ring type for ring {ring_atoms}: {e}")
                    ring_type = f"unknown_{len(ring_atoms)}"
                
                ring_systems.append({
                    'type': 'individual_ring',
                    'atoms': ring_atoms,
                    'bonds': ring_bonds,
                    'size': len(ring_atoms),
                    'is_aromatic': is_aromatic,
                    'is_heterocycle': is_hetero,
                    'ring_type': ring_type
                })
        
        # Fused ring systems
        fused_systems = self._identify_fused_ring_systems(mol, ring_info)
        ring_systems.extend(fused_systems)
        
        return ring_systems
    
    def _analyze_chain_systems(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Analyze chain systems (linear and branched)."""
        chain_systems = []
        
        # Find longest chains
        for atom in mol.GetAtoms():
            if atom.GetDegree() <= 2:  # Potential chain end
                chains = self._find_chains_from_atom(mol, atom.GetIdx())
                for chain in chains:
                    if len(chain) >= self.min_chain_length:
                        chain_systems.append({
                            'type': 'linear_chain',
                            'atoms': chain,
                            'length': len(chain),
                            'is_saturated': self._is_chain_saturated(mol, chain),
                            'has_branching': self._has_branching(mol, chain)
                        })
        
        # Remove redundant chains
        chain_systems = self._remove_redundant_chains(chain_systems)
        
        return chain_systems
    
    def _analyze_aromatic_systems(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Analyze aromatic systems."""
        aromatic_systems = []
        
        # Find aromatic atoms
        aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
        
        if not aromatic_atoms:
            return aromatic_systems
        
        # Group connected aromatic atoms
        aromatic_groups = self._group_connected_atoms(mol, aromatic_atoms)
        
        for group in aromatic_groups:
            if len(group) >= 6:  # Minimum aromatic system size
                # Safe aromatic system classification
                try:
                    system_type = self._classify_aromatic_system(mol, group)
                except Exception as e:
                    logger.warning(f"Failed to classify aromatic system: {e}")
                    system_type = f"aromatic_{len(group)}"
                
                aromatic_systems.append({
                    'type': 'aromatic_system',
                    'atoms': group,
                    'size': len(group),
                    'num_rings': self._count_rings_in_atoms(mol, group),
                    'system_type': system_type
                })
        
        return aromatic_systems
    
    def _identify_complex_fragments(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Identify complex molecular fragments."""
        complex_fragments = []
        
        # Multi-ring systems with heteroatoms
        ring_info = mol.GetRingInfo()
        if len(ring_info.AtomRings()) >= 2:
            # Find bridging atoms between rings
            bridging_atoms = self._find_bridging_atoms(mol, ring_info)
            if bridging_atoms:
                complex_fragments.append({
                    'type': 'bridged_system',
                    'atoms': bridging_atoms,
                    'complexity': 'high'
                })
        
        # Spiro centers
        spiro_centers = self._find_spiro_centers(mol)
        for spiro in spiro_centers:
            complex_fragments.append({
                'type': 'spiro_center',
                'atoms': spiro,
                'complexity': 'high'
            })
        
        # Highly substituted carbons
        highly_substituted = self._find_highly_substituted_atoms(mol)
        for atom_group in highly_substituted:
            complex_fragments.append({
                'type': 'highly_substituted',
                'atoms': atom_group,
                'complexity': 'medium'
            })
        
        return complex_fragments
    
    def _extract_pharmacophore_features(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """Extract pharmacophore-like features."""
        features = []
        
        # Hydrogen bond donors
        donors = self._find_hb_donors(mol)
        for donor in donors:
            features.append({
                'type': 'hb_donor',
                'atoms': donor,
                'pharmacophore_type': 'donor'
            })
        
        # Hydrogen bond acceptors
        acceptors = self._find_hb_acceptors(mol)
        for acceptor in acceptors:
            features.append({
                'type': 'hb_acceptor',
                'atoms': acceptor,
                'pharmacophore_type': 'acceptor'
            })
        
        # Hydrophobic regions
        hydrophobic = self._find_hydrophobic_regions(mol)
        for region in hydrophobic:
            features.append({
                'type': 'hydrophobic',
                'atoms': region,
                'pharmacophore_type': 'hydrophobic'
            })
        
        return features
    
    def _create_semantic_blocks(self, functional_groups: List[Dict], 
                              ring_systems: List[Dict], chain_systems: List[Dict],
                              aromatic_systems: List[Dict], 
                              complex_fragments: List[Dict]) -> Dict[str, List[List[int]]]:
        """Create semantic blocks for masking strategies."""
        blocks = {
            'functional_group': [],
            'ring': [],
            'chain': [],
            'aromatic': [],
            'complex': []
        }
        
        # Functional group blocks
        for fg in functional_groups:
            blocks['functional_group'].append(fg['atoms'])
        
        # Ring blocks
        for ring in ring_systems:
            blocks['ring'].append(ring['atoms'])
        
        # Chain blocks
        for chain in chain_systems:
            blocks['chain'].append(chain['atoms'])
        
        # Aromatic blocks
        for arom in aromatic_systems:
            blocks['aromatic'].append(arom['atoms'])
        
        # Complex blocks
        for complex_frag in complex_fragments:
            blocks['complex'].append(complex_frag['atoms'])
        
        return blocks
    
    def _create_atom_annotations(self, mol: Chem.Mol, functional_groups: List[Dict],
                                ring_systems: List[Dict], 
                                aromatic_systems: List[Dict]) -> Dict[int, List[str]]:
        """Create atom-level semantic annotations."""
        annotations = defaultdict(list)
        
        # Annotate functional group atoms
        for fg in functional_groups:
            for atom_idx in fg['atoms']:
                annotations[atom_idx].append(f"fg_{fg['type']}")
        
        # Annotate ring atoms - use safe access to avoid KeyError
        for ring in ring_systems:
            ring_type = ring.get('ring_type', 'unknown')
            for atom_idx in ring['atoms']:
                annotations[atom_idx].append(f"ring_{ring_type}")
        
        # Annotate aromatic atoms - use safe access to avoid KeyError
        for arom in aromatic_systems:
            system_type = arom.get('system_type', 'unknown')
            for atom_idx in arom['atoms']:
                annotations[atom_idx].append(f"aromatic_{system_type}")
        
        return dict(annotations)
    
    def _assess_structural_complexity(self, mol: Chem.Mol, functional_groups: List[Dict],
                                    ring_systems: List[Dict], 
                                    aromatic_systems: List[Dict]) -> Dict[str, float]:
        """Assess structural complexity for curriculum learning."""
        metrics = {}
        
        # Basic complexity
        num_atoms = mol.GetNumAtoms()
        metrics['size_complexity'] = min(num_atoms / 50.0, 1.0)  # Normalize by typical drug size
        
        # Functional group complexity
        num_fg = len(functional_groups)
        unique_fg_types = len(set(fg['type'] for fg in functional_groups))
        metrics['fg_complexity'] = min((num_fg + unique_fg_types) / 10.0, 1.0)
        
        # Ring complexity
        num_rings = len(ring_systems)
        fused_rings = sum(1 for ring in ring_systems if ring.get('type') == 'fused_system')
        metrics['ring_complexity'] = min((num_rings + fused_rings * 2) / 8.0, 1.0)
        
        # Aromatic complexity
        total_aromatic_atoms = sum(len(arom['atoms']) for arom in aromatic_systems)
        metrics['aromatic_complexity'] = min(total_aromatic_atoms / 20.0, 1.0)
        
        # Overall complexity (weighted average)
        metrics['overall_complexity'] = (
            0.3 * metrics['size_complexity'] +
            0.3 * metrics['fg_complexity'] +
            0.25 * metrics['ring_complexity'] +
            0.15 * metrics['aromatic_complexity']
        )
        
        return metrics
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'basic_properties': {},
            'functional_group': [],
            'ring_systems': [],
            'chain_systems': [],
            'aromatic_systems': [],
            'complex_fragments': [],
            'pharmacophore_features': [],
            'semantic_blocks': {
                'functional_group': [],
                'rings': [],
                'chains': [],
                'aromatic': [],
                'complex': []
            },
            'atom_annotations': {},
            'complexity_metrics': {'overall_complexity': 0.0}
        }
    
    # Helper methods (simplified implementations)
    def _remove_overlapping_groups(self, groups: List[Dict]) -> List[Dict]:
        """Remove overlapping functional groups, keeping larger ones."""
        if not groups:
            return groups
        
        # Sort by size (larger first)
        sorted_groups = sorted(groups, key=lambda x: x['size'], reverse=True)
        non_overlapping = []
        used_atoms = set()
        
        for group in sorted_groups:
            group_atoms = set(group['atoms'])
            if not group_atoms & used_atoms:  # No overlap
                non_overlapping.append(group)
                used_atoms.update(group_atoms)
        
        return non_overlapping
    
    def _classify_ring_type(self, mol: Chem.Mol, ring_atoms: List[int]) -> str:
        """Classify ring type with enhanced boundary checking."""
        if not ring_atoms or not mol:
            return 'unknown_0'
        
        size = len(ring_atoms)
        num_atoms = mol.GetNumAtoms()
        
        # Boundary check: ensure all atom indices valid
        if not all(0 <= atom_idx < num_atoms for atom_idx in ring_atoms):
            logger.warning(f"Invalid atom indices in ring: {ring_atoms}, mol has {num_atoms} atoms")
            return f'invalid_{size}'
        
        try:
            is_aromatic = all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms)
            has_hetero = any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in ring_atoms)
        except Exception as e:
            logger.warning(f"Error accessing atom properties for ring {ring_atoms}: {e}")
            return f'error_{size}'
        
        if is_aromatic:
            if size == 6 and not has_hetero:
                return 'benzene'
            elif size == 6 and has_hetero:
                return 'heteroaromatic_6'
            elif size == 5:
                return 'heteroaromatic_5'
            else:
                return 'aromatic_other'
        else:
            if has_hetero:
                return f'heteroaliphatic_{size}'
            else:
                return f'aliphatic_{size}'
    
    def _find_chains_from_atom(self, mol: Chem.Mol, start_atom: int) -> List[List[int]]:
        """Find chains starting from a specific atom."""
        chains = []
        visited = set()
        
        def dfs_chain(current_atom, current_chain):
            if current_atom in visited:
                return
            
            visited.add(current_atom)
            current_chain.append(current_atom)
            
            atom = mol.GetAtomWithIdx(current_atom)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            
            # Continue chain if exactly 2 neighbors (and not in ring)
            if len(neighbors) == 2 and not atom.IsInRing():
                for neighbor in neighbors:
                    if neighbor not in current_chain:
                        dfs_chain(neighbor, current_chain[:])
            elif len(current_chain) >= self.min_chain_length:
                chains.append(current_chain[:])
        
        dfs_chain(start_atom, [])
        return chains
    
    def _is_chain_saturated(self, mol: Chem.Mol, chain: List[int]) -> bool:
        """Check if chain is saturated."""
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                return False
        return True
    
    def _has_branching(self, mol: Chem.Mol, chain: List[int]) -> bool:
        """Check if chain has branching."""
        for atom_idx in chain:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetDegree() > 2:
                return True
        return False
    
    def _remove_redundant_chains(self, chains: List[Dict]) -> List[Dict]:
        """Remove redundant chain definitions."""
        # Simple implementation: remove chains that are subsets of others
        unique_chains = []
        for i, chain1 in enumerate(chains):
            is_subset = False
            atoms1 = set(chain1['atoms'])
            
            for j, chain2 in enumerate(chains):
                if i != j:
                    atoms2 = set(chain2['atoms'])
                    if atoms1.issubset(atoms2) and len(atoms1) < len(atoms2):
                        is_subset = True
                        break
            
            if not is_subset:
                unique_chains.append(chain1)
        
        return unique_chains
    
    def _group_connected_atoms(self, mol: Chem.Mol, atom_list: List[int]) -> List[List[int]]:
        """Group connected atoms into clusters."""
        if not atom_list:
            return []
        
        visited = set()
        groups = []
        
        def dfs_group(atom_idx, current_group):
            if atom_idx in visited or atom_idx not in atom_list:
                return
            
            visited.add(atom_idx)
            current_group.append(atom_idx)
            
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in atom_list and neighbor_idx not in visited:
                    dfs_group(neighbor_idx, current_group)
        
        for atom_idx in atom_list:
            if atom_idx not in visited:
                group = []
                dfs_group(atom_idx, group)
                if group:
                    groups.append(group)
        
        return groups
    
    def _count_rings_in_atoms(self, mol: Chem.Mol, atoms: List[int]) -> int:
        """Count number of rings involving the given atoms."""
        ring_info = mol.GetRingInfo()
        count = 0
        
        atom_set = set(atoms)
        for ring in ring_info.AtomRings():
            ring_set = set(ring)
            if ring_set.issubset(atom_set):
                count += 1
        
        return count
    
    def _classify_aromatic_system(self, mol: Chem.Mol, atoms: List[int]) -> str:
        """Classify type of aromatic system."""
        num_rings = self._count_rings_in_atoms(mol, atoms)
        has_hetero = any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in atoms)
        
        if num_rings == 1:
            if has_hetero:
                return 'heteroaromatic_single'
            else:
                return 'benzene'
        elif num_rings == 2:
            return 'bicyclic_aromatic'
        elif num_rings >= 3:
            return 'polycyclic_aromatic'
        else:
            return 'aromatic_unknown'
    
    def _identify_fused_ring_systems(self, mol: Chem.Mol, ring_info) -> List[Dict]:
        """Identify fused ring systems."""
        fused_systems = []
        atom_rings = ring_info.AtomRings()
        
        if len(atom_rings) < 2:
            return fused_systems
        
        # Find overlapping rings
        for i, ring1 in enumerate(atom_rings):
            for j, ring2 in enumerate(atom_rings[i+1:], i+1):
                overlap = set(ring1) & set(ring2)
                if len(overlap) >= 2:  # Rings share at least 2 atoms (fused)
                    combined_atoms = list(set(ring1) | set(ring2))
                    
                    # Add safe ring_type for fused systems
                    try:
                        ring_type = self._classify_ring_type(mol, combined_atoms)
                    except Exception as e:
                        logger.warning(f"Failed to classify fused ring type: {e}")
                        ring_type = f"fused_{len(combined_atoms)}"
                    
                    fused_systems.append({
                        'type': 'fused_system',
                        'atoms': combined_atoms,
                        'ring_count': 2,
                        'fusion_atoms': list(overlap),
                        'ring_type': ring_type
                    })
        
        return fused_systems
    
    def _find_bridging_atoms(self, mol: Chem.Mol, ring_info) -> List[int]:
        """Find atoms that bridge between rings."""
        # Simplified implementation
        bridging_atoms = []
        ring_atoms = set()
        
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)
        
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            ring_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms)
            if ring_neighbors >= 3:  # Potential bridging atom
                bridging_atoms.append(atom_idx)
        
        return bridging_atoms
    
    def _find_spiro_centers(self, mol: Chem.Mol) -> List[List[int]]:
        """Find spiro centers."""
        # Simplified spiro center detection
        spiro_centers = []
        ring_info = mol.GetRingInfo()
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            # Count how many different rings this atom belongs to
            ring_count = sum(1 for ring in ring_info.AtomRings() if atom_idx in ring)
            if ring_count >= 2 and atom.GetDegree() == 4:
                # Potential spiro center
                spiro_centers.append([atom_idx])
        
        return spiro_centers
    
    def _find_highly_substituted_atoms(self, mol: Chem.Mol) -> List[List[int]]:
        """Find highly substituted atoms."""
        highly_substituted = []
        
        for atom in mol.GetAtoms():
            if atom.GetDegree() >= 4 and atom.GetAtomicNum() == 6:  # Carbon with 4+ bonds
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                highly_substituted.append([atom.GetIdx()] + neighbors)
        
        return highly_substituted
    
    def _find_hb_donors(self, mol: Chem.Mol) -> List[List[int]]:
        """Find hydrogen bond donors."""
        donors = []
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8]:  # N or O with H
                donors.append([atom.GetIdx()])
        return donors
    
    def _find_hb_acceptors(self, mol: Chem.Mol) -> List[List[int]]:
        """Find hydrogen bond acceptors."""
        acceptors = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [7, 8] and len(atom.GetNeighbors()) < 3:  # N or O with lone pairs
                acceptors.append([atom.GetIdx()])
        return acceptors
    
    def _find_hydrophobic_regions(self, mol: Chem.Mol) -> List[List[int]]:
        """Find hydrophobic regions."""
        hydrophobic_regions = []
        
        # Find aromatic rings as hydrophobic regions
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) == 6:  # 6-membered rings
                atoms_in_ring = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
                if all(atom.GetIsAromatic() for atom in atoms_in_ring):
                    hydrophobic_regions.append(list(ring))
        
        return hydrophobic_regions


def analyze_molecule_semantics(mol: Chem.Mol, smiles: str = None, 
                             config: Dict = None) -> Dict[str, Any]:
    """
    Convenience function for molecular semantic analysis.
    
    Args:
        mol: RDKit molecule object
        smiles: SMILES string (optional)
        config: Analysis configuration
        
    Returns:
        Comprehensive semantic analysis results
    """
    analyzer = MolecularSemanticAnalyzer(config)
    return analyzer.analyze_molecule(mol, smiles)
