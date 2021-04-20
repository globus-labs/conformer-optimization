"""Tools for assessing the bond structure of a molecule and finding the dihedrals to move"""

from typing import Tuple, Set, Dict, List
from dataclasses import dataclass
from io import StringIO
import logging

from ase.io.xyz import read_xyz
from ase import Atoms
from openbabel import OBMolBondIter
import networkx as nx
import numpy as np
import pybel

logger = logging.getLogger(__name__)


def get_initial_structure(smiles: str) -> Tuple[Atoms, pybel.Molecule]:
    """Generate an initial guess for a molecular structure
    
    Args:
        smiles: SMILES string
    Returns: 
        Generate an Atoms object
    """

    # Make the 3D structure
    mol = pybel.readstring("smi", smiles)
    mol.make3D()

    # Convert it to ASE
    atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
    atoms.charge = mol.charge
    atoms.set_initial_charges([a.formalcharge for a in mol.atoms])
        
    return atoms, mol


@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule"""

    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    type: str = None

    def get_angle(self, atoms: Atoms) -> float:
        """Get the value of the specified dihedral angle

        Args:
            atoms: Structure to assess
        """
        return atoms.get_dihedral(*self.chain)


def detect_dihedrals(mol: pybel.Molecule) -> List[DihedralInfo]:
    """Detect the bonds to be treated as rotors.
    
    We use the more generous definition from RDKit: 
    https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E
    
    It matches pairs of atoms that are connected by a single bond,
    both bonds have at least one other bond that is not a triple bond
    and they are not part of the same ring.
    
    Args:
        mol: Molecule to assess
    Returns:
        List of dihedral angles. Most are defined 
    """
    dihedrals = []

    # Compute the bonding graph
    g = get_bonding_graph(mol)

    # Get the indices of backbond atoms
    backbone = set(i for i, d in g.nodes(data=True) if d['z'] > 1)

    # Step 1: Get the bonds from a simple matching
    smarts = pybel.Smarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    for i, j in smarts.findall(mol):
        dihedrals.append(get_dihedral_info(g, (i - 1, j - 1), backbone))
    return dihedrals


def get_bonding_graph(mol: pybel.Molecule) -> nx.Graph:
    """Generate a bonding graph from a molecule
    
    Args:
        mol: Molecule to be assessed
    Returns: 
        Graph describing the connectivity
    """

    # Get the bonding graph
    g = nx.Graph()
    g.add_nodes_from([
        (i, dict(z=a.atomicnum))
        for i, a in enumerate(mol.atoms)
    ])
    for bond in OBMolBondIter(mol.OBMol):
        g.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1,
                   data={"rotor": bond.IsRotor(), "ring": bond.IsInRing()})
    return g


def get_dihedral_info(graph: nx.Graph, bond: Tuple[int, int], backbone_atoms: Set[int]) -> DihedralInfo:
    """For a rotatable bond in a model, get the atoms which define the dihedral angle
    and the atoms that should rotate along with the right half of the molecule
    
    Args:
        graph: Bond graph of the molecule
        bond: Left and right indicies of the bond, respectively
        backbone_atoms: List of atoms defined as part of the backbone
    Returns:
        - Atom indices defining the dihedral. Last atom is the one that will be moved 
          by ase's "set_dihedral" function
        - List of atoms being rotated along with set_dihedral
    """

    # Pick the atoms to use in the dihedral, starting with the left
    points = list(bond)
    choices = set(graph[bond[0]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.insert(0, min(choices))

    # Then the right
    choices = set(graph[bond[1]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.append(min(choices))

    # Get the points that will rotate along with the bond
    h = graph.copy()
    h.remove_edge(*bond)
    a, b = nx.connected_components(h)
    if bond[1] in a:
        return DihedralInfo(chain=points, group=a, type='backbone')
    else:
        return DihedralInfo(chain=points, group=b, type='backbone')


def fix_cyclopropenyl(atoms: Atoms, mol: pybel.Molecule) -> Atoms:
    """Detect cyclopropenyl groups and assure they are planar.
    
    Args:
        atoms: Object holding the 3D positions
        mol: Object holidng the bonding information
    Returns:
        Version of atoms with the rings flattened
    """
    
    # Find cyclopropenyl groups
    smarts = pybel.Smarts('C1=C[C+]1')
    rings = smarts.findall(mol)
    if len(rings) == 0:
        return atoms  # no changes
    
    # For each ring, flatten it
    atoms = atoms.copy()
    g = get_bonding_graph(mol)
    for ring in rings:
        ring = tuple(x - 1 for x in rings[0])  # Pybel is 1-indexed
        
        # Get the normal of the ring
        normal = np.cross(*np.subtract(atoms.positions[ring[:2], :], atoms.positions[ring[2], :]))
        normal /= np.linalg.norm(normal)
        
        # Adjust the groups attached to each member of the ring
        for ring_atom in ring:
            # Get the ID of the group bonded to it
            bonded_atom = next(r for r in g[ring_atom] if r not in ring)
            
            # Determine the atoms that are part of that functional group
            h = g.copy()
            h.remove_edge(ring_atom, bonded_atom)
            a, b = nx.connected_components(h)
            mask = np.zeros((len(atoms),), dtype=bool)
            if bonded_atom in a:
                mask[list(a)] = True
            else:
                mask[list(b)] = True
            
            # Get the rotation angle
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            rot_angle = np.arccos(angle) - np.pi / 2
            logger.debug(f'Rotating by {rot_angle} radians')
            
            # Perform the rotation
            rotation_axis = np.cross(bond_vector, normal)
            atoms._masked_rotate(atoms.positions[ring_atom], rotation_axis, rot_angle, mask)
            
            # make the atom at a 150 angle with the the ring too
            another_ring = next(r for r in ring if r != ring_atom)
            atoms.set_angle(another_ring, ring_atom, bonded_atom, 150, mask=mask)
            assert np.isclose(atoms.get_angle(another_ring, ring_atom, bonded_atom), 150).all()
            
            # Make sure it worked
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            final_angle = np.arccos(angle)
            assert np.isclose(final_angle, np.pi / 2).all()
            
            
                              
        logger.info(f'Detected {len(rings)} cyclopropenyl rings. Ensured they are planar.')
        return atoms
    