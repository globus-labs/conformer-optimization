"""Tools for assessing the bond structure of a molecule"""

from typing import Tuple, Set, Dict, List
from dataclasses import dataclass
from io import StringIO

from ase.io.xyz import read_xyz
from ase import Atoms
from openbabel import OBMolBondIter
import networkx as nx
import pybel


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
    return atoms, mol

    
@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule"""
    
    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    
    
def detect_dihedrals(mol: pybel.Molecule, ring_groups: bool = False) -> List[DihedralInfo]:
    """Detect the bonds to be treated as rotors.
    
    We use the more generous definition from RDKit: 
    https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E
    
    It matches pairs of atoms that are connected by a single bond,
    both bonds have at least one other bond that is not a triple bond
    and they are not part of the same ring.
    
    You can optionally add in dihedral angles that control whether
    groups attached to rings are able to move bonded groups off
    the plane of the ring.
    
    Args:
        mol: Molecule to assess
        ring_groups: Whether to include dihedrals that move groups off the plane
            of the ring
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
    matches = []
    for i, j in smarts.findall(mol):
        dihedrals.append(get_dihedral_info(g, (i-1, j-1), backbone))
        
    if not ring_groups:
        return dihedrals
    
    # Step 2: Add the groups bonded to the ring
    #  Find the non-hydrogen atoms that are bonded to a ring)
    rings = nx.cycle_basis(g)
    smarts = pybel.Smarts('[!#1]!@[!R0]')
    for attached, ring_a in smarts.findall(mol):
        # Decrement their values (pybel is 1-indexed)
        attached -= 1
        ring_a -= 1
        
        # These two atoms define one end of a chain (attached->ring_a)
        # Step 1: Find a third atom to that is in the same ring as ring_a
        # Get atoms that are bonded to ring_a
        options = set(g[ring_a])
        options.remove(attached)
        
        # Find which of these are in the s
        ring_b = None
        for ring in rings:
            if ring_a in ring:
                ring_b = next(i for i in ring if i in options)
                break
        assert ring_b is not None
        
        # Get an atom bonded to ring_b that is not ring_a
        end = next(i for i in g[ring_b] if i != ring_a)
        
        # The atoms that will move when we rotate this dihedral
        #  are those attached to the group atom
        h = g.copy()
        h.remove_edge(attached, ring_a)
        a, b = nx.connected_components(h)
        chain = [attached, ring_a, ring_b, end]
        if attached in a:
            dihedrals.append(DihedralInfo(chain=chain, group=a))
        else:
            dihedrals.append(DihedralInfo(chain=chain, group=b))
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
        g.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, data={"rotor": bond.IsRotor(), "ring": bond.IsInRing()})
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
        return DihedralInfo(chain=points, group=a)
    else:
        return DihedralInfo(chain=points, group=b)
