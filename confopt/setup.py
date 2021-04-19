"""Tools for assessing the bond structure of a molecule and finding the dihedrals to move"""

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
    atoms.charge = mol.charge
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
