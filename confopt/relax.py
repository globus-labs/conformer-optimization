"""Tools for computing the energy of a molecule"""

from typing import List, Tuple
import os

from ase.calculators.psi4 import Psi4
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase import Atoms

from confopt import DihedralInfo

calc = torchani.models.ANI2x().ase()


def relax_structure(atoms: Atoms) -> float:
    """Relax and return the energy of the ground state
    
    Args:
        atoms
    """
    
    atoms.set_calculator(calc)
    
    dyn = BFGS(atoms, logfile=os.devnull)
    dyn.run(fmax=1e-3)
    
    return atoms.get_potential_energy()


    
def set_dihedrals_and_relax(atoms: Atoms, dihedrals: List[Tuple[float, DihedralInfo]]) -> float:
    """Set the dihedral angles and compute the energy of the system
    
    Args:
        atoms: Molecule to ajdust
        dihedrals: List of dihedrals to set to a certain angle
    Returns:
        Energy of the system
    """
    
    # Copy input so that we can loop over it twice (i.e., avoiding problems around zip being a generator)
    dihedrals = list(dihedrals)
    
    # Set the dihedral angles to desired settings
    for di in dihedrals:
        atoms.set_dihedral(*di[1].chain, di[0], indices=di[1].group)
        
    # Define the constraints
    dih_cnsts = [(di[0], di[1].chain) for di in dihedrals]
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dih_cnsts))
    
    return relax_structure(atoms)
