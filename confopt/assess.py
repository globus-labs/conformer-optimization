"""Tools for computing the energy of a molecule"""
from typing import List, Tuple, Union
import os

from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase import Atoms
import numpy as np

from confopt.setup import DihedralInfo


def evaluate_energy(angles: Union[List[float], np.ndarray], atoms: Atoms,
                    dihedrals: List[DihedralInfo], calc: Calculator,
                    relax: bool = True) -> Tuple[float, Atoms]:
    """Compute the energy of a cysteine molecule given dihedral angles

    Args:
        angles: List of dihedral angles
        atoms: Structure to optimize
        dihedrals: Description of the dihedral angles
        calc: Calculator used to compute energy/gradients
        relax: Whether to relax the non-dihedral degrees of freedom
    Returns:
        - (float) energy of the structure
        - (Atoms) Relaxed structure
    """
    # Make a copy of the input
    atoms = atoms.copy()

    # Set the dihedral angles to desired settings
    dih_cnsts = []
    for a, di in zip(angles, dihedrals):
        atoms.set_dihedral(*di.chain, a, indices=di.group)

        # Define the constraints
        dih_cnsts.append((a, di.chain))
        
    # If not relaxed, just compute the energy
    if not relax:
        return calc.get_potential_energy(atoms), atoms
        
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dih_cnsts))

    return relax_structure(atoms, calc)


def relax_structure(atoms: Atoms, calc: Calculator) -> Tuple[float, Atoms]:
    """Relax and return the energy of the ground state
    
    Args:
        atoms: Atoms object to be optimized
        calc: Calculator used to compute energy/gradients
    Returns:
        Energy of the minimized structure
    """

    atoms.set_calculator(calc)

    dyn = BFGS(atoms, logfile=os.devnull)
    dyn.run(fmax=1e-3)

    return atoms.get_potential_energy(), atoms
