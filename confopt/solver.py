"""Methods for solving the conformer option problem"""
import logging
import warnings
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

from modAL.acquisition import max_EI
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.optimize import minimize
from modAL.models import BayesianOptimizer
from ase.calculators.calculator import Calculator
from ase.io.xyz import simple_write_xyz
from ase import Atoms
import numpy as np

from confopt.assess import evaluate_energy, relax_structure
from confopt.setup import DihedralInfo


logger = logging.getLogger(__name__)


def _elementwise_expsine_kernel(x, y, gamma=10, p=360):
    """Compute the expoonential sine kernel

    Args:
        x, y: Coordinates to be compared
        gamma: Length scale of the kernel
        p: Periodicity of the kernel
    Returns:
        Kernel metric
    """

    # Compute the distances between the two points
    dists = np.subtract(x, y)

    # Compute the sine with a periodicity of p
    sine_dists = np.sin(np.pi * dists / p)

    # Return exponential of the squared kernel
    return np.sum(np.exp(-2 * np.power(sine_dists, 2) / gamma ** 2), axis=-1)


def _get_search_space(optimizer: BayesianOptimizer, n_dihedrals: int, n_samples: int = 32, width: float = 60):
    """Generate many samples by adding zero-mean Gaussian noise to the current minimum

    Args:
        optimizer: Optimizer being used to perform Bayesian optimization
        n_dihedrals: Number of dihedrals to optimize
        n_samples: Number of initial starts of the optimizer to use.
            Will return all points sampled by the optimizer
        width: Width of the 
    Returns:
        List of points to be considered
    """

    # Generate random starting points
    init_points = np.random.normal(0, width, size=(n_samples, n_dihedrals))
    best_point = np.array(optimizer.X_max)
    init_points += best_point[None, :]
    
    # Use local optimization to find the minima near these
    #  See: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html#optimize-minimize-powell
    final_points = []
    for init_point in init_points:
        result = minimize(
            # Define the function to be optimized
            lambda x: -optimizer.predict([x]),  # Model predicts the negative energy and requires a 2D array,
            init_point,  # Initial guess
            method='nelder-mead',  # A derivative-free optimizer
        )
        final_points.append(result.x)

    # Combine the results from the optimizer with the initial points sampled
    all_points = np.vstack([
        init_points,
        final_points
    ])
    return all_points


def run_optimization(atoms: Atoms, dihedrals: List[DihedralInfo], n_steps: int, calc: Calculator,
                     init_steps: int, out_dir: Optional[Path], relax: bool = True) -> Atoms:
    """Optimize the structure of a molecule by iteratively changing the dihedral angles

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evalaute
        calc: Calculator to pick the energy
        out_dir: Output path for logging information
        relax: Whether to relax non-dihedral degrees of freedom each step
    Returns:
        (Atoms) optimized geometry
    """
    # Perform an initial relaxation
    _, init_atoms = relax_structure(atoms, calc)
    if out_dir is not None:
        with open(out_dir.joinpath('relaxed.xyz'), 'w') as fp:
            simple_write_xyz(fp, [init_atoms])

    # Evaluate initial point
    start_coords = np.array([d.get_angle(init_atoms) for d in dihedrals])
    start_energy, start_atoms = evaluate_energy(start_coords, atoms, dihedrals, calc, relax)
    logger.info(f'Computed initial energy: {start_energy}')

    # Begin a structure log, if output available
    if out_dir is not None:
        log_path = out_dir.joinpath('structures.csv')
        with log_path.open('w') as fp:
            writer = DictWriter(fp, ['time', 'xyz', 'energy', 'ediff'])
            writer.writeheader()

        def add_entry(coords, atoms, energy):
            with log_path.open('a') as fp:
                writer = DictWriter(fp, ['time', 'coords', 'xyz', 'energy', 'ediff'])
                xyz = StringIO()
                simple_write_xyz(xyz, [atoms])
                writer.writerow({
                    'time': datetime.now().timestamp(),
                    'coords': coords.tolist(),
                    'xyz': xyz.getvalue(),
                    'energy': energy,
                    'ediff': energy - start_energy
                })
        add_entry(start_coords, start_atoms, start_energy)

    # Make some initial guesses
    init_guesses = np.random.normal(start_coords, 30, size=(init_steps, len(dihedrals)))
    init_energies = []
    for i, guess in enumerate(init_guesses):
        energy, cur_atoms = evaluate_energy(guess, start_atoms, dihedrals, calc, relax)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            add_entry(guess, cur_atoms, energy)

    # Prepare an a machine learning model
    gpr = GaussianProcessRegressor(
        kernel=kernels.PairwiseKernel(metric=_elementwise_expsine_kernel),
        n_restarts_optimizer=8
    )

    # Build an optimizer
    optimizer = BayesianOptimizer(
        estimator=gpr,
        X_training=np.concatenate([start_coords[None, :], init_guesses], axis=0),
        y_training=np.multiply(-1, [0] + init_energies),
        query_strategy=max_EI,
    )

    # Loop over many steps
    cur_atoms = start_atoms.copy()
    for step in range(n_steps):
        # Make a new search space
        sample_points = _get_search_space(optimizer, len(dihedrals))
        logger.debug(f'Generated {len(sample_points)} new points to evaluate')

        # Pick the best point to add to the dataset
        best_point, best_coords = optimizer.query(sample_points)
        best_coords = best_coords[0, :]

        # Compute the energies of those points
        energy, cur_atoms = evaluate_energy(best_coords, cur_atoms, dihedrals, calc, relax)
        logger.info(f'Evaluated energy in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')
        if energy - start_energy < -optimizer.y_max and out_dir is not None:
            with open(out_dir.joinpath('current_best.xyz'), 'w') as fp:
                simple_write_xyz(fp, [cur_atoms])

        # Update the log
        if out_dir is not None:
            add_entry(start_coords, cur_atoms, energy)

        # Update the model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            optimizer.teach([best_coords], [start_energy - energy])  # Remember: We are maximizing -energy

    # Final relaxations
    best_atoms = cur_atoms.copy()
    best_energy, best_atoms = evaluate_energy(optimizer.X_max, best_atoms, dihedrals, calc)
    logger.info('Performed final relaxation with dihedral constraints.'
                f'E: {best_energy}. E-E0: {best_energy - start_energy}')
    if out_dir is not None:
        add_entry(np.array(optimizer.X_max), best_atoms, best_energy)

    # Relaxations
    best_atoms.set_constraint()
    best_energy, best_atoms = relax_structure(best_atoms, calc)
    logger.info('Performed final relaxation without dihedral constraints.'
                f' E: {best_energy}. E-E0: {best_energy - start_energy}')
    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if out_dir is not None:
        add_entry(best_coords, best_atoms, best_energy)
    return best_atoms
