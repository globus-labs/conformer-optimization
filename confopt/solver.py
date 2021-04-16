"""Methods for solving the conformer option problem"""
import csv
import logging
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

from modAL.acquisition import max_EI
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from modAL.models import BayesianOptimizer
from scipy.optimize import minimize
from ase.calculators.calculator import Calculator
from ase.io.xyz import simple_write_xyz
from ase import Atoms
from csv import DictWriter
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


def _get_search_space(optimizer: BayesianOptimizer, n_dihedrals: int, n_samples: int = 32):
    """Generate many samples by attempting to find the minima using a multi-start local optimizer

    Generates new points by adding zero-mean Gaussian noise to the current minimum

    Args:
        optimizer: Optimizer being used to perform Bayesian optimization
        n_dihedrals: Number of dihedrals to optimize
        n_samples: Number of initial starts of the optimizer to use.
            Will return all points sampled by the optimizer
    Returns:
        List of points to be considered
    """

    # Generate random starting points
    init_points = np.random.normal(0, 60, size=(n_samples, n_dihedrals))
    best_point = np.array(optimizer.X_max)
    init_points += best_point[None, :]

    # Use local optimization to find the minima near these
    #  See: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html#optimize-minimize-powell
    points_sampled = []  # Will hold all samples tested by the optimizer
    for init_point in init_points:
        minimize(
            # Define the function to be optimized
            lambda x: -optimizer.predict([x]),  # Model predicts the negative energy and requires a 2D array,
            init_point,  # Initial guess
            method='nelder-mead',  # A derivative-free optimizer
            callback=points_sampled.append  # Stores the points sampled by the optimizer at each step
        )

    # Combine the results from the optimizer with the initial points sampled
    all_points = np.vstack([
        init_points,
        *points_sampled
    ])
    return all_points


def run_optimization(atoms: Atoms, dihedrals: List[DihedralInfo], n_steps: int, calc: Calculator,
                     init_steps: int, out_dir: Optional[Path]) -> Atoms:
    """Optimize the structure of a molecule by iteratively changing the dihedral angles

    Args:
        atoms: Atoms object with the initial geometry
        dihedrals: List of dihedral angles to modify
        n_steps: Number of optimization steps to perform
        init_steps: Number of initial guesses to evalaute
        calc: Calculator to pick the energy
        out_dir: Output path for logging information
    """

    # Evaluate initial point
    initial_point = np.array([d.get_angle(atoms) for d in dihedrals])
    start_energy, start_atoms = evaluate_energy(initial_point, atoms, dihedrals, calc)
    logger.info(f'Computed initial energy: {start_energy}')

    # Begin a structure log, if output available
    if out_dir is not None:
        log_path = out_dir.joinpath('structures.csv')
        with log_path.open('w') as fp:
            writer = DictWriter(fp, ['time', 'xyz', 'energy', 'ediff'])
            writer.writeheader()

        def add_entry(atoms, energy):
            with log_path.open('a') as fp:
                writer = DictWriter(fp, ['time', 'xyz', 'energy', 'ediff'])
                xyz = StringIO()
                simple_write_xyz(xyz, [atoms])
                writer.writerow({
                    'time': datetime.now().timestamp(),
                    'xyz': xyz.getvalue(),
                    'energy': energy,
                    'ediff': energy - start_energy
                })
        add_entry(start_atoms, start_energy)

    # Make some initial guesses
    init_guesses = np.random.normal(initial_point, 30, size=(init_steps, len(dihedrals)))
    init_energies = []
    for i, guess in enumerate(init_guesses):
        energy, cur_atoms = evaluate_energy(guess, start_atoms, dihedrals, calc)
        init_energies.append(energy - start_energy)
        logger.info(f'Evaluated initial guess {i+1}/{init_steps}. Energy-E0: {energy-start_energy}')

        if out_dir is not None:
            add_entry(cur_atoms, energy)

    # Prepare an a machine learning model
    gpr = GaussianProcessRegressor(
        kernel=kernels.PairwiseKernel(metric=_elementwise_expsine_kernel),
        n_restarts_optimizer=4
    )

    # Build an optimizer
    optimizer = BayesianOptimizer(
        estimator=gpr,
        X_training=init_guesses,
        y_training=np.multiply(-1, init_energies),
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
        energy, cur_atoms = evaluate_energy(best_coords, cur_atoms, dihedrals, calc)
        logger.info(f'Evaluated energy in step {step+1}/{n_steps}: Energy-E0: {energy-start_energy}')
        if energy < -optimizer.y_max and out_dir is not None:
            with open(out_dir.joinpath('current_best.xyz'), 'w') as fp:
                simple_write_xyz(fp, [cur_atoms])

        # Update the log
        if out_dir is not None:
            add_entry(cur_atoms, energy)

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
        add_entry(best_atoms, best_energy)

    # Relaxations
    best_atoms.set_constraint()
    best_energy, best_atoms = relax_structure(best_atoms, calc)
    logger.info('Performed final relaxation without dihedral constraints.'
                f'E: {best_energy}. E-E0: {best_energy - start_energy}')
    if out_dir is not None:
        add_entry(best_atoms, best_energy)
    return best_atoms
