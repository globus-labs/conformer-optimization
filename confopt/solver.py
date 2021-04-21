"""Methods for solving the conformer option problem"""
import logging
import warnings
from csv import DictWriter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional

from botorch.optim import optimize_acqf
from modAL.acquisition import max_EI
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.optimize import minimize
from modAL.models import BayesianOptimizer
from ase.calculators.calculator import Calculator
from ase.io.xyz import simple_write_xyz
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import kernels as gpykernels
from gpytorch.priors import NormalPrior
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


def select_next_points_modal(observed_X: List[List[float]], observed_y: List[float]) -> np.ndarray:
    """Generate the next sample to evaluate with XTB

    Uses modAL to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """

    # Make the inputs to arrays
    observed_X = np.array(observed_X)
    observed_y = np.array(observed_y)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Prepare an a machine learning model
        gpr = GaussianProcessRegressor(
            kernel=kernels.PairwiseKernel(metric=_elementwise_expsine_kernel),
            n_restarts_optimizer=8
        )

        # Build an optimizer
        optimizer = BayesianOptimizer(
            estimator=gpr,
            X_training=observed_X,
            y_training=np.multiply(-1, observed_y),
            query_strategy=max_EI,
        )

        sample_points = _get_search_space(optimizer, observed_X.shape[1])
        logger.debug(f'Generated {len(sample_points)} new points to evaluate')

        # Pick the best point to add to the dataset
        best_point, best_coords = optimizer.query(sample_points)
        return best_coords[0, :]


def select_next_points_botorch(observed_X: List[List[float]], observed_y: List[float]) -> np.ndarray:
    """Generate the next sample to evaluate with XTB

    Uses BOTorch to pick the next sample using Expected Improvement

    Args:
        observed_X: Observed coordinates
        observed_y: Observed energies
    Returns:
        Next coordinates to try
    """

    # Convert inputs to torch arrays
    train_X = torch.tensor(observed_X, dtype=torch.float)
    train_y = torch.tensor(observed_y, dtype=torch.float)
    train_y = train_y[:, None]
    train_y = standardize(-1 * train_y)

    # Make the GP
    gp = SingleTaskGP(train_X, train_y, covar_module=gpykernels.ProductStructureKernel(
        num_dims=train_X.shape[1],
        base_kernel=gpykernels.PeriodicKernel(period_length_prior=NormalPrior(360, 0.1))
    ))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Solve the optimization problem
    ei = ExpectedImprovement(gp, train_y.max())
    bounds = torch.zeros(2, train_X.shape[1])
    bounds[1, :] = 360
    candidate, acq_value = optimize_acqf(
        ei, bounds=bounds, q=1, num_restarts=64, raw_samples=64
    )
    return candidate.detach().numpy()[0, :]


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

    # Save the initial guesses
    observed_coords = [start_coords, *init_guesses.tolist()]
    observed_energies = [0.] + init_energies

    # Loop over many steps
    cur_atoms = start_atoms.copy()
    for step in range(n_steps):
        # Make a new search space
        best_coords = select_next_points_botorch(observed_coords, observed_energies)

        # Compute the energies of those points
        energy, cur_atoms = evaluate_energy(best_coords, cur_atoms, dihedrals, calc, relax)
        logger.info(f'Evaluated energy in step {step+1}/{n_steps}. Energy-E0: {energy-start_energy}')
        if energy - start_energy < np.min(observed_energies) and out_dir is not None:
            with open(out_dir.joinpath('current_best.xyz'), 'w') as fp:
                simple_write_xyz(fp, [cur_atoms])

        # Update the log
        if out_dir is not None:
            add_entry(start_coords, cur_atoms, energy)

        # Update the search space
        observed_coords.append(best_coords)
        observed_energies.append(energy - start_energy)

    # Final relaxations
    best_atoms = cur_atoms.copy()
    best_coords = observed_coords[np.argmin(observed_energies)]
    best_energy, best_atoms = evaluate_energy(best_coords, best_atoms, dihedrals, calc)
    logger.info('Performed final relaxation with dihedral constraints.'
                f'E: {best_energy}. E-E0: {best_energy - start_energy}')
    if out_dir is not None:
        add_entry(np.array(best_coords), best_atoms, best_energy)

    # Relaxations
    best_atoms.set_constraint()
    best_energy, best_atoms = relax_structure(best_atoms, calc)
    logger.info('Performed final relaxation without dihedral constraints.'
                f' E: {best_energy}. E-E0: {best_energy - start_energy}')
    best_coords = np.array([d.get_angle(best_atoms) for d in dihedrals])
    if out_dir is not None:
        add_entry(best_coords, best_atoms, best_energy)
    return best_atoms
