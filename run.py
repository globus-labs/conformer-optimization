import json
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import logging
import sys

from ase.io.xyz import simple_write_xyz
from ase.calculators.psi4 import Psi4
from xtb.ase.calculator import XTB
import torchani

from confopt.setup import get_initial_structure, detect_dihedrals, fix_cyclopropenyl
from confopt.solver import run_optimization

logger = logging.getLogger('confsolve')

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('smiles', type=str, help='SMILES string of molecule to optimize')
    parser.add_argument('--num-steps', type=int, default=32, help='Number of optimization steps to take')
    parser.add_argument('--init-steps', type=int, default=4, help='Number of initial guesses to make')
    parser.add_argument('--level', choices=['xtb', 'ani', 'b3lyp'], default='ani', help='Level of quantum chemistry')
    args = parser.parse_args()

    # Make an output directory
    params_hash = hashlib.sha256(str(args.__dict__).encode()).hexdigest()
    out_dir = Path(__file__).parent.joinpath(f'solutions/{args.smiles}-{args.level}-{params_hash[-6:]}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_dir.joinpath('run_params.json').open('w') as fp:
        json.dump(args.__dict__, fp)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir.joinpath('runtime.log')),
                logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logger.info(f'Started optimizing the conformers for {args.smiles}')

    # Make the initial guess
    init_atoms, mol = get_initial_structure(args.smiles)
    logger.info(f'Determined initial structure with {len(init_atoms)} atoms')
    
    # Fix cycloprop. rings
    init_atoms = fix_cyclopropenyl(init_atoms, mol)

    # Detect the dihedral angles
    dihedrals = detect_dihedrals(mol)
    logger.info(f'Detected {len(dihedrals)} dihedral angles')

    # Save the initial guess
    with out_dir.joinpath('initial.xyz').open('w') as fp:
        simple_write_xyz(fp, [init_atoms])

    # Set up the optimization problem
    if args.level == 'ani':
        calc = torchani.models.ANI2x().ase()
    elif args.level == 'xtb':
        calc = XTB()
    elif args.level == 'b3lyp':
        calc = Psi4(method='b3lyp', basis='3-21g', num_threads=4, multiplicity=1, charge=1)
    else:
        raise ValueError(f'Unrecognized QC level: {args.level}')
    final_atoms = run_optimization(init_atoms, dihedrals, args.num_steps, calc, args.init_steps, out_dir)

    # Save the final structure
    with out_dir.joinpath('final.xyz').open('w') as fp:
        simple_write_xyz(fp, [final_atoms])
    logger.info('Done.')
