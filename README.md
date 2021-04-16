# Conformer Optimization

This repo contains code for optimizing conformers using active learning and quantum chemistry computations.

*Background* Conformers define the different structures with the same bonding graph but different coordinates. 
Finding the lowest-energy conformation is a common task in molecular modeling, and one that often requires significant time to solve.
We implement optimal experimental design techniques to solve this problem
following [recent](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0354-7) [emerged](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00648)
that uses Bayesian optimization find optimize dihedral angles.

## Installation

Build the environment using anaconda:

```bash
conda env create --file environment.yml --force
```

## Use

`run.py` provides a simple interface to the code. To optimize cysteine with default arguments.

```bash
python run.py "C([C@@H](C(=O)O)N)S"
```

This will produce a folder in the `solutions` directory containing the optimized geometry 
(`final.xyz`) and many other files for debugging.

Call `python run.py --help` for available options.
