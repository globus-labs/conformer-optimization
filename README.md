# Conformer Optimization

This repo contains code for optimizing conformers using active learning and quantum chemistry computations.

*Background* Conformers define the different structures with the same bonding graph but different coordinates. 
Finding the lowest-energy conformation is a common task in molecular modeling, and one that often requires significant time to solve.
This homework is designed to explore the use of optimal experimental design techniques, which have [recently](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0354-7) [emerged](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00648) as a potential tool for accelerating the conformer search.

## Installation

Build the environment using anaconda:

```bash
conda env create --file environment.yml --force
```
