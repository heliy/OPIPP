# OPIPP: Optimization-based Pairwise Interaction Point Process

A Python implementation of **OPIPP**, a method for precisely generating artifical **retinal mosaics**, the spatial organization of retinal neurons. [Here](tutorial/0.background.md) is a short introduction of the background.

## Pipeline

![overview](tutorial/imgs/rm-overview.png)

We recommend the "import-analysis-simulation" pipeline for generating artificial mosaics and purpose a tutorial for each step, as
1. [Importing retinal spatial pattern datasets from local files](tutorial/1.import.md)
2. [Analyzing and visualizing spatial patterns of mosaics](tutorial/2.analysis.md)
3. [Simulating artifial retinal mosaics](tutorial/3.simulation.md)

Users are welcome to [extend and customize methods](tutorial/4.extension.md) for feature calculation and mosaic simulation.

## Tools

Here are useful tools that are not implemented in the current version. We plan to apply these methods in future development.

- [Estimate parameters of mosaic simulation by R](tutorial/estimate_inter_ps.md)
- [Parallel process for mosaic generation by MPI](tutorial/parallel_processing.md)


# Install

## PyPI (pip)

```console
pip install git+https://github.com/heliy/OPIPP
```

<!-- or 

```console
pip install OPIPP
``` -->

## Dependencies

- python >=3.8
- numpy >= 1.20.0
- scipy >= 1.9.0
- matplotlib >= 3.2.0
- networkx >= 2.0.0

# Citation

- TODO

# References

- The example mosaic for retinal horizontal cells and related spatial features are from [(Keeley et al., 2020)](https://doi.org/10.1002/cne.24880).

- In the optimization, we use the adaptive simulated annealing algorithm from [(Ingber, 1993)](https://optimization-online.org/wp-content/uploads/2001/03/291.pdf).



