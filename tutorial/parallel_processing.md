- [Overview](#overview)
- [Select the pattern](#select-the-pattern)
- [MPI](#mpi)
- [Simulation](#simulation)

# Overview

In this part, we show a case that uses **Message Passing Interface (MPI)** to archive multiple mosaic simulations in parallel. Here we chose [openmpi](https://www.open-mpi.org/)  and the [mpi4py](https://mpi4py.readthedocs.io/en/stable/), both of them can be replaced by other [MPI implementations](https://en.wikipedia.org/wiki/Message_Passing_Interface#Official_implementations) and [bindings on python](https://en.wikipedia.org/wiki/Message_Passing_Interface#Python).

The following Python script is in [scripts/parallel_simulate.py](../scripts/parallel_simulate.py). Then, you can run the `mpirun` command to begin the simulation process. For example, we use the script to generate 10 mosaics as,

```shell
mpirun -n 10 python scripts/parallel_simulate.py
# `mpirun` can be replaced by `mpiexec` or others, depending on your mpi tools
# `10` is the number of simulated cases during this process
```
Please check [the OpenMPI document](https://www.open-mpi.org/doc/current/man1/mpirun.1.php) for more information. 

# Select the pattern

The first part of the Python script is to decide the `Pattern`, features, and output in simulation.

```python
# Horizontal Cell Pattern
from examples.HorizontalCell import pattern as HC

# Mosaic Plane, and the interaction function
from examples.HorizontalCell import scope, h_func

# optimization features
features = ["NN", "VD"]

# The prefix of output files
save_prefix = "examples/simulated/HC/W1"
# chose a random seed
seed = 89375
# print the optimized process or not
verbose = True 
```

# MPI

The second part is to import the MPI tool and decide the rank (ID) of the thread.

```python
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # the ID of this thread

# reseed to avoid the unique random generator 
np.random.seed(seed=rank*seed)
```

# Simulation

The last step is simulating and saving the generated mosaic into local files with the rank so that you can output files that are not overlapped by each other.

```python
# use the O-PIPP method to generate new mosaics
new_mosaic = HC.new_mosaic(scope=scope)
new_mosaic, losses = HC.simulate(new_mosaic, h_func, features=features, verbose=verbose)
# save the mosaic with the rank 
new_mosaic.save("%s_%d.points"%(save_prefix, rank), separate=False)
np.savetxt("%s_%d.losses"%(save_prefix, rank), losses)
```

