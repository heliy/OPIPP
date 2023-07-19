# select the pattern
from examples.Keeley19_1 import pattern as HC
from examples.Keeley19_1 import scope, h_func
# set output files
save_prefix = "examples/simulated/Keeley19/HC/W1"
# chose a random seed
seed = 89375
# print the optimize process or not
verbose = True 
# then the command line is `mpirun -n 50 python parallel_simulate.py`
# `mpirun` can be replaced by `mpiexec` or others, depending on your mpi tools
# `50` is the total number of simulated cases during this process

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# reseed to avoid the unique random generator 
np.random.seed(seed=rank*seed)

new_mosaic = HC.new_mosaic(scope=scope)
new_mosaic, losses = HC.simulate(new_mosaic, h_func, features=["nn", "vd"], verbose=verbose)
new_mosaic.save("%s_%d.points"%(save_prefix, rank), split=False)
np.savetxt("%s_%d.losses"%(save_prefix, rank), losses)





