""" example script for parallelized simulation """

# select the pattern
from examples.HorizontalCell import pattern as HC
from examples.HorizontalCell import scope, h_func
features = ["nn", "vd"]
# set output files
save_prefix = "examples/simulated/HC/W1"
# chose a random seed
seed = 89375
# print the optimize process or not
verbose = True 
# then the command line is `mpirun -n 10 python parallel_simulate.py`
# `mpirun` can be replaced by `mpiexec` or others, depending on your mpi tools
# `10` is the total number of simulated cases during this process

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# reseed to avoid the unique random generator 
np.random.seed(seed=rank*seed)

# use the O-PIPP method to generate new mosaics
new_mosaic = HC.new_mosaic(scope=scope)
new_mosaic, losses = HC.simulate(new_mosaic, h_func, features=features, verbose=verbose)
new_mosaic.save("%s_%d.points"%(save_prefix, rank), separate=False)
np.savetxt("%s_%d.losses"%(save_prefix, rank), losses)

# # or use the PIPP method to generate new mosaics
# save_prefix = "examples/simulated/HC/PIPP"
# new_mosaic = HC.new_mosaic(scope=scope)
# new_mosaic, losses = HC.simulate(new_mosaic, h_func, features=features, verbose=verbose, schedule=None, max_step=20)
# new_mosaic.save("%s_%d.points"%(save_prefix, rank), separate=False)
# np.savetxt("%s_%d.losses"%(save_prefix, rank), losses)





