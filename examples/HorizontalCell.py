"""
Mosaic for Horizontal Cells from (Keeley et al., 2020) Figure 8-1.
"""

from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from OPIPP import *

CELLNAME = "Mouse Horizontal Cell"

EDGE_LEN = 300 # the length of two sides
scope = Scope(min_x=0, max_x=EDGE_LEN, # x-axis
              min_y=0, max_y=EDGE_LEN) # y-axis

########################################
#
# Create, Load and Save A Mosaic
#
########################################

# load 2D points
natural_points = np.loadtxt("examples/natural/HC/F8-1-points.txt")

# create a mosaic
natural_mosaic = Mosaic(points=natural_points, scope=scope)

# save coordinates into a texture file
natural_mosaic.save("examples/natural/HC/F8-1-points.txt", separate=False)

# Separate coordinates into two texture files
# the 1st is "examples/natural/HC/F8-1-points-x.txt" with x-coordinates
# the 2nd is "examples/natural/HC/F8-1-points-y.txt" with y-coordinates
natural_mosaic.save("examples/natural/HC/F8-1-points.txt", separate=True)

########################################
#
# Load and Save Mosaics in Pattern
#
########################################

pattern = Pattern(name=CELLNAME)

# add a natural mosaic
pattern.add_natural_mosaic(natural_mosaic)

# Load a simulated mosaic
simulated_points = np.loadtxt("examples/simulated/HC/W1_0.points")
simulated_mosaic = Mosaic(points=simulated_points, scope=scope)

# Add a simulated mosaic
pattern.add_simulated_mosaic(natural_mosaic, tag="S")

simulated_mosaic = pattern.get_simulated_mosaic(index=0, tag="S") 
# tag="default" if not specific
natural_mosaic = pattern.get_natural_mosaic(index=0)

# Discard simulated mosaics in tag "S", tag="default" if not specific
pattern.remove_mosaics(with_natural=False, simulated_tag="S")

# If with_natural=True, it also discards natural mosaics.
# If simulated_tag=None, it discards all simulated mosaics
# Therefore, this line discards all mosaics in the pattern
pattern.remove_mosaics(with_natural=True, simulated_tag=None)

pattern.load_from_files(point_fnames=["examples/natural/HC/F8-1-points.txt"], 
                        scope=scope, is_natural=True)

# list of file names
points_files = glob("examples/simulated/HC/W1_*.points")
# specifiy the tag of simulated mosiacs, # tag="default" if not specific
_ = pattern.load_from_files(point_fnames=points_files, scope=scope, 
                            is_natural=False, simulated_tag="O-PIPP")
points_files = glob("examples/simulated/HC/PIPP_*.points")
_ = pattern.load_from_files(point_fnames=points_files, scope=scope, 
                            is_natural=False, simulated_tag="PIPP")

pattern.dump_to_files(prefix="examples/simulated/HC/PIPP",
    ext="points", is_natural=False, separate=False, 
    simulated_tag="PIPP")

########################################
#
# Feature analysis in a Mosaic
#
########################################

# a numpy.darray(dtype=int) contains indices of boundary points
boundary_indices = natural_mosaic.get_boundary_indices()
# a numpy.darray(dtype=int) contains indices of effective points
effective_indices = natural_mosaic.get_effective_indices()

guy = 16 # We try to find the neighbors of this guy.
# get a list of indices of neighbors. 
neighbors = natural_mosaic.find_neighbors(p_index=guy, effective_only=False)

# NN distance and the NN neighbor of a point
nn_neighbor, nn_distance = natural_mosaic.find_nearest_neighbor(guy)
# nn_neighbor is the index of the NN neighbor.
# nn_distance is the distance from the point to its NN.

# NN distances of effective points
effective_nns = natural_mosaic.get_nns(indices=effective_indices, effective_filter=True)
# NN distances of all points
all_nns = natural_mosaic.get_nns(indices=None, effective_filter=False)

# VD areas of effective points
effective_vds = natural_mosaic.get_vorareas(indices=None, effective_filter=True)
# VD areas of all points
all_vds = natural_mosaic.get_vorareas(indices=None, effective_filter=False)

# Nearest Neighbor Regularity Index
natural_mosaic.NNRI() # 4.966138094971688
# Voronoi Domain Regularity Index
natural_mosaic.VDRI() # 5.790713936276296

########################################
#
# Probability Distribution for feature
#
########################################

# Distribution of NN distances
nn_distribution = Distribution(method="get_nns", min_value=0, max_value=50, n_bin=20)
# Or a callable method in the definition
# Distribution of VD areas
vd_distribution = Distribution(method=lambda mosaic: mosaic.get_vorareas(), 
                               min_value=0, max_value=4000, n_bin=20)

# Same as `natural_mosaic.get_nns()`
features = nn_distribution.extract_mosaic(natural_mosaic)
# Or extract features from a list of mosaics
features = nn_distribution.extract_mosaics([natural_mosaic, simulated_mosaic])

########################################
#
# Feature analysis in Pattern
#
########################################

pattern.set_feature("NN", nn_distribution)
pattern.set_feature("VD", vd_distribution)

# get values from mosaics
mosaics = [natural_mosaic]
values = nn_distribution.extract_mosaics(mosaics)
# get the histogram
hist = nn_distribution.get_hist(values)
# set the probabilities for optimization
nn_distribution.set_target(hist)

# Give the name of the feature, it will estimate 
# probabilities and put into the distribution object
probs = pattern.set_feature_target(feature_label="NN")

# data from (Keeley et al., 2020)
target = np.array([0.        , 0.        , 0.        , 0.0016756 , 0.00670241,
       0.02010724, 0.04356568, 0.08713137, 0.11394102, 0.17258713,
       0.16253351, 0.16253351, 0.1152815 , 0.06635389, 0.0325067 ,
       0.01005362, 0.00268097, 0.00134048, 0.00067024, 0.00033512, 
       0]) # n_bin=20 but len(target)=21
       # The last item represents the probability of values
       # larger than the `max_value`. 
nn_distribution.set_target(target)
vd_distribution.set_target(np.array([0.        , 0.        , 0.00608906, 0.0617784 , 0.17183275,
       0.27966751, 0.27633813, 0.13262011, 0.05992875, 0.0117453 ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        , 
       0.        ]))

# The NNRI distribution
nnri_distribution = Distribution("NNRI", 10)
pattern.set_feature("NNRI", nnri_distribution)
# The VDRI distribution
vdri_distribution = Distribution("VDRI", 10)
pattern.set_feature("VDRI", vdri_distribution)
# Do not require target probabilities

########################################
#
# Simulation and Optimization
#
########################################

density = pattern.estimate_density()
pattern.set_density(density=density)
pattern.set_density(87/90000.) # direct input the density

parameters = [7.5, 32.1206741, 2.64876305] # [δ, φ, α]
h_func = pattern.get_interaction_func(parameters) 


if __name__ == "__main__":
    

    ########################################
    #
    # Generate a new mosaic
    #
    ########################################
    mosaic = pattern.new_mosaic(scope=scope)
    mosaic, losses = pattern.simulate(mosaic=mosaic, interaction_func=h_func, 
                                      features=None, schedule=AdaptiveSchedule(), 
                                      max_step=None, update_ratop=None, 
                                      save_prefix="examples/simulated/HC/Sample", 
                                      save_step=500, verbose=True)
    mosaic.draw_points()
    pattern.add_simulated_mosaic(mosaic)
    pattern.draw_feature_hist("nn")
    pattern.draw_feature_hist("vd")
    plt.plot(losses)
    plt.show()


