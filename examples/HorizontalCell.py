"""
Mosaic for Horizontal Cells from (Keeley et al., 2019) Figure 8-1.
"""

import numpy as np

from OPIPP import *

CELLNAME = "Mouse Horizontal Cell"
EDGE_LEN = 300
MAX_NN = 50
N_NN = 20
MAX_VD = 4000
N_VD = 20

scope = Scope(min_x=0, max_x=EDGE_LEN, 
              min_y=0, max_y=EDGE_LEN)
nn_distribution = Distribution(max_value=MAX_NN, n_bin=N_NN)
vd_distribution = Distribution(max_value=MAX_VD, n_bin=N_VD)
pattern = Pattern(name=CELLNAME)
pattern.set_feature("nn", nn_distribution, get_nns)
pattern.set_feature("vd", vd_distribution, get_vorareas)

########################################
#
# Build the Pattern by natural mosaics
#
########################################

natural_points = np.loadtxt("examples/natural/HC/F8-1-points.txt")
natural_mosaic = Mosaic(points=natural_points, scope=scope)
pattern.add_natural_mosaic(natural_mosaic)
density = pattern.estimate_density()
pattern.set_density(density=density)
pattern.estimate_feature("nn")
pattern.estimate_feature("vd")

########################################
#
# OR Build the Pattern by probabilities
#
########################################

nn_distribution.set_target(np.array([0.        , 0.        , 0.        , 0.0016756 , 0.00670241,
       0.02010724, 0.04356568, 0.08713137, 0.11394102, 0.17258713,
       0.16253351, 0.16253351, 0.1152815 , 0.06635389, 0.0325067 ,
       0.01005362, 0.00268097, 0.00134048, 0.00067024, 0.00033512, 0]))
vd_distribution.set_target(np.array([0.        , 0.        , 0.00608906, 0.0617784 , 0.17183275,
       0.27966751, 0.27633813, 0.13262011, 0.05992875, 0.0117453 ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]))
# pattern.set_density(87/90000.) # direct set the density

h_func = pattern.get_interaction_func([7.5, 32.1206741, 2.64876305])


if __name__ == "__main__":
    
    ########################################
    #
    # Load simulated mosaics
    #
    ########################################
    from glob import glob

    # load simulated mosaics by the O-PIPP method
    points_files = glob("examples/simulated/HC/W1_*.points")
    pattern.load_from_files(points_files, scope=scope, is_natural=False, simulated_tag="O-PIPP")
    pattern.draw_feature_hist("nn", simulated_tag="O-PIPP")
    pattern.draw_feature_hist("vd", simulated_tag="O-PIPP")

    # load simulated mosaics by the PIPP method
    points_files = glob("examples/simulated/HC/PIPP_*.points")
    pattern.load_from_files(points_files, scope=scope, is_natural=False, simulated_tag="PIPP")
    pattern.draw_feature_hist("nn", simulated_tag="PIPP")
    pattern.draw_feature_hist("vd", simulated_tag="PIPP")
    
#     # draw RIs of two groups of simulated mosaics
#     pattern.set_feature("NNRI", Distribution(10, 1), get_NNRI)
#     pattern.draw_values_box(False, "NNRI", False, ["O-PIPP", "PIPP"])
#     pattern.set_feature("VDRI", Distribution(10, 1), get_VDRI)
#     pattern.draw_values_box(False, "VDRI", False, ["O-PIPP", "PIPP"])

#     # draw the KL divergency of two groups of simulated mosaics
#     pattern.draw_values_box(True, "nn", False, ["O-PIPP", "PIPP"])
#     pattern.draw_values_box(True, "vd", False, ["O-PIPP", "PIPP"])
#     pattern.draw_value_bars(True, {"vd": "brown", "nn": "gloden"}, method=np.mean, simulated_tags=["O-PIPP", "PIPP"])

#     ########################################
#     #
#     # Generate a new mosaic
#     #
#     ########################################
#     mosaic = pattern.new_mosaic(scope=scope)
#     mosaic, losses = pattern.simulate(mosaic, h_func, None, AdaptiveSchedule(), save_prefix="examples/simulated/HC/Sample", save_step=1000)
#     mosaic.draw_vorareas()
#     pattern.add_simulated_mosaic(mosaic)
#     pattern.draw_feature_hist("nn")
#     pattern.draw_feature_hist("vd")

