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
# Build the Pattern by nature mosaics
#
########################################

nature_points = np.loadtxt("examples/nature/Keeley19/F8-1-points.txt")
nature_mosaic = Mosaic(points=nature_points, scope=scope)
pattern.add_nature_mosaic(nature_mosaic)
density = pattern.estimate_density()
pattern.set_density(density=density)
pattern.estimate_feature("nn")
pattern.estimate_feature("vd")
