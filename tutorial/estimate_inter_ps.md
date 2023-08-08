- [Overview](#overview)
- [Installation](#installation)
- [Save points into local files](#save-points-into-local-files)
- [Non-parametric estimation in R](#non-parametric-estimation-in-r)
- [Parametric estimation in OPIPP](#parametric-estimation-in-opipp)

# Overview

The **interaction function** estimates the probability of a distance between any pairs of points. It is well used in the point process model of spatial patterns. The common formulation of the function for mosaic simulation is

$$h(u)= \begin{cases} 0,\quad u\leq δ\\ 1-exp(-((u-δ)/φ)^{α}), \quad u>δ \end{cases} \tag{1},$$

where $h(u)$ denotes the interaction function and $u$ denotes the distance. If `u>δ`, the $h(u)$ chose a sigmoid function. Otherwise, the function returns `0`. The threshold `δ` usually uses the minimal distance in the spatial pattern, while `φ` and `α` needs a specific estimation method to find their suitable values. 

In this part, we introduce how to use `R` and `OPIPP` to get the parameters of the interaction function $h(u)$. We follow the method in [(Elgen et al., 2005)](https://doi.org/10.1017/S0952523805226147) that nonparametrically estimate values alongside distances and then use a non-linear least squares method to find parameters in Eqn (1).

# Installation

The estimation requires the [spatstat](https://spatstat.org/) libaray in [R](https://www.r-project.org/) environment. For installing `R`, you can get the download and related instructions from [here](https://cran.r-project.org/). After installation of the environment, please open `R`, the `RGui` or `Rterm`, usually both are already installed. You can find a console and then type the following line to install the `spatstat` package,

```R
install.packages('spatstat')
```

We recommend learning the basic of the `R` language for further usage. In this part, we suppose you know the basic of a retinal mosaic and the usage of the `OPIPP` module. Furthermore, you need to modify R scripts for your simulations.

# Save points into local files

To estimate the interaction function from a mosaic, you need to dump points into local files so that the `R` can access the data. We recommend using the `Mosaic.save` method to complete this goal. For example, we save points in the `natural_mosaic` into two files, as

```python
# Separate coordinates into two texture files
# the 1st is "examples/natural/HC/F8-1-points-x.txt" with x-coordinates
# the 2nd is "examples/natural/HC/F8-1-points-y.txt" with y-coordinates
natural_mosaic.save("examples/natural/HC/F8-1-points.txt", separate=True)
```

# Non-parametric estimation in R

The next step is to run the `R` environment with the following script.

```R
# import libraries
library(spatstat)
library(pracma)

# load files that have coordinates of points
x <- scan("examples/natural/HC/F8-1-points-x.txt") # x axis of points
y <- scan("examples/natural/HC/F8-1-points-y.txt") # y axis of points

# x-axis values (distances) in the interaction function
r <- linspace(2, 50, 17) # similar to numpy.linsapce
# r is an arithmetic sequence from 2 to 50 and the step is 3
# It means the estimate will yield a list of probabilities corresponding to distances from 2 to 50.
# Please modify these values depending on the value range of distances in a given mosaic.

# define a Poisson point process
u <- ppp(x, y, c(0, 300), c(0, 300)) 
# The 3rd argument is c(min_x, max_y) 
# and the 4th is c(min_y, max_y), 
# corresponding to attributes in OPIPP.Scope
# Please modify them when apply to other mosaics.

# non-parametric estimate
fit <- ppm(u ~1, PairPiece(r = r))

# draw results
plot(fitin(fit)) 

# Write results to local files
write.table(parameters(fit)$gammas, "examples/natural/HC/F8-1-h-gammas.txt", row.names = FALSE, col.names = FALSE)
write.table(parameters(fit)$r, "examples/natural/HC/F8-1-h-r.txt", row.names = FALSE, col.names = FALSE)
```

# Parametric estimation in OPIPP

The last step is to use the `estimate_interaction` method to fit the curve of $h(u)$ and yield parameters ($δ$, $φ$, $α$) in it. Import the method as

```python
from OPIPP.utils import estimate_interaction
```

$δ$ is the minimum distance among cells in a mosaic or a series of mosaics from homotypic cells. We set $δ=7.5$ which is from the dataset from [(Keeley et al., 2020)](https://doi.org/10.1002/cne.24880). For finding a minimal distance in a mosaic, you can use the `Mosaic.get_distances` method. For example, we get the minimal distance in the `natural_mosiac` as

```python
min_distance = natural_mosaic.get_distances().min()
```

To get parameters of the interaction function, you need load estimation results by `R` and then deliver these data and the value of delta to the `estimate_interaction` method. The return is three parameters of the function, as

```python
import numpy as np

# load non-parametric estimation results
gammas = np.loadtxt("examples/natural/HC/F8-1-h-gammas.txt")
rs = np.loadtxt("examples/natural/HC/F8-1-h-r.txt")

# return is [7.5, 32.12066166693056, 2.6487686148563743]
# corresponding to [δ, φ, α]
parameters = estimate_interaction(gammas, rs, delta=7.5, draw=True) 
```

<p align="center">
<img src="imgs/eps-res.png" width="230">
<figcaption align = "center">Curves of the interaction function for the HC mosaic.</figcaption>
</p>

Here, you can use `draw=True` to check the results of estimation and fitting.

In the end, we get the interaction function of the HC mosaic, as

```python
# at last, we get the interaction function
h_func = pattern.get_interaction_func(parameters) 
```
