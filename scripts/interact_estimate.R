# a example script to calculate the interaction function by R

# First, install `R`(https://www.r-project.org/) 
# and install the package `spatstat`(https://spatstat.org/)

# Next, save the x/y locations of points into local files
# For instance, we save the points of the HC mosaic as
# python```
# from examples.HorizontalCell import nature_mosaic
# nature_mosaic.save("examples/nature/HC/F8-1-points.txt", split=True)
# ```

# Then: running R

library(spatstat)
library(pracma)

len <- 300 # the length of the side of the area
x <- scan("examples/nature/HC/F8-1-points-x.txt") # x axis of points
y <- scan("examples/nature/HC/F8-1-points-y.txt") # y axis of points
r <- linspace(2, 50, 17) # distances in the interaction function

u <- ppp(x, y, c(0, len), c(0, len))
fit <- ppm(u ~1, PairPiece(r = r))
plot(fitin(fit)) # draw results

# write results to local files
write.table(parameters(fit)$gammas, "examples/nature/HC/F8-1-h-gammas.txt", row.names = FALSE, col.names = FALSE)
write.table(parameters(fit)$r, "examples/nature/HC/F8-1-h-r.txt", row.names = FALSE, col.names = FALSE)

# Last, we use OPIPP.utils.estimate_interaction to get parameters
# python```
# from OPIPP.utils import estimate_interaction
# gammas = np.loadtxt("examples/nature/HC/F8-1-h-gammas.txt")
# rs = np.loadtxt("examples/nature/HC/F8-1-h-r.txt")
# estimate_interaction(gammas, rs, delta=7.5, draw=True) # delta is the mininal NN distance
# the return of the calculation is [7.5, 32.12066166693056, 2.6487686148563743]
# used in the examples.HorizontalCell.h_func
