import numpy as np

def get_poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_poly_centeroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_distances(loc, pointsArray):
    return np.sqrt((pointsArray[:, 0]-loc[0])**2+(pointsArray[:, 1]-loc[1])**2)
