from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

class Scope:
    """ 
    The rectangle area
        
    Parameters
    ----------
    min_x, max_x, min_y, max_y: float
        Axis of the rectangle.

    """
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def __eq__(self, __o: Scope) -> bool:
        return self.min_x == __o.min_x and self.max_x == __o.max_x and self.min_y == __o.min_y and self.max_y == __o.max_y

    def filter(self, points: np.ndarray, not_in: bool=False, return_idxes: bool=True) -> np.ndarray:
        mask = np.ones(points.shape[0]).astype(bool)
        mask[self.min_x > points[:, 0]] = False
        mask[self.max_x < points[:, 0]] = False
        mask[self.min_y > points[:, 1]] = False
        mask[self.max_y < points[:, 1]] = False
        ins = np.where(mask)[0]
        not_ins = np.where(1-mask)[0]
        if not_in:
            idxes = not_ins
        else:
            idxes = ins
        if return_idxes:
            return idxes
        else:
            return points[idxes]

    def get_edges_len(self) -> tuple:
        return (self.max_x-self.min_x, self.max_y-self.min_y)

    def get_area(self) -> float:
        return (self.max_x-self.min_x)*(self.max_y-self.min_y)

    def get_center(self) -> tuple:
        return ((self.max_x+self.min_x)/2, (self.max_y+self.min_y)/2)

    def get_corners(self) -> tuple:
        return ([self.min_x, self.max_y],
                [self.min_x, self.min_y],
                [self.max_x, self.min_y],
                [self.max_x, self.max_y])

    def get_random_loc(self, size: int=1) -> np.ndarray:
        locs = np.zeros((size, 2))
        locs[:, 0] = np.random.rand(size)*(self.max_x-self.min_x)+self.min_x
        locs[:, 1] = np.random.rand(size)*(self.max_y-self.min_y)+self.min_y
        return locs

    def get_reflect_points(self, point_x: float, point_y: float) -> tuple:
        assert self.min_x <= point_x <= self.max_x
        assert self.min_y <= point_y <= self.max_y
        points = []
        if point_x == self.min_x:
            points.append([self.min_x-0.01, point_y])
        elif point_x == self.max_x:
            points.append([self.max_x+0.01, point_y])
        elif point_x-self.min_x < self.max_x-point_x:
            points.append([self.min_x*2-point_x, point_y])
        else:
            points.append([self.max_x*2-point_x, point_y])
        if point_y == self.min_y:
            points.append([point_x, self.min_y-0.01])
        elif point_y == self.max_y:
            points.append([point_x, self.max_y+0.01])
        elif point_y-self.min_y < self.max_y-point_y:
            points.append([point_x, self.min_y*2-point_y])
        else:
            points.append([point_x, self.max_y*2-point_y])
        return points
    
    def distance2boundary(self, point_x: float, point_y: float) -> float:
        return np.abs([point_x-self.min_x, point_x-self.max_x, point_y-self.min_y, point_y-self.max_y]).min()
    
    def view(self, edgecolor: str="k", facecolor: str="gray", alpha: float=0.5):
        ax = plt.subplot()
        rect = Rectangle((self.min_x, self.min_y), 
                        self.max_x-self.min_x, self.max_y-self.min_y)
        pc = PatchCollection([rect], facecolor=facecolor, 
                            alpha=alpha, edgecolor=edgecolor)
        ax.add_collection(pc)
        ax.set_xlim(xmin=self.min_x-(self.max_x-self.min_x)/5, 
                    xmax=self.max_x+(self.max_x-self.min_x)/5)
        ax.set_ylim(ymin=self.min_y-(self.max_y-self.min_y)/5, 
                    ymax=self.max_y+(self.max_y-self.min_y)/5)
        plt.axis("scaled")
        plt.show()
        
