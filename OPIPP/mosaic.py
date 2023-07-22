from typing import Generator, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import Delaunay
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
import networkx as nx

from .utils import get_poly_area, get_distances
from .scope import Scope

class Mosaic(nx.Graph):
    """ 
    A mosaic with given locations.
        
    Parameters
    ----------
    points: np.ndarray or list
        locations of cells.

    scope: Scope
        the area of the mosaic.
    """
    def __init__(self, points: np.ndarray, scope: Scope, **attr):
        super().__init__(**attr)
        self.points = np.array(points)
        assert len(self.points.shape) == 2
        self.scope = scope
        self.__set_effective_filter()

        # add points mirroring with inner points by edges to ensure the correctness of voronoi domains
        appended_points = []
        effective_filter = self.get_effective_filter()
        for index, is_effective in enumerate(effective_filter):
            if not is_effective:
                reflect_points = self.scope.get_reflect_points(*self.points[index])
                appended_points.extend(reflect_points)
        appended_points = np.array(appended_points)
        appended_points = appended_points[self.scope.filter(appended_points, not_in=True)]
        all_points = self.points.tolist()+appended_points.tolist()
        triangle = Delaunay(all_points)
        self.extended_vor = Voronoi(all_points, qhull_options='Qbb Qc Qx')

        # build the network of points without reflected points
        maxN = self.points.shape[0]
        for path in triangle.simplices:
            a, b, c = tuple(path)
            if a < maxN and b < maxN:
                nx.add_path(self, [a, b])
            if b < maxN and c < maxN:
                nx.add_path(self, [b, c])
            if a < maxN and c < maxN:
                nx.add_path(self, [a, c])
        for u, v in self.edges():
            distance = euclidean(self.points[u], self.points[v])
            self.edges[u, v]["length"] = distance

    def __edge_length(self, u: int, v: int) -> float:
        try:
            return self.edges[u, v]["length"] 
        except:
            return self.edges[v, u]["length"]
        
    def get_points_n(self) -> int:
        return self.points.shape[0]

    def save(self, fname: str, split: bool=False) -> None:
        if split:
            cons = fname.split(".")
            pre_name = ".".join(cons[:-1])
            np.savetxt("%s-x.%s"%(pre_name, cons[-1]), self.points[:, 0], fmt="%f")
            np.savetxt("%s-y.%s"%(pre_name, cons[-1]), self.points[:, 1], fmt="%f")
        else:
            np.savetxt(fname, self.points, fmt="%f")

    ########################################
    #
    # Methods for boundary effects
    #
    ########################################

    def __set_effective_filter(self) -> None:
        fs = []
        mask = np.zeros(self.get_points_n()).astype(bool)
        for i, point in enumerate(self.points):
            distance2boundary = self.scope.distance2boundary(*point)
            mask[:] = True
            mask[i] = False
            distance2others = get_distances(point, self.points[mask]).min()
            fs.append(distance2others < distance2boundary)
        self.effective_filter = np.array(fs)

    def get_effective_filter(self) -> np.ndarray:
        return self.effective_filter
    
    def iter_effective_indices(self) -> Generator[int, None, None]:
        for p_index, is_effective in enumerate(self.get_effective_filter()):
            if is_effective:
                yield p_index

    def get_boundary_indices(self) -> np.ndarray:
        """ Gets indices of boundary points """
        in_surrounds = np.arange(self.points.shape[0])[1-self.get_effective_filter()]
        return in_surrounds
    
    ########################################
    #
    # Feature-related methods
    #
    ########################################

    def __get_features(self, feauture_func: Callable, indices: np.ndarray=None, effective_filter: bool=True) -> list:
        if indices is None or len(indices) == 0:
            indices = range(self.points.shape[0])
        if effective_filter:
            node_filter = self.get_effective_filter()
        else:
            node_filter = np.ones(self.points.shape[0]).astype(bool)
        values = []
        for index in indices:
            if node_filter[index]:
                feature = feauture_func(index)
                values.append(feature)
        return values

    def get_vorareas(self, indices: np.ndarray=None, effective_filter: bool=True) -> list:
        """ 
        get values of voronoi areas
        
        Parameters
        ----------
        indices: dict or list, optional(default=None)
            The indices of cells to query. Without specification, it will return values of all cells.

        effective_filter: bool, optional(default=True)
            If True, it will exclude boundary cells.
        """
        def vorarea_func(index):
            region = self.extended_vor.regions[self.extended_vor.point_region[index]]
            region = self.extended_vor.vertices[region]
            area = get_poly_area(region[:, 0], region[:, 1])
            return area
        return self.__get_features(feauture_func=vorarea_func, indices=indices, effective_filter=effective_filter)

    def find_neighbors(self, p_index: int, effective_only=False) -> list:
        neighbors = list(self.neighbors(p_index))
        if effective_only:
            return list(set(neighbors).difference(set(self.pnet.get_boundary_indices())))
        return neighbors

    def find_nearest_neighbor(self, node: int) -> tuple:
        """ Gets its NN neighbor and NN distance of a given cell """
        min_distance = self.scope.get_area()
        min_neighbor = -1
        for neighbor in self.find_neighbors(node):
            distance = self.__edge_length(neighbor, node)
            if distance < min_distance:
                min_distance = distance
                min_neighbor = neighbor
        return min_neighbor, min_distance

    def get_nn_graph(self, nodes: list=None) -> nx.DiGraph:
        """ 
        Gets a directed subgraph containing all points and NN edges 
        """
        if nodes is None:
            nodes = self.nodes
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        for node in nodes:
            nn_neighbor, _ = self.find_nearest_neighbor(node)
            graph.add_edge(node, nn_neighbor)
        return graph

    def get_nns(self, indices: np.ndarray=None, effective_filter: bool=True) -> list:
        """ 
        get values of nearest neighbor distances
        
        Parameters
        ----------
        indices: dict or list, optional(default=None)
            The indices of cells to query. Without specification, it will return values of all cells.

        effective_filter: bool, optional(default=True)
            If True, it will exclude boundary cells.
        """
        def nn_func(index):
            _, nn_distance = self.find_nearest_neighbor(index)
            return nn_distance
        return self.__get_features(feauture_func=nn_func, indices=indices, effective_filter=effective_filter)

    def get_distances(self, indices: list=None)-> np.ndarray:
        """ Gets distances of edges related to given cells, all edges if not specific """
        if indices is not None:
            edge_iter = self.edges(indices)
        else:
            edge_iter = self.edges()
        distances = [self.__edge_length(u, v) for u, v in edge_iter]
        return np.array(distances)
        
    ########################################
    #
    # Visualization methods
    #
    ########################################

    def draw_points(self, highlights: list=None, draw_grid: bool=True, grid: int=1, 
                    nonhighlight_alpha: float=0.3, equal_aspect: bool=True,
                    point_args: dict={"color": "r", "s": 5}) -> None:
        ax = plt.subplot()
        if equal_aspect:
            ax.set_aspect('equal')
        plt.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        plt.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=1.0, **point_args)
        ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, grid+1))
        ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, grid+1))
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.grid(draw_grid)
        plt.show()

    def draw_neighbors(self, highlights: list=None, grid: int=1, 
                       nonhighlight_alpha: float=0.3, equal_aspect: bool=True,
                       point_args={"s": 5, "color": "r"}, 
                       edge_args={"lw": 0.5, "color": "gray"}) -> None:
        ax = plt.subplot()
        if equal_aspect:
            ax.set_aspect('equal')
        plt.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        plt.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=1.0, **point_args)
        ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, grid+1))
        ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, grid+1))
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        for edge in self.edges:
            ax.plot([self.points[edge[0], 0], self.points[edge[1], 0]],
                    [self.points[edge[0], 1], self.points[edge[1], 1]], **edge_args)
        plt.show()

    def draw_nn_graph(self, highlights: list=None, grid: int=1, 
                        nonhighlight_alpha: float=0.3, equal_aspect: bool=True,
                        network_args={"edge_color": "k", "node_size": 0, "with_labels": False}, 
                        points_args={"color": "r", "s": 5}) -> None:
        ax = plt.subplot()
        if equal_aspect:
            ax.set_aspect('equal')
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        nn_graph = self.get_nn_graph(nodes=highlights)
        nx.draw_networkx(nn_graph, pos=self.points, **network_args)
        plt.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **points_args)
        plt.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=1.0, **points_args)
        ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, grid+1))
        ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, grid+1))
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def draw_vorareas(self, highlights: list=None, nonhighlight_alpha: float=0.3, 
                      equal_aspect: bool=True, region_args={"color": "gray", "alpha": 0.5},
                      voronoi_args={"show_points": False, "line_width": 0.5},
                      node_args={"color": "r", "s": 10}) -> None:
        ax = plt.subplot()
        if equal_aspect:
            ax.set_aspect('equal')
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        vor = self.extended_vor
        for rindex in vor.point_region[:self.get_points_n()]:
            polygon = [vor.vertices[i] for i in vor.regions[rindex]]
            ax.fill(*zip(*polygon), **region_args)
        voronoi_plot_2d(vor, ax=ax, **voronoi_args)

        p_indices = list(self.iter_effective_indices())
        ax.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **node_args)
        ax.scatter(self.points[p_indices][:, 0], self.points[p_indices][:, 1], alpha=1.0, **node_args)
        plt.xlim([self.scope.min_x, self.scope.max_x])
        plt.ylim([self.scope.min_y, self.scope.max_y])
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()
    
