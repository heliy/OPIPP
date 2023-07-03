#coding:UTF-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.qhull import Delaunay
from scipy.spatial import Voronoi
import networkx as nx

from .utils import get_poly_area
from .scope import Scope

class PointNetwork(nx.Graph):
    def __init__(self, points, scope: Scope, **attr):
        """ 
        Graph of points in a mosaic.
        
        Parameters
        ----------
        points: np.ndarray or list
            locations of cells.

        scope: Scope
            the area of the mosaic.
        """
        super().__init__(**attr)
        self.points = np.array(points)
        self.scope = scope
        self.__set_effective_filter()

        # add points mirroring with inner points by edges to ensure the correctness of voronoi domains
        appended_points = []
        effective_filter = self.get_effective_filter()
        for index, is_effective in enumerate(effective_filter):
            if not is_effective:
                reflect_points = self.scope.get_reflect_points(self.points[index])
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

    def __edge_length(self, u, v):
        try:
            return self.edges[u, v]["length"] 
        except:
            return self.edges[v, u]["length"]
        
    def __set_effective_filter(self):
        vor = Voronoi(self.points, qhull_options='Qbb Qc Qx')
        outs = self.scope.filter(vor.vertices, not_in=True)
        outs = set(list(outs))
        self.effective_filter = np.zeros(self.points.shape[0]).astype(bool)
        for i in range(self.points.shape[0]):
            region = list(vor.regions[vor.point_region[i]])
            if len(region) > 0 and -1 not in region and len(set(region)&outs) == 0:
                self.effective_filter[i] = True

    def get_effective_filter(self) -> np.ndarray:
        return self.effective_filter
    
    def __get_features(self, feauture_func, indices: np.ndarray=None, effective_filter: bool=True) -> list:
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

    def get_surround_points(self):
        """ Gets boundary points """
        in_surrounds = np.arange(self.points.shape[0])[1-self.get_effective_filter()]
        return in_surrounds

    def get_distances(self, indices: list=None):
        """ Gets distances of edges related to given cells, all edges if not specific """
        if indices is not None:
            edge_iter = self.edges(indices)
        else:
            edge_iter = self.edges()
        distances = [self.__edge_length(u, v) for u, v in edge_iter]
        return np.array(distances)

    def find_neighbors(self, pindex):
        """ Gets neighbors of a given cell """
        return list(self.neighbors(pindex))

    def find_nearest_neighbor(self, node: int) -> tuple:
        """ Gets its NN neighbor and NN distance of a given cell """
        min_distance = self.scope.get_area()
        min_neighbor = -1
        for neighbor in self.find_neighbors():
            distance = self.__edge_length(neighbor, node)
            if distance < min_distance:
                min_distance = distance
                min_neighbor = neighbor
        return min_neighbor, min_distance

    def get_nn_graph(self, nodes=None) -> nx.DiGraph:
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

    def draw(self, edge_color='r', node_color='b', node_size=5):
        nx.draw_networkx(self, pos=self.points, edge_color=edge_color, node_color=node_color, node_size=node_size)
        plt.show()