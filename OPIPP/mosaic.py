#coding:UTF-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import networkx as nx

from .utils import get_poly_centeroid
from .scope import Scope
from .graph import PointNetwork

class Mosaic:
    """ 
    A mosaic with given locations.
        
    Parameters
    ----------
    points: np.ndarray or list
        locations of cells.

    scope: Scope
        the area of the mosaic.
    """
    def __init__(self, points: np.ndarray, scope: Scope):
        self.points = np.array(points)
        assert len(self.points.shape) == 2
        self.scope = scope
        self.pnet = PointNetwork(points=self.points, scope=self.scope)

    def get_points_n(self):
        return self.points.shape[0]

    def iter_effective_indices(self):
        for p_index, is_effective in enumerate(self.pnet.get_effective_filter()):
            if is_effective:
                yield p_index

    def get_effective_nns(self):
        nns = []
        for p_index, is_effective in enumerate(self.pnet.get_effective_filter()):
            if is_effective:
                _, nn = self.pnet.find_nearest_neighbor(p_index)
            nns.append(nn)
        return np.array(nns)

    def get_effective_vorareas(self):
        return np.array(self.pnet.get_vorareas(effective_filter=True))

    def nnri(self):
        distances = self.get_effective_nns()
        return distances.mean()/distances.std()

    def vdri(self):
        areas = self.get_effective_vorareas()
        return areas.mean()/areas.std()

    def find_neighbors(self, p_index, effective_only=False):
        neighbors = self.pnet.find_neighbors(p_index)
        if effective_only:
            return list(set(neighbors).difference(set(self.pnet.get_boundary_indices())))
        return neighbors
        
    ########################################
    #
    # Visualization methods
    #
    ########################################

    def get_centeroids(self):
        domains = self.net.get_domains()
        centeroids = np.zeros(self.points.shape)
        centeroids[:] = np.inf
        for key in domains:
            centeroids[key] = get_poly_centeroid(domains[key])
        return centeroids
    
    def draw_centeroids_scatter(self, edge_color='gray'):
        centeroids = self.get_centeroids()
        indices = centeroids[:, 0] != np.inf
        N = np.where(indices)[0].shape[0]
        pos = np.concatenate((centeroids[indices], self.points[indices])).reshape((N*2, 2))
        graph = nx.DiGraph()
        graph.add_nodes_from(range(N*2))
        graph.add_edges_from([(i, i+N) for i in range(N)])

        ax = plt.subplot()
        nx.draw_networkx(graph, pos=pos, edge_color=edge_color, node_size=0, with_labels=False)
        plt.scatter(pos[:N, 0], pos[:N, 1], color='b')
        plt.scatter(pos[N:, 0], pos[N:, 1], color='r')
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        plt.show()

    def draw_centeroids_centered(self):
        centeroids = self.get_centeroids()
        indices = centeroids[:, 0] != np.inf
        N = np.where(indices)[0].shape[0]
        pos = self.points[indices]-centeroids[indices]
        ax = plt.subplot()
        ax.set_aspect('equal')
        plt.scatter(pos[1:, 0], pos[1:, 1], color='r', s=5)
        plt.scatter([0], [0], color='gray', s=5)
        lim = np.abs(pos).max()*1.5
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        plt.show()


    def draw_points(self, highlights=None, grid=1, color='r', size=5):
        # background
        ax = plt.subplot()
        ax.set_aspect('equal')
        plt.scatter(self.points[:, 0], self.points[:, 1], color=color, s=size, alpha=0.3)
        if highlights is None:
            highlights = list(self.iter_effective_point_indices())
        plt.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], color=color, s=size, alpha=1.0)
        ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, grid+1))
        ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, grid+1))
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        plt.grid()
        plt.show()

    def draw_triangulation(self, grid=1, color='r', size=5):
        # background
        ax = plt.subplot()
        ax.set_aspect('equal')
        plt.scatter(self.points[:, 0], self.points[:, 1], color=color, s=size, alpha=0.3)
        p_indices = list(self.iter_effective_point_indices())
        plt.scatter(self.points[p_indices][:, 0], self.points[p_indices][:, 1], color=color, s=size, alpha=1.0)
        ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, grid+1))
        ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, grid+1))
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        for edge in self.net.edges:
            ax.plot([self.points[edge[0], 0], self.points[edge[1], 0]], [self.points[edge[0], 1], self.points[edge[1], 1]], color='gray', lw=0.5)
        plt.grid()
        plt.show()

    def draw_nn_graph(self, edge_color='k', node_color='r', node_size=5):
        p_indices = list(self.iter_effective_point_indices())
        nn_graph = self.net.get_nn_graph(nodes=p_indices)
        ax = plt.subplot()
        ax.set_aspect('equal')
        nx.draw_networkx(nn_graph, pos=self.points, edge_color=edge_color, node_size=0, with_labels=False)
        plt.scatter(self.points[:, 0], self.points[:, 1], color=node_color, s=node_size, alpha=0.3)
        plt.scatter(self.points[p_indices][:, 0], self.points[p_indices][:, 1], color=node_color, s=node_size, alpha=1.0)
        ax.set_xlim([self.scope.min_x, self.scope.max_x])
        ax.set_ylim([self.scope.min_y, self.scope.max_y])
        plt.show()

    def draw_vorarea_graph(self, line_width=0.5, node_color='r', node_size=10):
        ax = plt.subplot()
        ax.set_aspect('equal')
        vor = self.net.vor
        for region in vor.regions:
            polygon = [self.net.vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color='gray', alpha=0.5)
        voronoi_plot_2d(vor, ax=ax, show_points=False, line_width=line_width)

        p_indices = list(self.iter_effective_point_indices())
        ax.scatter(self.points[:, 0], self.points[:, 1], color=node_color, s=node_size, alpha=0.3)
        ax.scatter(self.points[p_indices][:, 0], self.points[p_indices][:, 1], color=node_color, s=node_size, alpha=1.0)

        plt.xlim([self.scope.min_x, self.scope.max_x])
        plt.ylim([self.scope.min_y, self.scope.max_y])
        plt.show()
    