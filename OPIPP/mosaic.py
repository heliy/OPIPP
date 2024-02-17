from typing import Iterator, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
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
        
    Attributes
    ----------
    points: np.ndarray or list
        locations of cells.

    scope: Scope
        the area of the mosaic.
        
    Methods
    ----------
    get_points_n()
    save()
    get_effective_filter()
    iter_effective_indices()
    get_effective_indices()
    get_boundary_indices()
    get_random_indices()
    get_vorareas()
    VDRI()
    find_neighbors()
    find_nearest_neighbor()
    get_nn_graph()
    get_nns()
    NNRI()
    get_distances()
    draw_points()
    draw_neighbors()
    draw_nn_graph()
    draw_vds()
    """
    def __init__(self, points: np.ndarray, scope: Scope=None, **attr):
        """
        Args:
            points (np.ndarray): locations of cells.
            scope (Scope): the area.
        """        
        super().__init__(**attr)
        self.points = np.array(points)
        assert len(self.points.shape) == 2
        if scope is None:
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)
            scope = Scope(min_x=int(min_x), max_x=int(max_x)+1, min_y=int(min_y), max_y=int(max_y)+1)
        self.set_scope(scope)       
        
    def set_scope(self, scope: Scope):
        # update triangulation domains
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
        """get the number of cells."""
        return self.points.shape[0]

    def save(self, fname: str, separate: bool=False) -> None:
        """save into local files.

        Args:
            fname (str): name of the local file.
            separate (bool, optional): True for save to two files seperated by axis. Defaults to False.
        """
        if separate:
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
        """Gets the effective filter"""        
        return self.effective_filter
    
    def iter_effective_indices(self) -> Iterator[int]:
        """Iters the index of effective points"""
        for p_index, is_effective in enumerate(self.get_effective_filter()):
            if is_effective:
                yield p_index

    def get_effective_indices(self) -> np.ndarray:
        """ Gets indices of effective points """
        effectives = np.arange(self.points.shape[0])[self.get_effective_filter()]
        return effectives
    
    def get_boundary_indices(self) -> np.ndarray:
        """ Gets indices of boundary points """
        in_surrounds = np.arange(self.points.shape[0])[(1-self.get_effective_filter()).astype(bool)]
        return in_surrounds
    
    def get_random_indices(self, n: int=1) -> np.ndarray:
        """ Gets indices of n random points """
        return np.random.choice(self.get_points_n(), size=min(n, self.get_points_n()), 
                                replace=False)
    
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

    def get_vorareas(self, indices: np.ndarray=None, effective_filter: bool=True) -> np.ndarray:
        """ Gets values of voronoi areas

        Args:
            indices (np.ndarray, optional): The indices of cells to query. 
               Without specification, it will return values of all cells. Defaults to None.
            effective_filter (bool, optional): If True, it will exclude boundary cells. Defaults to True.

        Returns:
            np.ndarray: VD areas.
        """        
        def vorarea_func(index):
            region = self.extended_vor.regions[self.extended_vor.point_region[index]]
            region = self.extended_vor.vertices[region]
            area = get_poly_area(region[:, 0], region[:, 1])
            return area
        return self.__get_features(feauture_func=vorarea_func, indices=indices, effective_filter=effective_filter)
    
    def VDRI(self) -> float:
        """the VDRI of the mosaic"""
        values = self.get_vorareas(indices=None, effective_filter=True)
        return np.mean(values)/np.std(values)        

    def find_neighbors(self, p_index: int, effective_only=False) -> list:
        """get neighbors of a given point

        Args:
            p_index (int): the index of the point
            effective_only (bool, optional): Only return neighbors that are effective if it is True. Defaults to False.

        Returns:
            list: the list of indices of neighbors
        """        
        neighbors = list(self.neighbors(p_index))
        if effective_only:
            return list(set(neighbors).difference(set(self.pnet.get_boundary_indices())))
        return neighbors

    def find_nearest_neighbor(self, p_index: int) -> tuple:
        """ Gets its NN neighbor and NN distance of a given cell """
        min_distance = self.scope.get_area()
        min_neighbor = -1
        for neighbor in self.find_neighbors(p_index):
            distance = self.__edge_length(neighbor, p_index)
            if distance < min_distance:
                min_distance = distance
                min_neighbor = neighbor
        return min_neighbor, min_distance

    def get_nn_graph(self, p_indices: list=None) -> nx.DiGraph:
        """ 
        Gets a directed subgraph containing all points and NN edges 
        """
        if p_indices is None:
            p_indices = self.nodes
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        for node in p_indices:
            nn_neighbor, _ = self.find_nearest_neighbor(node)
            graph.add_edge(node, nn_neighbor)
        return graph

    def get_nns(self, indices: np.ndarray=None, effective_filter: bool=True) -> np.ndarray:
        """ Gets values of NN distances

        Args:
            indices (np.ndarray, optional): The indices of cells to query. 
               Without specification, it will return values of all cells. Defaults to None.
            effective_filter (bool, optional): If True, it will exclude boundary cells. Defaults to True.

        Returns:
            np.ndarray: NN distances.
        """
        def nn_func(index):
            _, nn_distance = self.find_nearest_neighbor(index)
            return nn_distance
        return self.__get_features(feauture_func=nn_func, indices=indices, effective_filter=effective_filter)
    
    def NNRI(self) -> float:
        """Gets the NNRI of the mosaic"""
        values = self.get_nns(indices=None, effective_filter=True)
        return np.mean(values)/np.std(values)        

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

    def draw_points(self, highlights: list=None, nonhighlight_alpha: float=0.3, 
                    ax_grid: int=1, draw_plane_grid: bool=False, ax_scaled: bool=True,
                    point_args: dict={"color": "r", "s": 5}, ax: plt.Axes=None) -> None:
        """
        Args:
            highlights (list, optional): the list of cell indices to highlight. None for effective cells. Defaults to None.
            nonhighlight_alpha (float, optional): the alpha of nonhighlight cells. Defaults to 0.3.
            ax_grid (int, optional): the number of grid in the plane. Defaults to 1.
            draw_plane_grid (bool, optional): If draw grids of the plane. Defaults to False.
            ax_scaled (bool, optional): Tune the length of x/y sides. Defaults to True.
            point_args (_type_, optional): args delivered into matplotlib.scatter to customize attributes of points. Defaults to {"color": "r", "s": 5}.
            ax (plt.Axes, optional): Alternative axes for drawing. Defaults to None.
        """        
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        if ax_scaled:
            my_ax.axis('scaled')
        my_ax.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        my_ax.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=1.0, **point_args)
        my_ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, ax_grid+1))
        my_ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, ax_grid+1))
        my_ax.set_xlim([self.scope.min_x, self.scope.max_x])
        my_ax.set_ylim([self.scope.min_y, self.scope.max_y])
        my_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if draw_plane_grid:
            my_ax.grid(ax_grid)
        if ax is None:
            plt.show()

    def draw_neighbors(self, highlights: list=None, nonhighlight_alpha: float=0.3, 
                    ax_grid: int=1, draw_plane_grid: bool=False, ax_scaled: bool=True,
                    point_args: dict={"s": 5, "color": "r"}, 
                    edge_args: dict={"lw": 0.5, "color": "gray"}, 
                    ax: plt.Axes=None) -> None:
        """
        Args:
            highlights (list, optional): the list of cell indices to highlight. None for effective cells. Defaults to None.
            nonhighlight_alpha (float, optional): the alpha of nonhighlight cells. Defaults to 0.3.
            ax_grid (int, optional): the number of grid in the plane. Defaults to 1.
            draw_plane_grid (bool, optional): If draw grids of the plane. Defaults to False.
            ax_scaled (bool, optional): Tune the length of x/y sides. Defaults to True.
            point_args (_type_, optional): args delivered into matplotlib.scatter to customize attributes of points. Defaults to {"s": 5, "color": "r"}.
            edge_args (_type_, optional): args delivered into matplotlib.plot to customize attributes of edges. Defaults to {"lw": 0.5, "color": "gray"}.
            ax (plt.Axes, optional): Alternative axes for drawing. Defaults to None.
        """        
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        if ax_scaled:
            my_ax.axis('scaled')
        for edge in self.edges:
            my_ax.plot([self.points[edge[0], 0], self.points[edge[1], 0]],
                       [self.points[edge[0], 1], self.points[edge[1], 1]], **edge_args)
        my_ax.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        my_ax.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=1.0, **point_args)
        my_ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, ax_grid+1))
        my_ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, ax_grid+1))
        my_ax.set_xlim([self.scope.min_x, self.scope.max_x])
        my_ax.set_ylim([self.scope.min_y, self.scope.max_y])
        my_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if draw_plane_grid:
            my_ax.grid(ax_grid)
        if ax is None:
            plt.show()

    def draw_nn_graph(self, highlights: list=None, nonhighlight_alpha: float=0.3, 
                      ax_grid: int=1, draw_plane_grid: bool=False, ax_scaled: bool=True,
                      point_args: dict={"s": 5, "color": "r"}, 
                      network_args: dict={"edge_color": "k", "with_labels": False}, 
                      ax: plt.Axes=None) -> None:
        """
        Args:
            highlights (list, optional): the list of cell indices to highlight. None for effective cells. Defaults to None.
            nonhighlight_alpha (float, optional): the alpha of nonhighlight cells. Defaults to 0.3.
            ax_grid (int, optional): the number of grid in the plane. Defaults to 1.
            draw_plane_grid (bool, optional): If draw grids of the plane. Defaults to False.
            ax_scaled (bool, optional): Tune the length of x/y sides. Defaults to True.
            point_args (_type_, optional): args delivered into matplotlib.scatter to customize attributes of points. Defaults to {"s": 5, "color": "r"}.
            network_args (_type_, optional): args delivered into networkx.draw_networkx to customize attributes of network. Defaults to {"edge_color": "k", "with_labels": False}.
            ax (plt.Axes, optional): Alternative axes for drawing. Defaults to None.
        """        
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        if ax_scaled:
            my_ax.axis('scaled')
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        network_args.setdefault("node_size", 0)
        nn_graph = self.get_nn_graph(p_indices=highlights)
        highlight_alpha = point_args.pop("alpha", 1.0)
        nonhighlight_alpha *= highlight_alpha
        nx.draw_networkx(nn_graph, pos=self.points, ax=my_ax, **network_args)
        my_ax.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        my_ax.scatter(self.points[highlights][:, 0], self.points[highlights][:, 1], alpha=highlight_alpha, **point_args)
        my_ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, ax_grid+1))
        my_ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, ax_grid+1))
        my_ax.set_xlim([self.scope.min_x, self.scope.max_x])
        my_ax.set_ylim([self.scope.min_y, self.scope.max_y])
        my_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if draw_plane_grid:
            my_ax.grid(ax_grid)
        if ax is None:
            plt.show()

    def draw_vds(self, highlights: list=None, nonhighlight_alpha: float=0.3, 
                 ax_grid: int=1, ax_scaled: bool=True, 
                 plane_args: dict={"facecolor": "gray", "alpha": 0.3},
                 voronoi_args: dict={"show_points": False, "line_width": 0.5},
                 point_args: dict={"color": "r", "s": 10}, 
                 ax: plt.Axes=None) -> None:
        """
        Args:
            highlights (list, optional): the list of cell indices to highlight. None for effective cells. Defaults to None.
            nonhighlight_alpha (float, optional): the alpha of nonhighlight cells. Defaults to 0.3.
            ax_grid (int, optional): the number of grid in the plane. Defaults to 1.
            ax_scaled (bool, optional): Tune the length of x/y sides. Defaults to True.
            plane_args (_type_, optional): args delivered into matplotlib.PatchCollection to customize attributes of the plane. Defaults to {"facecolor": "gray", "alpha": 0.3}.
            voronoi_args (_type_, optional): args delivered into scipy.spatial.voronoi_plot_2d to customize attributes of VD areas. Defaults to {"show_points": False, "line_width": 0.5}.
            point_args (_type_, optional): args delivered into matplotlib.scatter to customize attributes of points. Defaults to {"color": "r", "s": 10}.
            ax (plt.Axes, optional): Alternative axes for drawing. Defaults to None.
        """        
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        if ax_scaled:
            my_ax.axis('scaled')
        if highlights is None:
            highlights = list(self.iter_effective_indices())
        rect = Rectangle((self.scope.min_x, self.scope.min_y), *self.scope.get_edges_len())
        pc = PatchCollection([rect], **plane_args)
        my_ax.add_collection(pc)
        vor = self.extended_vor
        voronoi_plot_2d(vor, ax=my_ax, **voronoi_args)

        p_indices = list(self.iter_effective_indices())
        my_ax.scatter(self.points[:, 0], self.points[:, 1], alpha=nonhighlight_alpha, **point_args)
        my_ax.scatter(self.points[p_indices][:, 0], self.points[p_indices][:, 1], alpha=1.0, **point_args)
        my_ax.set_xticks(np.linspace(self.scope.min_x, self.scope.max_x, ax_grid+1))
        my_ax.set_yticks(np.linspace(self.scope.min_y, self.scope.max_y, ax_grid+1))
        my_ax.set_xlim([self.scope.min_x, self.scope.max_x])
        my_ax.set_ylim([self.scope.min_y, self.scope.max_y])
        my_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        if ax is None:
            plt.show()
    
