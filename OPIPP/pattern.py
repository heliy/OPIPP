from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from .scope import Scope
from .distribution import Distribution
from .mosaic import Mosaic

class Pattern:
    """
    """

    def __init__(self, name: str):
        self.name = name
        self.density = None
        self.nature_mosaics = []
        self.simulated_mosaics = []
        self.distributions = {}
        self.methods = {}

    def clear_mosaics(self, with_nature: bool=False):
        self.simulated_mosaics = []
        if with_nature:
            self.nature_mosaics = []

    ########################################
    #
    # Load & Save
    #
    ########################################

    def add_nature_mosaic(self, mosaic: Mosaic):
        self.nature_mosaics.append(mosaic)

    def add_simulated_mosaic(self, mosaic: Mosaic):
        self.simulated_mosaics.append(mosaic)

    def load_from_files(self, fnames: list, scope: Scope, is_nature: bool=True):
        for fname in fnames:
            if fname[-4:] == ".txt":
                points = np.loadtxt(fname)
            elif fname[-4:] == ".npy":
                points = np.load(fname)
            else:
                raise Exception("Unknown file type: %s"%fname)
            mosaic = Mosaic(points=points, scope=scope)
            if is_nature:
                self.add_nature_mosaic(mosaic)
            else:
                self.add_simulated_mosaic(mosaic)

    def dump_to_files(self, prefix: str):
        pass

    ########################################
    #
    # Features and Density estimations
    #
    ########################################

    def set_density(self, density: float):
        self.density = density

    def estimate_density(self):
        if len(self.nature_mosaics) == 0:
            return -1
        total_area = 0.
        total_num = 0.
        for mosaic in self.nature_mosaics:
            total_num += mosaic.get_points_n()
            total_area += mosaic.scope.get_area()
        return total_num / total_area
    
    def set_feature(self, feature_label: str, distribution: Distribution, feature_method: Callable=None):
        self.distributions[feature_label] = distribution
        self.methods[feature_label] = feature_method

    def estimate_feature(self, feature_label: str):
        assert feature_label in self.methods and self.methods[feature_label] is not None
        assert len(self.nature_mosaics) > 0
        values = np.concatenate(list(self.methods[feature_label](mosaic) for mosaic in self.nature_mosaics)).flatten()
        hist = self.distributions[feature_label].get_hist(values)
        self.distributions[feature_label].set_target(hist)

    def get_usabel_features(self) -> list:
        features = []
        for key in self.distributions:
            if self.distributions[key].has_target() and self.methods[key] is not None:
                features.append(key)
        return features 

    ########################################
    #
    # Visualization Methods
    #
    ########################################

    ########################################
    #
    # Mosaic Generation Methods
    #
    ########################################





