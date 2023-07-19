from typing import Callable
from time import time

import numpy as np
import matplotlib.pyplot as plt

from .utils import get_distances, estimate_interaction
from .cooling import AdaptiveSchedule
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
            if fname[-4:] == ".npy":
                points = np.load(fname)
            else:
                try:
                    points = np.loadtxt(fname)
                except:
                    raise Exception("Unknown file type: %s"%fname)
            mosaic = Mosaic(points=points, scope=scope)
            if is_nature:
                self.add_nature_mosaic(mosaic)
            else:
                self.add_simulated_mosaic(mosaic)

    def dump_to_files(self, prefix: str, ext: str="points", is_nature: bool=True, split: bool=False):
        if is_nature:
            mosaics = self.nature_mosaics
        else:
            mosaics = self.simulated_mosaics
        for i, mosaic in enumerate(mosaics):
            fname = "%s-%d.%s"%(prefix, i, ext)
            mosaic.save(fname, split=split)

    ########################################
    #
    # Features and Density estimations
    #
    ########################################

    def set_density(self, density: float):
        self.density = density

    def estimate_density(self) -> float:
        if len(self.nature_mosaics) == 0:
            return -1
        total_area = 0.
        total_num = 0.
        for mosaic in self.nature_mosaics:
            total_num += mosaic.get_points_n()
            total_area += mosaic.scope.get_area()
        return float(total_num / total_area)
    
    def set_feature(self, feature_label: str, distribution: Distribution, feature_method: Callable=None):
        self.distributions[feature_label] = distribution
        self.methods[feature_label] = feature_method

    def estimate_feature(self, feature_label: str, nature: bool=True) -> np.ndarray:
        assert feature_label in self.methods and self.methods[feature_label] is not None
        if nature:
            assert len(self.nature_mosaics) > 0
            values = np.concatenate(list(self.methods[feature_label](mosaic) for mosaic in self.nature_mosaics)).flatten()
        else:
            assert len(self.simulated_mosaics) > 0
            values = np.concatenate(list(self.methods[feature_label](mosaic) for mosaic in self.simulated_mosaics)).flatten()
        hist = self.distributions[feature_label].get_hist(values).astype(float)
        hist /= hist.sum()
        return hist

    def get_useable_features(self) -> list:
        features = []
        for key in self.distributions:
            if self.distributions[key].has_target() and self.methods[key] is not None:
                features.append(key)
        return features 
    
    def __get_feature_values(self, mosaics: list, feature_label: str):
        method = self.methods[feature_label]
        values = np.concatenate(list(method(mosaic) for mosaic in mosaics))
        return values
    
    def evaluate(self, mosaics: list, features: list) -> float:
        loss = 0
        for feature in features:
            values = self.__get_feature_values(mosaics=mosaics, feature_label=feature)
            loss += self.distributions[feature].KL(values)
        return loss

    ########################################
    #
    # Visualization Methods
    #
    ########################################

    def draw_feature_hist(self, feature_label: str, nature_color: str="skyblue", 
                          target_color: str="gray", simulated_color: str="red"):
        distribution = self.distributions[feature_label]
        centers = distribution.get_value_centers()
        width = centers[1]-centers[0]
        ax = plt.subplot()
        if nature_color is not None and len(self.nature_mosaics) > 0:
            nature_probs = self.estimate_feature(feature_label, nature=True)
            plt.bar(centers, nature_probs, color=nature_color, label="Nature", align="center", alpha=0.4, width=width)
        if target_color is not None and distribution.has_target():
            plt.bar(centers, distribution.target_probs, color=target_color, label="Target", align="center", alpha=0.4, width=width)
        if simulated_color is not None and len(self.simulated_mosaics) > 0:
            simulated_probs = self.estimate_feature(feature_label, nature=False)
            plt.bar(centers, simulated_probs, color=simulated_color, label="Simulated", align="center", alpha=0.4, width=width)            
        plt.legend()
        ax.set_ylabel("Frequency")
        ax.set_title("Feature: %s"%feature_label)
        plt.show()

    def draw_feature_boxes(self, feature_label: str, nature_color: str="skyblue", 
                          target_color: str="gray", simulated_color: str="red"):
        pass
        

    ########################################
    #
    # Mosaic Generation Methods
    #
    ########################################

    def get_interaction_func(self, values: list, axis: np.ndarray=None, p0: list=[20, 1], delta: float=3., draw: bool=False) -> Callable:
        if axis is None:
            # estimated parameters from values
            theta, phi, alpha = tuple(values)
        else:
            # fit the interaction parameters with given values and axis
            assert p0 is not None
            theta, phi, alpha = tuple(estimate_interaction(gammas=values, axis=axis, p0=p0, delta=delta, draw=draw))

        def interaction_func(distances):
            probs = np.copy(distances) - theta
            probs[probs < 0] = 0
            probs = 1-np.exp(-((probs/phi)**alpha))
            return probs
        return interaction_func

    def new_mosaic(self, scope: Scope, n: int=None) -> Mosaic:
        if n is None or n <= 0:
            if self.density is None:
                if len(self.nature_mosaics) == 0:
                    raise Exception("")
                self.set_density(self.estimate_density())
            n = int(self.density*scope.get_area())
        points = scope.get_random_loc(n)
        return Mosaic(points=points, scope=scope)

    def simulate(self, mosaic: Mosaic, interaction_func: Callable=None, 
                 features: list=None, schedule: AdaptiveSchedule=AdaptiveSchedule(), 
                 max_step: int=None, update_ratio: float=None,
                 save_prefix: str=None, save_step: int=1, verbose: bool=True):
        if interaction_func is None:
            interaction_func = self.get_interaction_func()
        useable_features = self.get_useable_features()
        if features is None:
            features = useable_features
        else:
            features = list(set(features) & set(useable_features))
            print("Using features: %s"%str(features))

        def save(i_step: int, mosaic: Mosaic):
            if save_prefix is None:
                return
            if i_step%save_step != 0:
                return
            np.savetxt("%s_%d.points"%(save_prefix, i_step), mosaic.points)

        begin = time()
        losses = [self.evaluate([mosaic], features=features)]
        if verbose:
            print()
        if schedule is None:
            # Original PIPP
            if max_step is None:
                max_step = 20
            if update_ratio is None:
                update_ratio = 1.0  
            for i_step in range(max_step):
                mosaic = self.__routine(mosaic, interaction_func=interaction_func, 
                                    update_ratio=update_ratio, non_neighbor=False)
                loss = self.evaluate([mosaic], features=features)
                if verbose:
                    print("Step #%d: loss = %f"%(i_step, loss))
                losses.append(loss)
                save(i_step, mosaic=mosaic)
        else:
            # Optimization-based PIPP
            if update_ratio is None:
                update_ratio = 0.01
            schedule.init()
            best_points = np.copy(mosaic.points)
            current_loss = losses[0]
            best_loss = losses[0]
            i_step = 0
            while schedule.has_next():
                if max_step is not None and i_step >= max_step:
                    break
                if best_loss <= 1e-5:
                    # loss is small enough
                    break
                bak_points = np.copy(mosaic.points)
                new_mosaic = self.__routine(mosaic, interaction_func=interaction_func, 
                                    update_ratio=update_ratio, non_neighbor=True)
                loss = self.evaluate([new_mosaic], features=features)
                if loss < best_loss:
                    best_loss = current_loss
                    best_points = np.copy(new_mosaic.points)
                is_update = True
                accept_t = schedule.next(loss)
                if loss > current_loss:
                    accept_p = np.exp(-(loss-current_loss)/accept_t)
                    if np.random.rand() > accept_p:
                        is_update = False
                if verbose:
                    if loss > current_loss:
                        print("Step #%d: loss=%f, t=%f, delta=%f, accept_p=%f, is_update=%s"%(i_step, loss, accept_t, (loss-current_loss), accept_p, str(is_update)))
                    else:
                        print("Step #%d: loss=%f, t=%f, delta=%f, is_update=%s"%(i_step, loss, accept_t, (loss-current_loss), str(is_update)))
                if is_update:
                    current_loss = loss
                    mosaic = new_mosaic
                else:
                    mosaic = Mosaic(bak_points, scope=mosaic.scope)
                losses.append(loss)
                save(i_step, mosaic=mosaic)
                i_step += 1
            mosaic = Mosaic(points=best_points, scope=mosaic.scope)
        if verbose:
            end = time()
            print("Simulation End, use %f seconds"%(end-begin))
        if save_prefix is not None:
            np.savetxt("%s.losses"%save_prefix, losses)
        return mosaic, losses
        
    def __routine(self, mosaic: Mosaic, interaction_func: Callable, 
                  update_ratio: float=1.0, non_neighbor: bool=False) -> Mosaic:
        relocateds = []
        banned = set()
        N = mosaic.get_points_n()
        cell_mask = np.ones(N).astype(bool)
        for i_cell in np.random.choice(N, N, replace=False):
            if non_neighbor and i_cell in banned:
                continue
            if np.random.rand() >= update_ratio:
                continue
            if non_neighbor:
                banned.update(mosaic.find_neighbors(i_cell))

            cell_mask[:] = True
            cell_mask[i_cell] = False
            others = mosaic.points[cell_mask]
            relocate_loc = None
            while relocate_loc is None:
                new_loc = mosaic.scope.get_random_loc(1)[0]
                distances2others = get_distances(new_loc, pointsArray=others)
                prob = np.prod(interaction_func(distances2others))
                if np.random.rand() < prob:
                    relocate_loc = new_loc

            points = mosaic.points
            points[i_cell] = relocate_loc
            mosaic = Mosaic(points=points, scope=mosaic.scope)
            relocateds.append(i_cell)
            if non_neighbor:
                banned.update(mosaic.find_neighbors(i_cell))
        return mosaic





