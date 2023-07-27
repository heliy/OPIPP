from typing import Callable, Tuple
from time import time

import numpy as np
import matplotlib.pyplot as plt

from .utils import get_distances, estimate_interaction
from .cooling import AdaptiveSchedule
from .scope import Scope
from .distribution import Distribution
from .mosaic import Mosaic

SIMULATED_TAG = "default"

class Pattern:
    """
    """

    def __init__(self, name: str):
        self.name = name
        self.density = None
        self.natural_mosaics = []
        self.simulated_mosaics = {SIMULATED_TAG: []}
        self.distributions = {}
        self.methods = {}

    def __str__(self):
        num_simulated_tags = sum(list(len(i) for i in (self.simulated_mosaics.values())))
        s1 = "Spatial pattern of %s, \n- Density: %s,\n- Natural mosaics: %d samples,\n- Simulated mosaics: total %d samples"%(self.name, self.__get_density_str(), len(self.natural_mosaics), num_simulated_tags)
        if num_simulated_tags == 0:
            s2 = ",\n"
        else:
            s2 = "\n"+"\n".join(["   %d samples in tag '%s',"%(len(self.simulated_mosaics[tag]), tag) for tag in self.simulated_mosaics])
            s2 += "\n"
        feature_keys = list(set(list(self.distributions.keys())+list(self.methods.keys())))
        s3 = "- Features: %d"%len(feature_keys)
        if len(feature_keys) == 0:
            s3 += ".\n"
        else:
            s3 += "\n    \t Label\t| Has Method \t| Has Distribution \t| Has Target Probs \n"+"\n".join([self.__get_feature_str(key) for key in feature_keys])
            s3 +=".\n"
        return s1+s2+s3
    
    def __get_feature_str(self, feature_label):
        return "   \t %s\t| %r \t| %r \t| %r "%(feature_label, feature_label in self.methods, feature_label in self.distributions, feature_label in self.distributions and self.distributions[feature_label].has_target())
    
    def __get_density_str(self):
        if self.density is None:
            return "Unknown"
        else:
            return "%.7f (cells/μm^2)"%(self.density)

    def clear_mosaics(self, with_natural: bool=False) -> None:
        self.simulated_mosaics = {SIMULATED_TAG: []}
        if with_natural:
            self.natural_mosaics = []

    ########################################
    #
    # Load & Save
    #
    ########################################

    def add_natural_mosaic(self, mosaic: Mosaic) -> None:
        self.natural_mosaics.append(mosaic)

    def add_simulated_mosaic(self, mosaic: Mosaic, tag: str=SIMULATED_TAG) -> None:
        if tag in self.simulated_mosaics:
            self.simulated_mosaics[tag].append(mosaic)
        else:
            self.simulated_mosaics[tag] = [mosaic]

    def load_from_files(self, fnames: list, scope: Scope, is_natural: bool=True, 
                        simulated_tag: str=SIMULATED_TAG) -> list:
        mosaics = []
        for fname in fnames:
            if fname[-4:] == ".npy":
                points = np.load(fname)
            else:
                try:
                    points = np.loadtxt(fname)
                except:
                    raise Exception("Unknown file type: %s"%fname)
            mosaic = Mosaic(points=points, scope=scope)
            if is_natural:
                self.add_natural_mosaic(mosaic)
            else:
                self.add_simulated_mosaic(mosaic, tag=simulated_tag)
            mosaics.append(mosaic)
        return mosaics

    def dump_to_files(self, prefix: str, ext: str="points", is_natural: bool=True, split: bool=False, 
                      simulated_tag: str=SIMULATED_TAG) -> None:
        if is_natural:
            mosaics = self.natural_mosaics
        else:
            mosaics = self.simulated_mosaics[simulated_tag]
        for i, mosaic in enumerate(mosaics):
            fname = "%s-%d.%s"%(prefix, i, ext)
            mosaic.save(fname, split=split)

    ########################################
    #
    # Features and Density estimations
    #
    ########################################

    def set_density(self, density: float) -> None:
        self.density = density

    def estimate_density(self) -> float:
        if len(self.natural_mosaics) == 0:
            return -1
        total_area = 0.
        total_num = 0.
        for mosaic in self.natural_mosaics:
            total_num += mosaic.get_points_n()
            total_area += mosaic.scope.get_area()
        return float(total_num / total_area)
    
    def set_feature(self, feature_label: str, distribution: Distribution, feature_method: Callable=None) -> None:
        self.distributions[feature_label] = distribution
        self.methods[feature_label] = feature_method

    def estimate_feature(self, feature_label: str, natural: bool=True, simulated_tag: str=SIMULATED_TAG) -> np.ndarray:
        assert feature_label in self.methods and self.methods[feature_label] is not None
        if natural:
            assert len(self.natural_mosaics) > 0
            values = np.concatenate(list(self.methods[feature_label](mosaic) for mosaic 
                                         in self.natural_mosaics)).flatten()
        else:
            assert len(self.simulated_mosaics) > 0
            values = np.concatenate(list(self.methods[feature_label](mosaic) for mosaic 
                                         in self.simulated_mosaics[simulated_tag])).flatten()
        hist = self.distributions[feature_label].get_hist(values).astype(float)
        hist /= hist.sum()
        return hist

    def get_useable_features(self) -> list:
        features = []
        for key in self.distributions:
            if self.distributions[key].has_target() and self.methods[key] is not None:
                features.append(key)
        return features 
    
    def __get_feature_values(self, mosaics: list, feature_label: str) -> np.ndarray:
        method = self.methods[feature_label]
        values = np.concatenate(list(method(mosaic) for mosaic in mosaics))
        return values
    
    def __get_kl(self, mosaics: list, feature_label: str) -> float:
        values = self.__get_feature_values(mosaics=mosaics, feature_label=feature_label)
        return self.distributions[feature_label].KL(values)
    
    def evaluate(self, mosaics: list, features: list) -> float:
        return sum(list(self.__get_kl(mosaics, feature_label=f) for f in features))

    ########################################
    #
    # Visualization Methods
    #
    ########################################

    def draw_feature_hist(self, feature_label: str, natural_color: str="skyblue", 
                          target_color: str="gray", simulated_color: str="red", simulated_tag: str=SIMULATED_TAG,
                          **args) -> None:
        distribution = self.distributions[feature_label]
        centers = distribution.get_value_centers()
        width = centers[1]-centers[0]
        ax = plt.subplot()
        if natural_color is not None and len(self.natural_mosaics) > 0:
            natural_probs = self.estimate_feature(feature_label, natural=True)
            plt.bar(centers, natural_probs, color=natural_color, label="natural", align="center", alpha=0.4, width=width, **args)
        if target_color is not None and distribution.has_target():
            plt.bar(centers, distribution.target_probs, color=target_color, label="Target", align="center", alpha=0.4, width=width, **args)
        if simulated_color is not None and len(self.simulated_mosaics[simulated_tag]) > 0:
            simulated_probs = self.estimate_feature(feature_label, natural=False, simulated_tag=simulated_tag)
            plt.bar(centers, simulated_probs, color=simulated_color, label="Simulated: %s"%simulated_tag, align="center", alpha=0.4, width=width, **args)            
        plt.legend()
        ax.set_ylabel("Frequency")
        ax.set_title("Feature: %s"%feature_label)
        plt.show()

    def draw_values_box(self, draw_loss: bool, feature_label: str, draw_natural: bool=False, 
                        simulated_tags: list=[SIMULATED_TAG], **args) -> None:
        if draw_loss:
            assert self.distributions[feature_label].has_target()
        x_labels = []
        ys = []
        if draw_natural and len(self.natural_mosaics) > 0:
            if draw_loss:
                values = list(self.__get_kl([m], feature_label) for m in self.natural_mosaics)
            else:
                values = self.__get_feature_values(self.natural_mosaics, feature_label)
            x_labels.append("natural")
            ys.append(values)
        for tag in simulated_tags:
            if tag not in self.simulated_mosaics:
                continue
            if draw_loss:
                values = list(self.__get_kl([m], feature_label) for m in self.simulated_mosaics[tag])
            else:
                values = self.__get_feature_values(self.simulated_mosaics[tag], feature_label)
            x_labels.append("Simulated: %s"%tag)
            ys.append(values)
        ax = plt.subplot()
        plt.boxplot(ys, vert=True, patch_artist=True, labels=x_labels, **args)
        if draw_loss:
            ax.set_title("KL divergency of features: %s"%feature_label)
        else:
            ax.set_title("Values of features: %s"%feature_label)
        plt.show()
        
    def draw_value_bars(self, draw_loss: bool, feature_colors: dict, method: Callable=np.mean, 
                        draw_natural: bool=False, simulated_tags: list=[SIMULATED_TAG], **args) -> None:
        features = list(feature_colors.keys())
        if draw_loss:
            for label in feature_colors:
                assert self.distributions[label].has_target()
        x_labels = []
        ys = []
        for i_feature, feature_label in enumerate(features):
            ys.append([])
            if draw_natural and len(self.natural_mosaics) > 0:
                if draw_loss:
                    values = list(self.__get_kl([m], feature_label) for m in self.natural_mosaics)
                else:
                    values = self.__get_feature_values(self.natural_mosaics, feature_label)
                ys[i_feature].append(method(values))
                if i_feature == 0:
                    x_labels.append("natural")
            for tag in simulated_tags:
                if tag not in self.simulated_mosaics:
                    continue
                if i_feature == 0:
                    x_labels.append("Simulated: %s"%tag)
                if draw_loss:
                    values = list(self.__get_kl([m], feature_label) for m in self.simulated_mosaics[tag])
                else:
                    values = self.__get_feature_values(self.simulated_mosaics[tag], feature_label)
                ys[i_feature].append(method(values))
        for i_feature in range(1, len(ys)):
            ys[i_feature] = np.array(ys[i_feature])
            ys[i_feature] += ys[i_feature-1]
        ax = plt.subplot()
        for i_feature, feature_label in enumerate(features[::-1]):
            plt.bar(range(len(x_labels)), ys[len(features)-i_feature-1], label=feature_label, **args)
        print(x_labels)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        if draw_loss:
            ax.set_title("KL divergency of features: %s"%", ".join(features))
        else:
            ax.set_title("Values of features: %s"%", ".join(features))
        plt.legend()
        plt.show()

    ########################################
    #
    # Mosaic Generation Methods
    #
    ########################################

    def get_interaction_func(self, values: list, axis: np.ndarray=None, p0: list=[20, 1], 
                             delta: float=3., draw: bool=False) -> Callable:
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
                if len(self.natural_mosaics) == 0:
                    raise Exception("")
                self.set_density(self.estimate_density())
            n = int(self.density*scope.get_area())
        points = scope.get_random_loc(n)
        return Mosaic(points=points, scope=scope)

    def simulate(self, mosaic: Mosaic, interaction_func: Callable=None, 
                 features: list=None, schedule: AdaptiveSchedule=AdaptiveSchedule(), 
                 max_step: int=None, update_ratio: float=None,
                 save_prefix: str=None, save_step: int=1, verbose: bool=True) -> Tuple[Mosaic, list]:
        if interaction_func is None:
            interaction_func = lambda x: 1.0 # accept all
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





