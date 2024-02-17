from typing import Callable, Tuple, List
from time import time

import numpy as np
import matplotlib.pyplot as plt

from .utils import get_distances, estimate_interaction
from .cooling import AdaptiveSchedule
from .scope import Scope
from .feature import Feature
from .mosaic import Mosaic

SIMULATED_TAG = "default"

class Pattern:
    """
    Spatial pattern and related mosaics of a cell type.
    
    Attributes
    ----------
    name: str, the name of the cell type (spatial pattern)
    density: float, density of cells (number/mm^2)
    natural_mosaics: List[Mosaic], natural mosaics
    simulated_mosaics: Dict[str, Mosaic], simulated mosaics orgnaized by tags
    features: Dict[str, Feature], features and corresponding names
    
    Methods
    ----------
    add_natural_mosaic()
    get_natural_mosaic()
    add_simulated_mosaic()
    get_simulated_mosaic()
    remove_mosaics()
    load_from_files()
    dump_to_files()
    set_density()
    estimate_density()
    set_feature()
    estimate_feature()
    set_feature_target()
    get_usable_features()
     
    """

    def __init__(self, name: str):
        self.name = name
        self.density = None
        self.natural_mosaics = []
        self.simulated_mosaics = {SIMULATED_TAG: []}
        self.features = {}

    def __str__(self):
        num_simulated_tags = sum(list(len(i) for i in (self.simulated_mosaics.values())))
        s1 = "Spatial pattern of %s, \n- Density: %s,\n- Natural mosaics: %d case(s),\n- Simulated mosaics: total %d case(s)"%(self.name, self.__get_density_str(), len(self.natural_mosaics), num_simulated_tags)
        if num_simulated_tags == 0:
            s2 = ",\n"
        else:
            s2 = "\n"+"\n".join(["   %d case(s) in tag '%s',"%(len(self.simulated_mosaics[tag]), tag) for tag in self.simulated_mosaics])
            s2 += "\n"
        feature_keys = list(self.features.keys())
        s3 = "- Features: %d"%len(feature_keys)
        if len(feature_keys) == 0:
            s3 += ".\n"
        else:
            s3 += "\n    \t Label\t| Has target probabilities\n"
            s3 +="\n".join([self.__get_feature_str(label, self.features[label]) 
                            for label in self.features])
            s3 +=".\n"
        return s1+s2+s3
    
    def __get_feature_str(self, label: str, feature: Feature):
        return "   \t %s\t| %r "%(label, feature.has_target())
    
    def __get_density_str(self):
        if self.density is None:
            return "Unknown"
        else:
            return "%.7f (cells/unit^2)"%(self.density)

    ########################################
    #
    # Load & Save
    #
    ########################################

    def add_natural_mosaic(self, mosaic: Mosaic) -> None:
        self.natural_mosaics.append(mosaic)

    def get_natural_mosaic(self, index: int, tag: str=SIMULATED_TAG) -> Mosaic:
        if index >= len(self.natural_mosaics):
            return None
        return self.natural_mosaics[index]
    
    def add_simulated_tag(self, tag: str):
        if tag in self.simulated_mosaics:
            ensure = input("clear mosaics in tag '%s', sure? [yes/no]"%tag)
            if ensure in ["Y", "y", "Yes", "YES", "yes"]:
                self.simulated_mosaics[tag] = []
        else:
            self.simulated_mosaics[tag] = []
            
    def get_simulated_tags(self, INCLUDEDEFAULT: bool=False) -> List[str]:
        tags = list(self.simulated_mosaics.keys())
        if not INCLUDEDEFAULT:
            tags.remove(SIMULATED_TAG)
        return tags

    def add_simulated_mosaic(self, mosaic: Mosaic, tag: str=SIMULATED_TAG) -> None:
        if tag in self.simulated_mosaics:
            self.simulated_mosaics[tag].append(mosaic)
        else:
            self.simulated_mosaics[tag] = [mosaic]

    def get_simulated_mosaic(self, index: int, tag: str=SIMULATED_TAG) -> Mosaic:
        if tag not in self.simulated_mosaics:
            return None
        if index >= len(self.simulated_mosaics[tag]):
            return None
        return self.simulated_mosaics[tag][index]
    
    def get_simulated_mosaics(self, tag: str=SIMULATED_TAG) -> List[Mosaic]:
        if tag not in self.simulated_mosaics:
            return None
        return self.simulated_mosaics[tag]

    def remove_mosaics(self, with_natural: bool=False, simulated_tag: str=SIMULATED_TAG) -> None:
        if with_natural:
            self.natural_mosaics = []
        if simulated_tag is None:
            self.simulated_mosaics = {SIMULATED_TAG: []}
        elif simulated_tag == SIMULATED_TAG:
            self.simulated_mosaics[SIMULATED_TAG] = []
        elif simulated_tag in self.simulated_mosaics:
            self.simulated_mosaics[simulated_tag] = None
        else:
            pass

    def load_from_files(self, point_fnames: list, scope: Scope, is_natural: bool=True, 
                        simulated_tag: str=SIMULATED_TAG) -> list:
        mosaics = []
        for fname in point_fnames:
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

    def dump_to_files(self, prefix: str, ext: str="points", is_natural: bool=True, separate: bool=False, 
                      simulated_tag: str=SIMULATED_TAG) -> None:
        if is_natural:
            mosaics = self.natural_mosaics
        else:
            mosaics = self.simulated_mosaics[simulated_tag]
        for i, mosaic in enumerate(mosaics):
            fname = "%s_%d.%s"%(prefix, i, ext)
            mosaic.save(fname, separate=separate)

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
    
    def set_feature(self, feature_label: str, feature: Feature) -> None:
        self.features[feature_label] = feature

    def estimate_feature(self, feature_label: str, natural: bool=True, simulated_tag: str=SIMULATED_TAG) -> np.ndarray:
        assert feature_label in self.features
        if natural:
            assert len(self.natural_mosaics) > 0
            values = self.__get_feature_values(self.natural_mosaics, feature_label)
        else:
            assert len(self.simulated_mosaics[simulated_tag]) > 0
            values = self.__get_feature_values(self.simulated_mosaics[simulated_tag], feature_label)
        hist = self.features[feature_label].get_hist(values).astype(float)
        hist /= hist.sum()
        return hist
    
    def set_feature_target(self, feature_label: str) -> None:
        if feature_label not in self.features:
            return None
        if len(self.natural_mosaics) == 0:
            return None
        probs = self.estimate_feature(feature_label=feature_label, natural=True)
        self.features[feature_label].set_target(probs)
        return probs

    def get_useable_features(self) -> list:
        features = []
        for key in self.features:
            if self.features[key].has_target():
                features.append(key)
        return features 
    
    def __get_feature_values(self, mosaics: list, feature_label: str) -> np.ndarray:
        return self.features[feature_label].extract_mosaics(mosaics)
    
    def __get_entropy(self, mosaics: list, feature_label: str) -> float:
        values = self.__get_feature_values(mosaics, feature_label)
        return self.features[feature_label].entropy(values)
    
    def evaluate(self, mosaics: list, features: list, SUM: bool=True) -> float:
        values = list(self.__get_entropy(mosaics, feature_label=f) for f in features)
        if SUM:
            return sum(values)
        return values

    ########################################
    #
    # Visualization Methods
    #
    ########################################

    def draw_feature_hist(self, feature_label: str, natural_color: str="skyblue", 
                          target_color: str="gray", simulated_color: str="red", simulated_tag: str=SIMULATED_TAG,
                          bar_args: dict={}, ax: plt.Axes=None) -> None:
        feature = self.features[feature_label]
        centers = feature.get_value_centers()
        width = centers[1]-centers[0]
        bar_args.setdefault("width", width)
        bar_args.setdefault("alpha", 0.5)
        bar_args.setdefault("align", "center")
        bar_args.pop("color", None)
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        if natural_color is not None and len(self.natural_mosaics) > 0:
            natural_probs = self.estimate_feature(feature_label, natural=True)
            my_ax.bar(centers, natural_probs, color=natural_color, label="natural", **bar_args)
        if target_color is not None and feature.has_target():
            my_ax.bar(centers, feature.target_probs, color=target_color, label="Target", **bar_args)
        if simulated_color is not None and len(self.simulated_mosaics[simulated_tag]) > 0:
            simulated_probs = self.estimate_feature(feature_label, natural=False, simulated_tag=simulated_tag)
            my_ax.bar(centers, simulated_probs, color=simulated_color, label="Simulated: %s"%simulated_tag, **bar_args)            
        my_ax.legend()
        my_ax.set_ylabel("Frequency")
        my_ax.set_title("Feature: %s"%feature_label)
        if ax is None:
            plt.show()

    def draw_feature_boxes(self, feature_label: str=None, 
                        draw_natural: bool=False, 
                        simulated_tags: list=None, box_args: dict={}, ax: plt.Axes=None) -> None:
        if feature_label is None:
            assert self.features[feature_label].has_target()
        if simulated_tags is None:
            simulated_tags = [i for i in self.simulated_mosaics.keys() if len(self.simulated_mosaics[i]) > 0]
        x_labels = []
        ys = []
        if draw_natural and len(self.natural_mosaics) > 0:
            if feature_label is None:
                values = list(self.__get_entropy([m], feature_label) for m in self.natural_mosaics)
            else:
                values = self.__get_feature_values(self.natural_mosaics, feature_label)
            x_labels.append("natural")
            ys.append(values)
        for tag in simulated_tags:
            if tag not in self.simulated_mosaics:
                continue
            if feature_label is None:
                values = list(self.__get_entropy([m], feature_label) for m in self.simulated_mosaics[tag])
            else:
                values = self.__get_feature_values(self.simulated_mosaics[tag], feature_label)
            x_labels.append("Simulated: %s"%tag)
            ys.append(values)
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        my_ax.boxplot(ys, vert=True, patch_artist=True, labels=x_labels, **box_args)
        if feature_label is None:
            my_ax.set_title("Entropy of features: %s"%feature_label)
        else:
            my_ax.set_title("Values of features: %s"%feature_label)
        if ax is None:
            plt.show()
        
    def draw_value_bars(self, value_method: Callable,
                        feature_colors: dict, 
                        draw_loss: bool=True,  
                        draw_natural: bool=False, 
                        simulated_tags: list=None, 
                        bar_args: dict={},
                        ax: plt.Axes=None) -> None:
        if simulated_tags is None:
            simulated_tags = [i for i in self.simulated_mosaics.keys() if len(self.simulated_mosaics[i]) > 0]
        bar_args.pop("color", None)
        features = list(feature_colors.keys())
        if draw_loss:
            for label in feature_colors:
                assert self.features[label].has_target()
        x_labels = []
        ys = []
        for i_feature, feature_label in enumerate(features):
            ys.append([])
            if draw_natural and len(self.natural_mosaics) > 0:
                if draw_loss:
                    values = list(self.__get_entropy([m], feature_label) for m in self.natural_mosaics)
                else:
                    values = self.__get_feature_values(self.natural_mosaics, feature_label)
                ys[i_feature].append(value_method(values))
                if i_feature == 0:
                    x_labels.append("natural")
            for tag in simulated_tags:
                if tag not in self.simulated_mosaics:
                    continue
                if i_feature == 0:
                    x_labels.append("Simulated: %s"%tag)
                if draw_loss:
                    values = list(self.__get_entropy([m], feature_label) for m in self.simulated_mosaics[tag])
                else:
                    values = self.__get_feature_values(self.simulated_mosaics[tag], feature_label)
                ys[i_feature].append(value_method(values))
        for i_feature in range(1, len(ys)):
            ys[i_feature] = np.array(ys[i_feature])
            ys[i_feature] += ys[i_feature-1]
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        bar_args.pop("color", None)
        for i_feature, feature_label in enumerate(features[::-1]):
            my_ax.bar(range(len(x_labels)), ys[len(features)-i_feature-1], label=feature_label, color=feature_colors[feature_label], **bar_args)
        my_ax.set_xticks(range(len(x_labels)))
        my_ax.set_xticklabels(x_labels)
        if draw_loss:
            my_ax.set_title("Entropy of features: %s"%", ".join(features))
        else:
            my_ax.set_title("Values of features: %s"%", ".join(features))
        my_ax.legend()
        if ax is None:
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
    
    def get_points_n(self, scope: Scope) -> int:
        if self.density is None:
            if len(self.natural_mosaics) == 0:
                raise Exception("")
            self.set_density(self.estimate_density())
        return int(self.density*scope.get_area())
    
    def new_mosaic_dmin(self, scope: Scope, dmin_mean: float, dmin_std:float, 
                    n: int=None, max_try: int=5000, FILLRANDOM: bool=False) -> Mosaic:
        if n is None or n <= 0:
            n = self.get_points_n(scope)
        points = []
        i_point = 0
        while i_point < n:
            pointsArray = np.array(points)
            if i_point > 0:
                fine = False
                i_try = 0
                while not fine:
                    i_try += 1
                    if i_try > max_try:
                        break
                    loc = scope.get_random_loc(1)[0]
                    distances = np.sqrt((pointsArray[:, 0]-loc[0])**2+
                                        (pointsArray[:, 1]-loc[1])**2)
                    dis = np.random.normal(loc=dmin_mean, scale=dmin_std)
                    if dis < distances.min():
                        # safe~
                        points.append(loc)
                        fine = True
            else:
                points.append(scope.get_random_loc(1)[0])
            i_point += 1
        if len(points) == n:
            return Mosaic(points=points, scope=scope)  
        else:
            if FILLRANDOM:
                points.extend(scope.get_random_loc(n-len(points)).tolist())
                return Mosaic(points=points, scope=scope)
            elif len(points) > 3:
                return Mosaic(points=points, scope=scope)
        return None

    def new_mosaic(self, scope: Scope, n: int=None) -> Mosaic:
        if n is None or n <= 0:
            n = self.get_points_n(scope)
        points = scope.get_random_loc(n)
        return Mosaic(points=points, scope=scope)

    def simulate(self, mosaic: Mosaic, interaction_func: Callable=None, 
                 features: list=None, schedule: AdaptiveSchedule=AdaptiveSchedule(), 
                 max_step: int=None, update_ratio: float=None,
                 save_prefix: str=None, save_step: int=1000, verbose: bool=True) -> Tuple[Mosaic, list]:
        if interaction_func is None:
            interaction_func = lambda x: 1.0 # accept all
            
        useable_features = self.get_useable_features()
        if features is None:
            features = useable_features
        else:
            features = list(set(features) & set(useable_features))

        def save(i_step: int, mosaic: Mosaic):
            if save_prefix is None:
                return
            if i_step%save_step != 0:
                return
            np.savetxt("%s_%d.points"%(save_prefix, i_step), mosaic.points)

        begin = time()
        losses = self.evaluate([mosaic], features=features, SUM=False)
        if verbose:
            print()
            print("Using features: %s"%str(features))
            print("Initial Loss: %f"%losses[0])
        if schedule is None:
            # Original PIPP
            if max_step is None:
                max_step = 20
            if update_ratio is None:
                update_ratio = 1.0  
            best_points = np.copy(mosaic.points)
            best_loss = losses[0]
            for i_step in range(max_step):
                new_mosaic, _ = self.__routine(mosaic, interaction_func=interaction_func, 
                                    update_ratio=update_ratio, non_neighbor=False)
                loss = self.evaluate([new_mosaic], features=features)
                if loss < best_loss:
                    best_loss = loss
                    best_points = np.copy(new_mosaic.points)
                if verbose:
                    print("Step #%d: loss = %f"%(i_step, loss))
                losses.append(loss)
                save(i_step, mosaic=new_mosaic)
            mosaic = Mosaic(points=best_points, scope=mosaic.scope)
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
                new_mosaic, n_relocated = self.__routine(mosaic, interaction_func=interaction_func, 
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
                        print("Step #%d: update %d cells, loss=%f, t=%f, delta=%f, accept_p=%f, is_update=%s"%(i_step, n_relocated, loss, accept_t, (loss-current_loss), accept_p, str(is_update)))
                    else:
                        print("Step #%d: update %d cells, loss=%f, t=%f, delta=%f, is_update=%s"%(i_step, n_relocated, loss, accept_t, (loss-current_loss), str(is_update)))
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
            print("Simulation End, use %f seconds, Final Loss: %f"%(end-begin, best_loss))
        if save_prefix is not None:
            np.savetxt("%s.losses"%save_prefix, losses)
        return mosaic, losses
        
    def __routine(self, mosaic: Mosaic, interaction_func: Callable, 
                  update_ratio: float=1.0, non_neighbor: bool=False) -> Tuple[Mosaic, int]:
        relocateds = []
        banned = set()
        N = mosaic.get_points_n()
        new_mosaic = Mosaic(points=np.copy(mosaic.points), scope=mosaic.scope)
        cell_mask = np.ones(N).astype(bool)
        for i_cell in np.random.choice(N, N, replace=False):
            if non_neighbor and i_cell in banned:
                continue
            if np.random.rand() >= update_ratio:
                continue
            if non_neighbor:
                banned.update(new_mosaic.find_neighbors(i_cell))

            cell_mask[:] = True
            cell_mask[i_cell] = False
            others = new_mosaic.points[cell_mask]
            relocate_loc = None
            while relocate_loc is None:
                new_loc = new_mosaic.scope.get_random_loc(1)[0]
                distances2others = get_distances(new_loc, pointsArray=others)
                prob = np.prod(interaction_func(distances2others))
                if np.random.rand() < prob:
                    relocate_loc = new_loc

            points = new_mosaic.points
            points[i_cell] = relocate_loc
            new_mosaic = Mosaic(points=points, scope=new_mosaic.scope)
            relocateds.append(i_cell)
            if non_neighbor:
                banned.update(new_mosaic.find_neighbors(i_cell))
        return new_mosaic, len(relocateds)





