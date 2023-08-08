import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Union
from .mosaic import Mosaic

class Distribution:
    """ 
    Probability distribution of feature values
        
    Parameters
    ----------
    max_value: float
        The maximum value for bins in histogram.

    n_bin: int
        The number of bins in histogram.

    min_value: float, optional(default=0.)
        The minimum value for bins in histogram.

    Methods
    -------
    set_target()
        Sets the target histogram.

    get_prob()
        Gets the probability of a given value.

    get_hist()
        Gets the histrogram of probablities of given values.

    KL()
        Calculates the KL divergency between probabllities of given values and the target.
    
    sample_values()
        Generates random values following the target probabilities.
    
    sample_ri()
        Calculate the Regularity Index of values generated randomly.
    """
    def __init__(self, method: Union[Callable, str], max_value: float, n_bin: int=1, min_value: float=0.):
        if type(method) == str and hasattr(Mosaic, method):
            if callable(getattr(Mosaic, method)):
                self.method = lambda mosaic: getattr(mosaic, method)()
            else:
                self.method = lambda mosaic: getattr(mosaic, method)
        else:
            self.method = method

        assert max_value > min_value
        self.max_value = max_value
        self.min_value = min_value
        self.n_bin = n_bin
        self.step = (max_value-min_value)/n_bin
        self.target_probs = None

    def __get_index(self, value: float) -> int:
        return min(int((value-self.min_value)/self.step), self.n_bin)
    
    def set_target(self, target_probs: list) -> None:
        assert len(target_probs) == self.n_bin+1
        self.target_probs = np.copy(target_probs).astype(float)
        self.target_probs[self.target_probs<=0] = 1e-5 
        self.target_probs /= self.target_probs.sum()

    def has_target(self) -> bool:
        return self.target_probs is not None

    def extract_mosaic(self, mosaic: Mosaic) -> np.ndarray:
        return np.array([self.method(mosaic)]).flatten()
    
    def extract_mosaics(self, mosaics: list) -> list:
        values = np.concatenate(list(self.extract_mosaic(mosaic) for mosaic in mosaics))
        return values.flatten()
    
    def get_prob(self, value: float) -> float:
        return self.target_probs[self.__get_index(value)]

    def get_value_centers(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, num=self.n_bin+1)+self.step/2

    def get_hist(self, values: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=self.n_bin, range=(self.min_value, self.max_value))
        hist = np.concatenate((hist, [len(values)-sum(hist)])).astype(float)
        return hist

    def KL(self, values: np.ndarray) -> float:
        hist = self.get_hist(values)
        probs = hist/hist.sum()
        kl = np.sum([pk * np.log(pk / qk) for pk, qk in zip(probs, self.target_probs) if qk > 0 and pk > 0])
        return kl
    
    def entropy(self, values: np.ndarray) -> float:
        return self.KL(values=values)

    def sample_values(self, n: int=1) -> np.ndarray:
        centers = np.random.choice(self.get_value_centers(), p=self.target_probs, size=n)
        return (np.random.rand(n)-0.5)*self.step+centers

    def sample_RI(self, n_sample: int=50000) -> float:
        values = self.sample_values(n_sample)
        return values.mean()/values.std()
    
    def view(self, bar_args: dict={}, ax: plt.Axes=None) -> None:
        if not self.has_target():
            return
        centers = self.get_value_centers()
        width = centers[1]-centers[0]
        bar_args.setdefault("width", width)
        bar_args.setdefault("alpha", 0.5)
        bar_args.setdefault("color", "gray")
        bar_args.setdefault("align", "center")
        if ax is None:
            my_ax = plt.subplot()
        else:
            my_ax = ax
        my_ax.bar(centers, self.target_probs, **bar_args)
        my_ax.set_ylabel("Frequency")
        if ax is None:
            plt.show()
