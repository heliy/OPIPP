from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def get_poly_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))

def get_distances(loc: Tuple[float, float], pointsArray: np.ndarray) -> np.ndarray:
    return np.sqrt((pointsArray[:, 0]-loc[0])**2+(pointsArray[:, 1]-loc[1])**2)

def estimate_interaction(gammas: np.ndarray, axis: np.ndarray, p0: list=[20, 1], delta: float=3, 
                         append_n: int=10, draw: bool=False) -> list:
    if isinstance(gammas, np.ndarray):
        gammas = gammas.tolist()
    gammas.extend([1]*append_n)
    gammas = np.array(gammas, dtype=float)
    x = np.arange(len(gammas))*(axis[1]-axis[0])+axis[0]
    if delta is None:
        assert len(p0) == 3
        def func(x, delta, phi, alpha):
            ys = 1-np.exp(-((x-delta)/phi)**alpha)
            ys[x <= delta] = 0
            return ys
    else:
        assert len(p0) == 2
        def func(x, phi, alpha):
            ys = 1-np.exp(-((x-delta)/phi)**alpha)
            ys[x <= delta] = 0
            return ys
    popt, pcov = curve_fit(func, x, gammas, p0=p0)
    if draw:
        plt.plot(x, gammas, "--", color='r', label="non-parameteric")
        plt.plot(x, func(x, *popt), color='k', label="paramteric")
        plt.legend()
        plt.show()
    if delta is None:
        return list(popt)
    else:
        return [delta]+list(popt)

