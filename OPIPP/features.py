import numpy as np

from .mosaic import Mosaic

def get_nns(mosaic: Mosaic) -> list:
    return mosaic.get_nns()

def get_vorareas(mosaic: Mosaic) -> list:
    return mosaic.get_vorareas()

def get_NNRI(mosaic: Mosaic) -> list:
    values = mosaic.get_nns()
    RI = np.mean(values)/np.std(values)
    return [RI]

def get_VDRI(mosaic: Mosaic) -> list:
    values = mosaic.get_vorareas()
    RI = np.mean(values)/np.std(values)
    return [RI]
