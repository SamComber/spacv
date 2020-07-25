import numpy as np

def geometry_to_2d(geometry):
    return np.array(list(map(lambda x : (x.x, x.y), geometry)))