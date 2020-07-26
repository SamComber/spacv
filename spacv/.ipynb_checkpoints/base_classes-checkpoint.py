import numpy as np
from .utils import convert_geoseries

class BaseSpatialCV():
    
    def __init__(
        self,
        random_state = None
    ):
        self.random_state = random_state
        self.buffer_radius = buffer_radius
        
    def split(self, X):
        X = convert_geoseries(X)

        minx, miny, maxx, maxy = X.total_bounds
        buffer_radius = self.buffer_radius
        
        if buffer_radius > maxx-minx or buffer_radius > maxy-miny:
            raise ValueError(
                "Buffer radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        for test_index, train_excluded in self._iter_test_indices(X):
            # Exclude training instances within buffered region of geometry
            train_index = np.setdiff1d(
                                np.union1d(
                                    indices,
                                    train_excluded
                                ), np.intersect1d(indices, train_excluded)
                            )
            test_index = indices[test_index]
                                
            yield train_index, test_index