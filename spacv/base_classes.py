import numpy as np

class BaseSpatialCV():
    
    def __init__(
        self,
        XYs = None,
        random_state = None
    ):
        self.XYs = XYs
        self.random_state = random_state
        
        minx, miny, maxx, maxy = self.XYs.total_bounds
        
        if buffer_radius > maxx-minx or buffer_radius > maxy-miny:
            raise ValueError(
                "Radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
    
    
    def split(self, X):

        num_samples = X.shape[0]

        indices = np.arange(num_samples)
    
        for test_index, train_excluded in self._iter_test_indices(X):

            train_index = np.setdiff1d(
                                np.union1d(
                                    indices,
                                    train_excluded
                                ), np.intersect1d(indices, train_excluded)
                            )

            test_index = indices[test_index]
            
            yield train_index, test_index