import numpy as np
from .utils import convert_geoseries

class BaseSpatialCV():
    
    def __init__(
        self,
        random_state = None
    ):
        self.random_state = random_state
        self.buffer_radius = buffer_radius
        
    def split(self, XYs, y=None, groups=None):
        XYs = convert_geoseries(XYs)

        minx, miny, maxx, maxy = XYs.total_bounds
        buffer_radius = self.buffer_radius
        
        if buffer_radius > maxx-minx or buffer_radius > maxy-miny:
            raise ValueError(
                "buffer_radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
            
        num_samples = XYs.shape[0]
        indices = np.arange(num_samples)
        for test_index, train_excluded in self._iter_test_indices(XYs):         
            # Exclude training instances within buffered region of geometry
            train_index = np.setdiff1d(
                                np.union1d(
                                    indices,
                                    train_excluded
                                ), np.intersect1d(indices, train_excluded)
                            )
            if len(train_index) < 1:
                raise ValueError(
                    "Training set is empty. Try lowering buffer_radius to include more training instances."
                )
            test_index = indices[test_index]          
            yield train_index, test_index