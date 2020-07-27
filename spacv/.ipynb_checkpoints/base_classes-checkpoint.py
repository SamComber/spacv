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
        for test_indices, train_excluded in self._iter_test_indices(XYs):    
            # Exclude the training indices within buffer
            train_excluded = np.concatenate([test_indices, train_excluded])
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
            test_index = indices[test_indices]          
            yield train_index, test_index
            
            
    def _yield_test_indices(self, XYs, test_indices, buffer_radius):
        # Remove training points from dead zone buffer
        if buffer_radius > 0:    
            # Buffer grid and clip training instances
            candidate_deadzone = XYs.loc[~XYs.index.isin( test_indices )]
            grid_poly_buffer = grid.loc[[grid_id]].buffer(buffer_radius)
            deadzone_points = gpd.clip(candidate_deadzone, grid_poly_buffer)
            train_exclude = deadzone_points[~deadzone_points.index.isin(test_indices)].index.values
            return test_indices, train_exclude

        else:
            # Yield empty array because no training data removed in dead zone when buffer is zero
            _ = np.array([], dtype=np.int)
            return test_indices, _
