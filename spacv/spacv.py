import numpy as np
import geopandas as gpd
from sklearn.metrics import make_scorer
from .base_classes import BaseSpatialCV
from .grid_builder import construct_blocks, assign_pt_to_grid
from .utils import geometry_to_2d, convert_geodataframe

class HBLOCK(BaseSpatialCV):
    """
    H-Blocking spatial cross-validator.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    tiles_x : integer
    
    
    
    
    
    """
    def __init__(
        self,
        tiles_x=5,
        tiles_y=5,
        method='unique',
        buffer_radius=0,
        shuffle=False,
        direction='diagonal',
        n_groups=5,
        data=None,
        n_sims=10
    ):        
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.method = method
        self.buffer_radius = buffer_radius
        self.shuffle = shuffle
        self.direction = direction
        self.n_groups = n_groups
        self.data = data
        self.n_sims = n_sims
        
    def _iter_test_indices(self, XYs):
                
        # Define grid type used in CV procedure
        grid = construct_blocks(XYs, 
                      tiles_x = self.tiles_x, 
                      tiles_y = self.tiles_y, 
                      method = self.method, 
                      direction = self.direction, 
                      n_groups = self.n_groups,
                      data = self.data, 
                      n_sims = self.n_sims)
    
        # Convert to GDF to use Geopandas functions
        XYs = gpd.GeoDataFrame(({'geometry':XYs}))
        
        # Assign pts to grids
        XYs = assign_pt_to_grid(XYs, grid)
        grid_ids = np.unique(grid.grid_id)
        
        # Shuffle grid order 
        if self.shuffle:
            check_random_state(self.random_state).shuffle(grid_ids)

        # Yield test indices and optionally training indices within buffer
        for grid_id in grid_ids:
            test_points = XYs.loc[XYs['grid_id'] == grid_id ].index.values

            # Remove empty grids
            if len(test_points) < 1:
                continue
            
            # Remove training points from dead zone buffer
            if self.buffer_radius > 0:    
                # Buffer grid and clip training instances
                grid_poly = grid.loc[[grid_id]]
                grid_poly_buffer = grid_poly.buffer(self.buffer_radius)
                deadzone_points = gpd.clip(XYs, grid_poly_buffer)
                hblock_train_exclude = deadzone_points[~deadzone_points.index.isin(test_points)].index.values
                yield test_points, hblock_train_exclude

            else:
                # Yield empty array because no training data removed in dead zone when buffer is zero
                empty = np.array([], dtype=np.int)
                yield test_points, empty

class SLOO(BaseSpatialCV):
    
    def __init__(
        self,
        buffer_radius = None,
        shuffle = False,
        random_state = None
    ):
        self.buffer_radius = buffer_radius
        self.shuffle = shuffle
        self.random_state = random_state
    
    def _iter_test_indices(self, X):
        sloo_n = X.shape[0]
            
        for test_index in range(sloo_n):
                        
            # Build LOO buffer
            loo_buffer = X.loc[[test_index]].centroid.buffer(self.buffer_radius)
    
            # Exclude training instances in dead zone buffer 
            sloo_train_exclude = gpd.clip(X, loo_buffer).index.values
            sloo_train_exclude = sloo_train_exclude[sloo_train_exclude != test_index]
            
            # Convert test instane from scalar to array (1,)
            test_index = np.array([test_index])
                
            yield test_index, sloo_train_exclude
                  
class SKCV(BaseSpatialCV):
    
    def __init__(
        self,
        buffer_radius = None,
        shuffle = False,
        random_state = None
    ):
        pass

class RepeatSKCV(SKCV):
    
    def __init__(
        self,
        tiles_x=5,
        tiles_y=5,
        method='unique',
        buffer_radius=0,
        shuffle=False,
        direction='diagonal',
        n_groups=5,
        data=None,
        n_sims=10
    ):
        
        super().__init__(tiles_x, tiles_y, method, buffer_radius, 
                         shuffle, direction, n_groups, data, n_sims)
                
                
def cross_val_score(
    model,
    coordinates,
    X,
    y,
    cv,
    scoring
):
    # Fallback to aspatial CV if None
    if cv is None:
        cv = KFold(shuffle=True, random_state=0, n_splits=5)
    
    # check inputs 
#     coordinates, X, y = check_inputs(coordinates, X, y)
    
    
    X = np.array(X)
    y = np.array(y)
    
    scores = []
    scorer = make_scorer(scoring)
    for train_index, test_index in cv.split(coordinates):
        model.fit(X[train_index], y[train_index])
        scores.append(        
            scorer(model, X[test_index], 
                          y[test_index])
            
        )
    scores = np.asarray(scores)    
    return scores



