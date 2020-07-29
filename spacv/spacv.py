import numbers
import numpy as np
import geopandas as gpd
from .base_classes import BaseSpatialCV
from .grid_builder import construct_blocks, assign_pt_to_grid
from .utils import geometry_to_2d, convert_geodataframe
from sklearn.cluster import MiniBatchKMeans

class HBLOCK(BaseSpatialCV):
    """
    H-Blocking spatial cross-validator.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    tiles_x : integer, default=5
        Number of grid tiles in the West-East direction.
    tiles_y : integer, default=5
        Number of grid tiles in the North-South direction.
    shape : string, default='square'
        Specify 
    method : string, default='unique'
    
    buffer_radius : integer, default=0
    
    shuffle : boolean, default=False
    
    direction : string, default='diagonal'
    
    n_groups : integer, default=5
    
    data : dataframe
    
    n_sims : integer, default=10
    
    distance_metric : string, default='euclidean'
    
    """
    def __init__(
        self,
        tiles_x=5,
        tiles_y=5,
        shape='square',
        method='unique',
        buffer_radius=0,
        shuffle=False,
        direction='diagonal',
        n_groups=5,
        data=None,
        n_sims=10,
        distance_metric='euclidean'
    ):        
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.shape = shape
        self.method = method
        self.buffer_radius = buffer_radius
        self.shuffle = shuffle
        self.direction = direction
        self.n_groups = n_groups
        self.data = data
        self.n_sims = n_sims
        self.distance_metric = distance_metric
        
    def _iter_test_indices(self, XYs):
                
        # Define grid type used in CV procedure
        grid = construct_blocks(XYs, 
                      tiles_x = self.tiles_x, 
                      tiles_y = self.tiles_y, 
                      shape = self.shape,
                      method = self.method, 
                      direction = self.direction, 
                      n_groups = self.n_groups,
                      data = self.data, 
                      n_sims = self.n_sims)
    
        # Convert to GDF to use Geopandas functions
        XYs = gpd.GeoDataFrame(({'geometry':XYs}))
                            
        # Assign pts to grids
        XYs = assign_pt_to_grid(XYs, grid, self.distance_metric)
        grid_ids = np.unique(grid.grid_id)
                
        # Shuffle grid order 
        if self.shuffle:
            check_random_state(self.random_state).shuffle(grid_ids)

        # Yield test indices and optionally training indices within buffer
        for grid_id in grid_ids:
            test_indices = XYs.loc[XYs['grid_id'] == grid_id ].index.values
            # Remove empty grids
            if len(test_indices) < 1:
                continue
            grid_poly_buffer = grid.loc[[grid_id]].buffer(self.buffer_radius)
            test_indices, train_exclude = \
                super()._remove_buffered_indices(XYs, test_indices, 
                                            self.buffer_radius, grid_poly_buffer)
            yield test_indices, train_exclude                     
                    
class SKCV(BaseSpatialCV):
    """
    H-Blocking spatial cross-validator.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    def __init__(
        self,
        folds=10,
        buffer_radius = 0,
        shuffle = False,
        random_state = None
    ):
        self.folds = folds
        self.buffer_radius = buffer_radius
        
    def _iter_test_indices(self, XYs):
        if self.folds > len(XYs) :
            raise ValueError(
                "Number of specified folds is larger than number of data points. Given {} observations and {} folds.".format(
                    len(XYs), self.folds
                )
            )
        sloo = len(XYs) == self.folds
            
        # If K = N, SLOO
        if sloo:
            num_samples = XYs.shape[0]
            indices_from_folds = np.arange(num_samples)
            
        else:
            # Partion XYs space into folds
            XYs_to_2d = geometry_to_2d(XYs)
            km_skcv = MiniBatchKMeans(n_clusters = self.folds)
            labels = km_skcv.fit(XYs_to_2d).labels_
            indices_from_folds = [np.argwhere(labels == i).reshape(-1) 
                                      for i in range( self.folds )]
        
        for fold_indices in indices_from_folds:   
            if sloo:
                test_indices = np.array([fold_indices])
            else:
                test_indices = np.array(fold_indices)
            fold_convex_hull = XYs.loc[test_indices].unary_union.convex_hull
            test_indices, train_exclude = \
                super()._remove_buffered_indices(XYs, test_indices, 
                                            self.buffer_radius, fold_convex_hull)
            yield test_indices, train_exclude
                
class RepeatSKCV(SKCV):
    """
    
    Parameters
    ----------
    
    n_repeats
    
    n_folds
    
    
    """
    
    def __init__(
        self,
        n_repeats=10,
        folds=10,
        **kwargs
    ):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        cv = SKCV
        self.cv = cv
        self.n_repeats = n_repeats
        self.folds = folds
        self.kwargs = kwargs
                
    def split(self, XYs):
        n_repeats = self.n_repeats
        for idx in range(n_repeats):
            cv = self.cv(self.folds, **self.kwargs)
            for train_index, test_index in cv.split(XYs):
                yield train_index, test_index
        
def compute_gcv(y, X):
    """
    Compute generalized cross-validation (GCV) for i.i.d data. GSV
    is an approximation of leave-one-out (LOO) CV.
    
    
    ADD: accomodate space.
    """
    y = y.reshape(-1, 1)
    ols = spreg.ols.OLS(y, X)
    hat = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    mse = np.mean( (y - ols.predy)**2)
    n = len(y)
    h_value = np.trace( np.identity(n) - hat ) / n
    gcv = mse / h_value**2

    return gcv