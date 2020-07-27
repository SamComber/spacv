import numbers
import numpy as np
import geopandas as gpd
from sklearn.metrics import make_scorer
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
        n_sims=10,
        distance_metric='euclidean'
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
        self.distance_metric = distance_metric
        
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
            
            test_indices, train_exclude = \
                super()._yield_test_indices(XYs, test_indices, self.buffer_radius)
            
            yield test_indices, train_exclude                     
                    
class SKCV(BaseSpatialCV):
    
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
        # If K = N, SLOO
        if len(XYs) == self.folds:
            indices_from_folds = XYs.index.values
        else:
            # Partion XYs space into folds
            XYs_to_2d = geometry_to_2d(XYs)
            km_skcv = MiniBatchKMeans(n_clusters = self.folds)
            labels = km_skcv.fit(XYs_to_2d).labels_
            indices_from_folds = [np.argwhere(labels == i).reshape(-1) 
                                      for i in range( self.folds )]
        
        for fold_indices in indices_from_folds:   
            
            if len(XYs) == self.folds:
                test_indices = np.array([fold_indices])
            else:
                test_indices = np.array(fold_indices)
            
            test_indices, train_exclude = \
                super()._yield_test_indices(XYs, test_indices, self.buffer_radius)
            
            yield test_indices, train_exclude
              
                
class RepeatSKCV(SKCV):
    
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
        
               
def cross_val_score(
    model,
    XYs,
    X,
    y,
    cv,
    scoring
):
    # Fallback to aspatial CV if None
    if cv is None:
        cv = KFold(shuffle=True, random_state=0, n_splits=5)
    XYs, X, y = validate_inputs(XYs, X, y)
    
    scores = []
    scorer = make_scorer(scoring)
    for train_index, test_index in cv.split(XYs):
        model.fit(X[train_index], y[train_index])
        scores.append(        
            scorer(model, X[test_index], 
                          y[test_index])
        )
    scores = np.asarray(scores)    
    return scores

def validate_inputs(XYs, X, y):
    """
    Validate data inputs to cross-validation procedure.
    """
    if not len(XYs) == len(X) == len(y):
        raise ValueError(
            "Data arrays are different lengths. Data lengths: XYs={} X={} y={}.".format(
                len(XYs), X.shape, y.shape
            )
        )
    shapes = [XY.shape for XY in geometry_to_2d(XYs)]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(
            "XYs are different shapes. Coordinate shapes: {}".format(
                shapes
            )
        )
    X = np.array(X)
    y = np.array(y)
        
    return XYs, X, y
