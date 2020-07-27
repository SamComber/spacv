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
            
            # Remove training points from dead zone buffer
            if self.buffer_radius > 0:    
                # Buffer grid and clip training instances
                candidate_deadzone = XYs.loc[~XYs.index.isin( test_indices )]
                grid_poly_buffer = grid.loc[[grid_id]].buffer(self.buffer_radius)
                deadzone_points = gpd.clip(candidate_deadzone, grid_poly_buffer)
                hblock_train_exclude = deadzone_points[~deadzone_points.index.isin(test_indices)].index.values
                yield test_indices, hblock_train_exclude

            else:
                # Yield empty array because no training data removed in dead zone when buffer is zero
                _ = np.array([], dtype=np.int)
                yield test_indices, _
                
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
            indices_from_folds = [np.argwhere(labels == i).reshape(-1) for i in range(10)]
        
        for fold_indices in indices_from_folds:
            test_indices = np.array([fold_indices])

            # Remove training points from dead zone buffer
            if self.buffer_radius > 0:    
                # Buffer fold and clip training instances
                candidate_deadzone = XYs.loc[~XYs.index.isin( test_indices)]
                fold_convex_hull = gpd.GeoSeries(XYs.loc[test_indices].unary_union.convex_hull).buffer(self.buffer_radius)
                deadzone_points = gpd.clip(candidate_deadzone, fold_convex_hull)
                                
                hblock_train_exclude = deadzone_points[~deadzone_points.index.isin(test_indices)].index.values
                yield test_indices, hblock_train_exclude

            else:
                # Yield empty array because no training data removed in dead zone when buffer is zero
                _ = np.array([], dtype=np.int)
                yield test_indices, _
                
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
