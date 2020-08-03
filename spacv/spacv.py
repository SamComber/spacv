import warnings 
import numbers
import numpy as np
import geopandas as gpd
from sklearn.cluster import MiniBatchKMeans
from .base_classes import BaseSpatialCV
from .grid_builder import construct_blocks, assign_pt_to_grid
from .utils import geometry_to_2d, convert_geodataframe, load_custom_polygon

__all__ = [
    "HBLOCK",
    "SKCV",
    "RepeatedSKCV",
    "UserDefinedSCV"
]

class HBLOCK(BaseSpatialCV):
    """
    H-Blocking spatial cross-validator. Partitions study area
    into a series of grid polygons that are assigned into
    different folds based on several user-defined options. HBLOCK
    exposes several parameters for choosing block sizes, types and 
    fold assignment.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    tiles_x : integer, default=5
        Number of grid tiles in the West-East direction.
    tiles_y : integer, default=5
        Number of grid tiles in the North-South direction.
    shape : string, default='square'
        Specify shape of grid polygons, square or hex.
    method : string, default='unique'
        Choose grid ID assignment method to build folds. Options
        are: unique, where every polygon in the grid is a fold; 
        systematic, where folds reflect diagonal or anti-diagonal 
        patterns across the study area; random, where folds are 
        randomly assigned into groups determined by n_groups parameter;
        and optimized_random, where random assignment of grids into
        groups are optimized by reducing disimilarity between folds.
    buffer_radius : integer, default=0
        Buffer radius (dead zone) to exclude training points that are 
        within a defined distance of test data within a fold.
    direction : string, default='diagonal'
        Choose direction of pattern for systematic grid assignment,
        diagonal or anti-diagonal (anti).
    n_groups : integer, default=5
        Number of folds to randomly assign grid polygons into.
    data : array
        Array containing covariates used in predictive task. Used to
        calculate disimilarity of feature space between folds to 
        find the optimized random grid assignment.
    n_sims : integer, default=10
        Number of iterations in which to find optimized grid assignment
        into folds.
    distance_metric : string, default='euclidean'
        Distance metric used to reconcile points that sit at exact
        border between two grids. Defaults to euclidean assuming
        projected coordinate system, otherwise use haversine for
        unprojected spaces.
    random_state : int, default=None
        random_state is the seed used by the random number generator.
        
    Examples
    -------- 
    
    
    """
    def __init__(
        self,
        tiles_x=5,
        tiles_y=5,
        shape='square',
        method='unique',
        buffer_radius=0,
        direction='diagonal',
        n_groups=5,
        data=None,
        n_sims=10,
        distance_metric='euclidean',
        random_state=None
    ):        
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.shape = shape
        self.method = method
        self.buffer_radius = buffer_radius
        self.direction = direction
        self.n_groups = n_groups
        self.data = data
        self.n_sims = n_sims
        self.distance_metric = distance_metric
        self.n_splits = tiles_x*tiles_y
        self.random_state = random_state
        
    def _iter_test_indices(self, XYs):
        """
        Generates integer indices corresponding to test sets and 
        training indices to be excluded from model training.
        
        Parameters
        ----------
        XYs : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.
            
        Yields
        ------
        test_indices : array
            The testing set indices for that fold.
        train_exclude : array
            The training set indices to exclude for that fold.
        """       
        # Define grid type used in CV procedure
        grid = construct_blocks(XYs, 
                      tiles_x = self.tiles_x, 
                      tiles_y = self.tiles_y, 
                      shape = self.shape,
                      method = self.method, 
                      direction = self.direction, 
                      n_groups = self.n_groups,
                      data = self.data, 
                      n_sims = self.n_sims,
                      random_state = self.random_state)
    
        # Convert to GDF to use Geopandas functions
        XYs = gpd.GeoDataFrame(({'geometry':XYs}))
                            
        # Assign pts to grids
        XYs = assign_pt_to_grid(XYs, grid, self.distance_metric)
        grid_ids = np.unique(grid.grid_id)

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
    Spatial K-fold cross-validator. Modification of standard 
    CV to overcome biased prediction performance of estimates 
    due to autocorrelation in spatial data. Overoptimistic bias 
    in performance is prevented by ensuring spatial proximity of 
    test data, and maintaing a training set that is within a certain
    spatial distance from the test dataset.
    
    When K=N, SKCV becomes a spatial leave-one-out (SLOO) cross-validator.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    buffer_radius : integer, default=0
        Buffer radius (dead zone) to exclude training points that are 
        within a defined distance of test data within a fold.
    random_state : int, RandomState instance or None, optional, default=None
        random_state is the seed used by the random number generator.

    Examples
    -------- 
    
    """
    def __init__(
        self,
        n_splits=10,
        buffer_radius = 0,
        random_state = None
    ):
        self.n_splits = n_splits
        self.buffer_radius = buffer_radius
        self.random_state = random_state
        
    def _iter_test_indices(self, XYs):
        """
        Generates integer indices corresponding to test sets.
        
        Parameters
        ----------
        X : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.
            
        Yields
        ------
        test_indices : array
            The testing set indices for that fold.
        train_exclude : array
            The training set indices to exclude for that fold.
        """
              
        if self.n_splits > len(XYs) :
            raise ValueError(
                "Number of specified n_splits (folds) is larger than number of data points. Given {} observations and {} folds.".format(
                    len(XYs), self.n_splits
                )
            )
        sloo = len(XYs) == self.n_splits
        lattice = any(XYs.geom_type == 'Polygon') or any(XYs.geom_type == 'MultiPolygon')
        
        # If K = N, SLOO
        if sloo:
            num_samples = XYs.shape[0]
            indices_from_folds = np.arange(num_samples)
        else:
            # Partion XYs space into folds
            XYs_to_2d = geometry_to_2d(XYs)
            km_skcv = MiniBatchKMeans(n_clusters = self.n_splits, random_state=self.random_state)
            labels = km_skcv.fit(XYs_to_2d).labels_
            uniques, counts = np.unique(labels, return_counts=True)

            check_fold_n = (counts < 2)
            if check_fold_n.any():
                warn = "{} folds contain less than three points and do not form polygons.".format( check_fold_n.sum() )
                warnings.warn(warn)
            indices_from_folds = [np.argwhere(labels == i).reshape(-1) 
                                                  for i in uniques]    
        for fold_indices in indices_from_folds:   
            if sloo:
                test_indices = np.array([fold_indices])
                fold_polygon = XYs.loc[test_indices].buffer(self.buffer_radius)
            elif lattice:
                test_indices = np.array(fold_indices)
                fold_polygon = XYs.loc[test_indices].unary_union.buffer(self.buffer_radius) 
            else: # skcv
                test_indices = np.array(fold_indices)
                fold_polygon = XYs.loc[test_indices].unary_union.convex_hull.buffer(self.buffer_radius)
                
            test_indices, train_exclude = \
                super()._remove_buffered_indices(XYs, test_indices, 
                                            self.buffer_radius, fold_polygon)
            yield test_indices, train_exclude
                
class RepeatedSKCV(SKCV):
    """
    Repeats Spatial K-Fold cross-validation (SKCV) n times with different 
    K-means randomization in each repetition. Given the sensitivity of K-means
    to the initial, randomly initialised starting values of the centroid seeds,
    RepeatedSKCV repeats SKCV a number of times, yielding a generator of 
    n_repeats * K test and training splits.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    **cvargs : additional params
        Constructor parameters for cv.
    """
    
    def __init__(
        self,
        n_repeats=10,
        n_splits=10,
        random_state=None,
        **cvargs
    ):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        self.cv = SKCV
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.cvargs = cvargs
                
    def split(self, XYs):
        n_repeats = self.n_repeats
        for idx in range(n_repeats):
            cv = self.cv(self.n_splits, **self.cvargs)
            for train_index, test_index in cv.split(XYs):
                yield train_index, test_index
                
class UserDefinedSCV(BaseSpatialCV):
    """
    Spatial cross-validation using user-defined polygons.
    
    Yields indices to split data into training and test sets.
    
    Parameters
    ----------
    custom_polygons : string, GeoSeries
        File path to user defined grid polygons used to assign data
        points into folds.
    buffer_radius : integer, default=0
        Buffer radius (dead zone) to exclude training points that are 
        within a defined distance of test data within a fold.
    distance_metric : string, default='euclidean'
        Distance metric used to reconcile points that sit at exact
        border between two grids. Defaults to euclidean assuming
        projected coordinate system, otherwise use haversine for
        unprojected spaces.
        
    Yields
    ------
    test_indices : array
        The testing set indices for that fold.
    train_exclude : array
        The training set indices to exclude for that fold.
    
    """
    def __init__(
        self,
        custom_polygons,
        buffer_radius = 0,
        distance_metric = 'euclidean'
    ):  
        self.buffer_radius = buffer_radius
        self.custom_polygons = load_custom_polygon(custom_polygons)
        self.distance_metric = distance_metric
        
    def _iter_test_indices(self, XYs):
        """
        Generates integer indices corresponding to test sets and 
        training indices to be excluded from model training.
        
        Parameters
        ----------
        XYs : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.
            
        Yields
        ------
        test_indices : array
            The testing set indices for that fold.
        train_exclude : array
            The training set indices to exclude for that fold.
        """  
        grid = self.custom_polygons
        grid['grid_id'] = grid.index
        grid_ids = np.unique(grid.grid_id)
        
        XYs = assign_pt_to_grid(XYs, grid, self.distance_metric)
        
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