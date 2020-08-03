from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import geopandas as gpd
from sklearn.model_selection import BaseCrossValidator
from .utils import convert_geoseries, convert_geodataframe

class BaseSpatialCV(BaseCrossValidator, metaclass=ABCMeta):
    """
    Base class for partitioning-based spatial cross-validation approaches.
    """
    def __init__(
        self
    ):
        self.buffer_radius = buffer_radius
        
    def split(self, XYs):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        XYs : GeoSeries
            GeoSeries containing shapely Points that identify Easting
            and Northing coordinates of data points.
    
        Yields
        ------
        train : ndarray
            Training set indices for iteration.
        test : ndarray
            Testing set indices for iteration.
        """
        XYs = convert_geoseries(XYs).reset_index(drop=True)
        minx, miny, maxx, maxy = XYs.total_bounds
        
        buffer_radius = self.buffer_radius
        if buffer_radius > maxx-minx or buffer_radius > maxy-miny:
            raise ValueError(
                "buffer_radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
        num_samples = XYs.shape[0]
        indices = XYs.index.values
        
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
            
    def _remove_buffered_indices(self, XYs, test_indices, buffer_radius, geometry_buffer):
        # Remove training points from dead zone buffer
        if buffer_radius > 0:            
            # Buffer grid and clip training instances            
            candidate_deadzone = XYs.loc[~XYs.index.isin( test_indices )]            
            candidate_deadzone = convert_geodataframe(candidate_deadzone)
            geometry_buffer = convert_geodataframe(geometry_buffer)
            deadzone_points = gpd.sjoin(candidate_deadzone, geometry_buffer)
            train_exclude = deadzone_points.loc[~deadzone_points.index.isin(test_indices)].index.values
            return test_indices, train_exclude
        else:
            # Yield empty array because no training data removed in dead zone when buffer is zero
            _ = np.array([], dtype=np.int)
            return test_indices, _
        
    @abstractmethod
    def _iter_test_indices(self, XYs):
        """
        Generates integer indices corresponding to test sets and 
        training indices to be excluded from model training.
        
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
    
    def get_n_splits(self):
        """
        Returns the number of folds used in the cross-validation.
        
        Returns
        -------
        n_splits : int
            Returns the number of folds in the cross-validator.
        """
        return self.n_splits

        