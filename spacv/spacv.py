import numpy as np
import geopandas as gpd
from sklearn.metrics import make_scorer
from .base_classes import BaseSpatialCV

class HBLOCK(BaseSpatialCV):
    
    def __init__(
        self,
        XYs=None,
        tiles_x=5,
        tiles_y=5,
        method='unique',
        buffer_radius=None
    ):
        
        # 
        if isinstance(XYs, gpd.GeoDataFrame):
            XYs = XYs['geometry']
        
        self.XYs = XYs
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.method = method
        self.buffer_radius = buffer_radius
            

    def _iter_test_indices(self, X):
        
        tiles_x = self.tiles_x
        tiles_y = self.tiles_y
        buffer_radius = self.buffer_radius

        XYs = gpd.GeoDataFrame(({'geometry':X}))
                
        grid = blocks(XYs, tiles_x = tiles_x, tiles_y = tiles_y)

        # Yield test indices and optionally training indices within buffer
        for grid_id in grid.grid_id:

            # 
            grid_poly = grid.loc[[grid_id]]
            test_points = gpd.sjoin(XYs, grid_poly).index.values

            # Remove empty grids
            if len(test_points) < 1:
                continue

            # Remove training points from dead zone buffer
            if buffer_radius > 0:    

                grid_poly_buffer = grid_poly.buffer(buffer_radius)

                deadzone_points = gpd.clip(XYs, grid_poly_buffer)

                hblock_train_exclude = deadzone_points[~deadzone_points.index.isin(test_points)].index.values

                yield test_points, hblock_train_exclude

            # Else yield test points
            else:
                
                yield test_points, None


class SLOO(BaseSpatialCV):
    
    def __init__(
        self,
        XYs = None,
        buffer_radius = None,
        shuffle = False,
        random_state = None
    ):
        self.XYs = XYs
        self.buffer_radius = buffer_radius
        self.shuffle = shuffle
        self.random_state = random_state
        
        minx, miny, maxx, maxy = self.XYs.total_bounds
        
        if buffer_radius > maxx-minx or buffer_radius > maxy-miny:
            raise ValueError(
                "Radius too large and excludes all points. Given {}.".format(
                    self.buffer_radius
                )
            )
            
    def _iter_test_indices(self, X):
    
        buffer_radius = self.buffer_radius
          
        sloo_n = X.shape[0]
            
        for test_index in range(sloo_n):
                        
            # eliminate data points outside radius from training set
            loo_buffer = X.loc[[test_index]].centroid.buffer(radius)
    
            # build indices to exclude from training
            sloo_train_exclude = gpd.clip(X, loo_buffer).index.values
        
            # exclude test index from training set removal indices
            sloo_train_exclude = sloo_train_exclude[sloo_train_exclude != test_index]
                
            yield test_index, sloo_train_exclude
            
            
def cross_val_score(
    model,
    X,
    y,
    cv,
    scoring
):
    # Fallback to (a)spatial CV if None
    if cv is None:
        cv = KFold(shuffle=True, random_state=0, n_splits=5)
    
    X = np.array(X)
    y = np.array(y)
    
    scores = []
    scorer = make_scorer(scoring)
    for train_index, test_index in cv.split(X):
        model.fit(X[train_index], y[train_index])
        
        # generalise this
        scores.append(        
            scorer(model, X[test_index].reshape(1, -1), 
                          y[test_index].reshape(-1, 1))
        )
    scores = np.asarray(scores)    
    
    return scores