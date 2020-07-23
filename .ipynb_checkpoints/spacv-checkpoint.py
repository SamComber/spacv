import numpy as np
import geopandas as gpd
from sklearn.metrics import make_scorer


class HBLOCK():
    
    pass

def build_grid(XYs, tiles_x, tiles_y):
    
    minx, miny, maxx, maxy = XYs.total_bounds

    dx = (maxx - minx) / tiles_x
    dy = (maxy - miny) / tiles_y

    polys = []

    for tile_y in range(0, tiles_y):

        for tile_x in range(0, tiles_x):

            polys.append(
                Polygon([
                    
                    # bottom-left
                    (minx + (tile_x * dx), miny + ( tile_y * dy ) ), 

                    # bottom-right
                    (minx + ((tile_x + 1) * dx), miny + ( tile_y * dy  ) ), 

                    # top-right
                    (minx + ( (tile_x + 1) * dx), miny  + ( (tile_y + 1) * dy) ), 

                    #top-left
                    (minx + (tile_x * dx), miny + (tile_y + 1) * dy )
            ]))

    return gpd.GeoDataFrame({'geometry':polys})

def assign_randomized(grid, n_groups):
    
    n_random_grps = np.arange(0, n_groups)
    
    n_grids = grid.shape[0]

    grid_id = np.random.choice(n_random_grps, size=n_grids, replace=True)

    # ADD: check most dissimilar
    
    return grid_id

def assign_systematic(grid, direction='norm'):

    sys_matrix = np.arange(0, tiles_x * tiles_y).reshape(tiles_y, tiles_x)

    leng, wid= sys_matrix.shape

    # matrix diagonal direction (anti vs. norm)
    if direction == 'norm':
        diags = [sys_matrix.diagonal(i) for i in range(wid-1, -leng,-1)]
    if direction == 'anti':
        diags = [sys_matrix[::-1,:].diagonal(i) for i in range(-leng+1, wid)]
        
    systematic_lookup = dict([tuple([element, key])  for key, diag in enumerate(diags) for element in diag])

    grid_id = grid.index.map(systematic_lookup)

    return grid_id

def grid_assignment(grid, method, *args):
    
    # assign grid assignment type
    if method == 'randomized':
        
        grid['grid_id'] = assign_randomized(grid)
        
    if method == 'unique':
        
        grid['grid_id'] = grid.index
        
    if method == 'systematic':
        
        grid['grid_id'] = assign_systematic(grid, *args)
        
    return grid
    
    


class SLOO():
    
    def __init__(
        self,
        XYs = None,
        radius = None,
        shuffle = False,
        random_state = None
    ):
        self.XYs = XYs
        self.radius = radius
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X, y=None):
 
        num_samples = X.shape[0]

        indices = np.arange(num_samples)
                
        for test_index, sloo_train_exclude in self._iter_test_indices(X):
                
            train_index = np.setdiff1d(
                                np.union1d(
                                    indices,
                                    sloo_train_exclude
                                ), np.intersect1d(indices, sloo_train_exclude)
                            )
            
            test_index = indices[test_index]
                
            yield train_index, test_index    
            
    def _iter_test_indices(self, X=None):
    
        radius = self.radius
        XYs = self.XYs
          
        sloo_n = X.shape[0]
            
        for test_index in range(sloo_n):
                        
            # eliminate data points outside radius from training set
            loo_buffer = XYs.loc[[test_index]].centroid.buffer(radius)
    
            # build indices to exclude from training
            sloo_train_exclude = gpd.clip(XYs, loo_buffer).index.values
        
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
    X = np.array(X)
    y = np.array(y)
    
    scores = []
    
    scorer = make_scorer(scoring)
    
    for train_index, test_index in cv.split(X):

        model.fit(X[train_index], y[train_index])
        
        scores.append(        
            scorer(model, X[test_index].reshape(1, -1), 
                          y[test_index].reshape(-1, 1))
        )
    return scores