import numpy as np
import geopandas as gpd
from sklearn.metrics import make_scorer

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