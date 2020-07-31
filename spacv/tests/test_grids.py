import unittest
import numpy as np
import geopandas as gpd
from spacv.grid_builder import *

class SquareUniqueGrid_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        self.tiles_x = 5
        self.tiles_y = 10
        self.method = 'unique'
        x = np.random.randint(0, 3000, 100)
        y = np.random.randint(0, 3000, 100)
        self.XYs = gpd.GeoDataFrame(
            {'geometry' : gpd.points_from_xy(x,y)}
        )            
    
    def test_unique_grid(self):
        np.random.seed(10)
        grid = construct_blocks(self.XYs, self.tiles_x, self.tiles_y, self.method)
        # Test count of unique folds
        np.testing.assert_equal(grid.shape[0], 5 * 10 )
        np.testing.assert_equal( len(grid.grid_id.unique()), 50 )
        
        
class SquareSystematicGrid_Tester(unittest.TestCase):
    def setUp(self):  
        np.random.seed(10)
        self.tiles_x = 4
        self.tiles_y = 4
        self.method = 'random'
        self.n_groups = 5
        
        self.grid_assignment = np.array([
            [1, 4, 0, 1],
            [3, 4, 1, 0],
            [1, 2, 0, 1],
            [0, 2, 0, 4]
        ])

        x = np.random.randint(0, 1000, 100)
        y = np.random.randint(0, 1000, 100)

        self.XYs = gpd.GeoDataFrame(
                    {'geometry' : gpd.points_from_xy(x,y)}
                )     
        
    def test_systematic_grid(self):
        np.random.seed(10)
    
        grid = construct_blocks(self.XYs, 
                                self.tiles_x, 
                                self.tiles_y, 
                                self.method, 
                                self.n_groups)
        
        np.testing.assert_equal(grid.shape[0], 4 * 4 )
        np.testing.assert_equal( len(grid.grid_id.unique()), 5 )
        np.testing.assert_equal(grid.grid_id.values.reshape(4,4), self.grid_assignment)

        
suite = unittest.TestSuite()
test_classes = [SquareUniqueGrid_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)




    
