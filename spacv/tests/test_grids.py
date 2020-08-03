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
        
        
class SquareRandomGrid_Tester(unittest.TestCase):
    def setUp(self):  
        np.random.seed(10)
        self.tiles_x = 4
        self.tiles_y = 4
        self.method = 'random'
        self.n_groups = 5
        self.shape = 'square'
        self.random_state = 123
        
        self.grid_assignment = np.array([
               [2, 4, 2, 1],
               [3, 2, 3, 1],
               [1, 0, 1, 1],
               [0, 0, 1, 3]
        ])

        x = np.random.randint(0, 1000, 100)
        y = np.random.randint(0, 1000, 100)

        self.XYs = gpd.GeoDataFrame(
                    {'geometry' : gpd.points_from_xy(x,y)}
                )     
        
    def test_random_grid(self):
        np.random.seed(10)
    
        grid = construct_blocks(XYs = self.XYs, 
                                tiles_x = self.tiles_x, 
                                tiles_y = self.tiles_y, 
                                method = self.method, 
                                n_groups = self.n_groups,
                                shape = self.shape,
                                random_state = self.random_state)

        np.testing.assert_equal(grid.shape[0], 4 * 4 )
        np.testing.assert_equal( len(grid.grid_id.unique()), 5 )
        
        np.testing.assert_equal(grid.grid_id.values.reshape(4,4), self.grid_assignment)

        
class SquareSystematicGrid_Tester(unittest.TestCase):
    def setUp(self):  
        np.random.seed(10)
        self.tiles_x = 5
        self.tiles_y = 5
        self.method = 'systematic'
        self.direction = 'diagonal'
        self.shape = 'square'
        
        self.grid_assignment = np.array([[4, 3, 2, 1, 0],
                                         [5, 4, 3, 2, 1],
                                         [6, 5, 4, 3, 2],
                                         [7, 6, 5, 4, 3],
                                         [8, 7, 6, 5, 4]])


        x = np.random.randint(0, 1000, 100)
        y = np.random.randint(0, 1000, 100)

        self.XYs = gpd.GeoDataFrame(
                    {'geometry' : gpd.points_from_xy(x,y)}
                )     
        
    def test_systematic_grid(self):
        np.random.seed(10)
    
        grid = construct_blocks(XYs = self.XYs, 
                                tiles_x = self.tiles_x, 
                                tiles_y = self.tiles_y, 
                                method = self.method, 
                                direction = self.direction,
                                shape = self.shape)
        
        np.testing.assert_equal(grid.shape[0], 5 * 5 )
        np.testing.assert_equal(grid.grid_id.values.reshape(5,5), self.grid_assignment)

        
class SquareOptimizedRandomGrid_Tester(unittest.TestCase):
    def setUp(self): 
        pass
        
    def test_random_optimized(self):
        pass
        
class HexUniqueGrid_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        pass
    
    def test_hex_unique_grid(self):
        np.random.seed(10)
        
        
suite = unittest.TestSuite()
test_classes = [SquareUniqueGrid_Tester, SquareRandomGrid_Tester, SquareSystematicGrid_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)




    
