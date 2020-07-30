import sys, os
sys.path.append(os.path.abspath("../.."))
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
    
    def test_skcv(self):
        np.random.seed(10)
        grid = construct_blocks(self.XYs, self.tiles_x, self.tiles_y, self.method)
        # Test count of unique folds
        np.testing.assert_equal(grid.shape[0], 50 )
        np.testing.assert_equal( len(grid.grid_id.unique()), 50 )
        
suite = unittest.TestSuite()
test_classes = [SquareUniqueGrid_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)




    
