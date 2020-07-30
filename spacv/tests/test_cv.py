import sys, os
sys.path.append(os.path.abspath("../.."))
import unittest
import numpy as np
import geopandas as gpd
import spacv

class SKCV_Tester(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)
        x = np.random.randint(0, 3000, 30)
        y = np.random.randint(0, 3000, 30)
        
        self.gdf = gpd.GeoSeries(
            {'geometry' : gpd.points_from_xy(x,y)}
        )
        
        self.fold_test_one = np.array([1, 5, 13, 16, 17, 18, 21, 22, 23, 25, 26, 28])
        self.fold_test_two = np.array([2, 4, 6, 7, 11, 12, 14, 27, 29])
        self.fold_test_three = np.array([0, 3, 8, 9, 10, 15, 19, 20, 24])
    
        self.fold_train_one = np.array([0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15, 19, 20, 24, 27, 29])
        self.fold_train_two = np.array([0, 1, 3, 5, 8, 9, 10, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28])
        self.fold_train_three = np.array([1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29])
    
    def test_skcv(self):
        np.random.seed(10)
        scv = spacv.SKCV(folds = 3, buffer_radius = 450, random_state=123)
        self.assertEqual(scv.folds, 3)
        self.assertEqual(scv.buffer_radius, 450)
        self.assertEqual(scv.random_state, 123)
        
        fold_train, fold_test = [], []
        for train, test in scv.split(self.gdf):
            fold_train.append(train)
            fold_test.append(test)
            print(test)

        scv_fold_one = fold_test[0]
        scv_fold_two = fold_test[1]
        scv_fold_three = fold_test[2]
        
        scv_train_one = fold_train[0]
        scv_train_two = fold_train[1]
        scv_train_three = fold_train[2]
        
        np.testing.assert_equal(self.fold_test_one, scv_fold_one)
        np.testing.assert_equal(self.fold_test_two, scv_fold_two)
        np.testing.assert_equal(self.fold_test_three, scv_fold_three)
        
        # Training pts removed in deadzone buffer check
        np.testing.assert_equal(self.fold_train_one, scv_train_one)
        np.testing.assert_equal(self.fold_train_two, scv_train_two)
        np.testing.assert_equal(self.fold_train_three, scv_train_three)
        
        
class HBLOCK_Tester(unittest.TestCase):
    def setUp(self):
        
        pass
    
    def test_skcv(self):
        
        pass

    
class RepeatSKCV_Tester(unittest.TestCase):
    def setUp(self):
        
        pass
    
    def test_skcv(self):
        
        pass
        
suite = unittest.TestSuite()
test_classes = [SKCV_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)




    
