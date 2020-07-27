import unittest

class SKCV_Tester(unittest.TestCase):
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