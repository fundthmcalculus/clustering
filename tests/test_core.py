import unittest
from clustering.core import simple_cluster

class TestClustering(unittest.TestCase):
    def test_empty_data(self):
        self.assertEqual(simple_cluster([]), [])
        
    def test_single_element(self):
        self.assertEqual(simple_cluster([1]), [[1]])
        
    def test_basic_clustering(self):
        data = [1, 2, 10, 11]
        expected = [[1, 2], [10, 11]]
        self.assertEqual(simple_cluster(data, threshold=5), expected)
        
    def test_unsorted_data(self):
        data = [11, 1, 10, 2]
        expected = [[1, 2], [10, 11]]
        self.assertEqual(simple_cluster(data, threshold=5), expected)

if __name__ == '__main__':
    unittest.main()
