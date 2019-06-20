import unittest

from eolearn.core.graph import DirectedGraph


class TestDAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d1 = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})

    def test_is_edge(self):
        d = DirectedGraph({1: [2]})
        self.assertTrue(d.is_edge(1, 2))
        self.assertFalse(d.is_edge(2, 1))

    def test_add_edge(self):
        d = DirectedGraph()
        self.assertFalse(d.is_edge(1, 2))
        self.assertEqual(d.get_outdegree(1), 0)

        d.add_edge(1, 2)

        self.assertEqual(d.get_outdegree(1), 1)
        self.assertEqual(d.get_indegree(2), 1)
        self.assertTrue(d.is_edge(1, 2))

        d.add_edge(1, 2)

        self.assertTrue(1, 2)
        self.assertEqual(d.get_indegree(2), 1)

    def test_del_edge(self):
        d = DirectedGraph({1: [2]})
        self.assertTrue(d.is_edge(1, 2))
        d.del_edge(1, 2)
        self.assertFalse(d.is_edge(1, 2))

    def test_neigbors(self):
        self.assertEqual(self.d1.neighbors(1), [2, 3])
        self.assertEqual(self.d1.neighbors(2), [4])
        self.assertEqual(self.d1.neighbors(3), [4])
        self.assertEqual(self.d1.neighbors(4), [])

    def test_get_indegree(self):
        self.assertEqual(self.d1.get_indegree(1), 0)
        self.assertEqual(self.d1.get_indegree(2), 1)
        self.assertEqual(self.d1.get_indegree(3), 1)
        self.assertEqual(self.d1.get_indegree(4), 2)

    def test_get_outdegree(self):
        self.assertEqual(self.d1.get_outdegree(1), 2)
        self.assertEqual(self.d1.get_outdegree(2), 1)
        self.assertEqual(self.d1.get_outdegree(3), 1)
        self.assertEqual(self.d1.get_outdegree(4), 0)

    def test_vertices(self):
        self.assertEqual(set(self.d1.vertices()), set([1, 2, 3, 4]))

        d2 = DirectedGraph()
        d2.add_edge(1, 2)
        d2.add_edge(2, 3)
        d2.add_edge(3, 4)
        self.assertEqual(set(d2.vertices()), set([1, 2, 3, 4]))

    def test_add_vertex(self):
        d = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})
        d.add_vertex(5)
        self.assertTrue(5 in d)
        self.assertEqual(d.get_indegree(5), 0)
        self.assertEqual(d.get_outdegree(5), 0)
        self.assertEqual(d[5], [])

    def test_del_vertex(self):
        d = DirectedGraph({1: [2, 3], 2: [4], 3: [4]})

        self.assertEqual(d.get_outdegree(1), 2)
        self.assertEqual(d.get_indegree(4), 2)
        self.assertEqual(len(d), 4)
        self.assertTrue(2 in d)

        d.del_vertex(2)

        self.assertEqual(d.get_outdegree(1), 1)
        self.assertEqual(d.get_indegree(4), 1)
        self.assertEqual(len(d), 3)
        self.assertFalse(2 in d)


if __name__ == '__main__':
    unittest.main()
