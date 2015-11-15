from __future__ import absolute_import
import unittest
import numpy as np

__author__ = 'noe'



class TestNetworkPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10000, 100)
        # bias the first half
        cls.X[:2000] += 1.0

        # two-pass algorithm on the full data:
        T_ref = np.shape(cls.X)[0]
        cls.s_ref = cls.X.sum(axis=0)
        m_ref = cls.s_ref / (1.0 * T_ref)
        X_meanfree = cls.X - m_ref
        cls.S_ref = np.dot(X_meanfree.T, X_meanfree)
        return cls

    def test_chunking(self):
        from variational.estimators.covar import Chunked_Covar
        # many passes
        cc = Chunked_Covar()
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add_chunk(self.X[i:i+L])
        assert np.allclose(cc.sum1, self.s_ref)
        assert np.allclose(cc.sum2, self.S_ref)


if __name__ == "__main__":
    unittest.main()