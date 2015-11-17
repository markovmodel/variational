from __future__ import absolute_import
import unittest
import numpy as np
from variational.estimators import moments

__author__ = 'noe'

class TestMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # dense data
        cls.X_10 = np.random.rand(10000, 10)
        cls.Y_10 = np.random.rand(10000, 10)
        cls.X_100 = np.random.rand(10000, 100)
        cls.Y_100 = np.random.rand(10000, 100)
        # sparse data
        cls.X_10_sparse = np.zeros((10000, 10))
        cls.X_10_sparse[:, 0] = cls.X_10[:, 0]
        cls.Y_10_sparse = np.zeros((10000, 10))
        cls.Y_10_sparse[:, 0] = cls.Y_10[:, 0]
        cls.X_100_sparse = np.zeros((10000, 100))
        cls.X_100_sparse[:, :10] = cls.X_100[:, :10]
        cls.Y_100_sparse = np.zeros((10000, 100))
        cls.Y_100_sparse[:, :10] = cls.Y_100[:, :10]

        #cls.C_X10_X10 =

        return cls

    def _test_moments_X(self, X, remove_mean=False, min_const_cols_sparse=600):
        # proposed solution
        s_X, C_XX = moments.moments_XX(X, remove_mean=remove_mean, modify_data=False,
                                       min_const_cols_sparse=min_const_cols_sparse)
        # reference
        s_X_ref = X.sum(axis=0)
        if remove_mean:
            X = X - X.mean(axis=0)
        C_XX_ref = np.dot(X.T, X)
        # test
        assert np.allclose(s_X, s_X_ref)
        assert np.allclose(C_XX, C_XX_ref)

    def test_moments_X(self):
        # simple test, dense
        self._test_moments_X(self.X_10, remove_mean=False, min_const_cols_sparse=1000)
        self._test_moments_X(self.X_100, remove_mean=False, min_const_cols_sparse=1000)
        # mean-free, dense
        self._test_moments_X(self.X_10, remove_mean=True, min_const_cols_sparse=1000)
        self._test_moments_X(self.X_100, remove_mean=True, min_const_cols_sparse=1000)
        # simple test, sparse
        self._test_moments_X(self.X_10_sparse, remove_mean=False, min_const_cols_sparse=0)
        self._test_moments_X(self.X_100_sparse, remove_mean=False, min_const_cols_sparse=0)
        # mean-free, sparse
        self._test_moments_X(self.X_10_sparse, remove_mean=True, min_const_cols_sparse=0)
        self._test_moments_X(self.X_100_sparse, remove_mean=True, min_const_cols_sparse=0)


    def _test_moments_XY(self, X, Y, symmetrize=False, remove_mean=False, min_const_cols_sparse=600):
        s_X, s_Y, C_XX, C_XY = moments.moments_XXXY(X, Y, remove_mean=remove_mean, modify_data=False,
                                                    symmetrize=symmetrize, min_const_cols_sparse=min_const_cols_sparse)
        # reference
        s_X_ref = X.sum(axis=0)
        s_Y_ref = Y.sum(axis=0)
        if symmetrize:
            s_X_ref = 0.5*(s_X_ref + s_Y_ref)
            s_Y_ref = s_X_ref
        if remove_mean:
            X = X - s_X_ref/float(X.shape[0])
            Y = Y - s_Y_ref/float(X.shape[0])
        C_XX_ref = np.dot(X.T, X)
        C_XY_ref = np.dot(X.T, Y)
        if symmetrize:
            C_XY_ref = 0.5 * (np.dot(X.T, Y) + np.dot(Y.T, X))
        # test
        assert np.allclose(s_X, s_X_ref)
        assert np.allclose(s_Y, s_Y_ref)
        assert np.allclose(C_XX, C_XX_ref)
        assert np.allclose(C_XY, C_XY_ref)

    def test_moments_XY(self):
        # simple test, dense
        self._test_moments_XY(self.X_10, self.Y_10, symmetrize=False, remove_mean=False, min_const_cols_sparse=1000)
        self._test_moments_XY(self.X_100, self.Y_10, symmetrize=False, remove_mean=False, min_const_cols_sparse=1000)
        self._test_moments_XY(self.X_100, self.Y_100, symmetrize=False, remove_mean=False, min_const_cols_sparse=1000)
        # mean-free, dense
        self._test_moments_XY(self.X_10, self.Y_10, symmetrize=False, remove_mean=True, min_const_cols_sparse=1000)
        self._test_moments_XY(self.X_100, self.Y_10, symmetrize=False, remove_mean=True, min_const_cols_sparse=1000)
        self._test_moments_XY(self.X_100, self.Y_100, symmetrize=False, remove_mean=True, min_const_cols_sparse=1000)
        # simple test, sparse
        self._test_moments_XY(self.X_10_sparse, self.Y_10_sparse, symmetrize=False, remove_mean=False, min_const_cols_sparse=0)
        self._test_moments_XY(self.X_100_sparse, self.Y_10_sparse, symmetrize=False, remove_mean=False, min_const_cols_sparse=0)
        self._test_moments_XY(self.X_100_sparse, self.Y_100_sparse, symmetrize=False, remove_mean=False, min_const_cols_sparse=0)
        # mean-free, sparse
        self._test_moments_XY(self.X_10_sparse, self.Y_10_sparse, symmetrize=False, remove_mean=True, min_const_cols_sparse=0)
        self._test_moments_XY(self.X_100_sparse, self.Y_10_sparse, symmetrize=False, remove_mean=True, min_const_cols_sparse=0)
        self._test_moments_XY(self.X_100_sparse, self.Y_100_sparse, symmetrize=False, remove_mean=True, min_const_cols_sparse=0)


#
#
# class TestMomentCombination(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.X = np.random.rand(10000, 100)
#         # bias the first half
#         cls.X[:2000] += 1.0
#
#         # two-pass algorithm on the full data:
#         T_ref = np.shape(cls.X)[0]
#         cls.s_ref = cls.X.sum(axis=0)
#         m_ref = cls.s_ref / (1.0 * T_ref)
#         X_meanfree = cls.X - m_ref
#         cls.S_ref = np.dot(X_meanfree.T, X_meanfree)
#         return cls
#
#     def test_chunking(self):
#         from variational.estimators.covar import Chunked_Covar
#         # many passes
#         cc = Chunked_Covar()
#         L = 1000
#         for i in range(0, self.X.shape[0], L):
#             cc.add_chunk(self.X[i:i+L])
#         assert np.allclose(cc.sum1, self.s_ref)
#         assert np.allclose(cc.sum2, self.S_ref)


if __name__ == "__main__":
    unittest.main()