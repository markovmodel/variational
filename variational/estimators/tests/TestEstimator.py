from __future__ import absolute_import
import numpy as np
import os
import unittest

import variational.estimators.lagged_correlation as lcor


class TestEstimator(unittest.TestCase):
    """ Performs tests for correlation matrix estimator using two long
        trajectories from one-dimensional double-well potential and 11
        Gaussian functions.
            
    """
        
    def setUp(self):
        """ Load the data and construct estimator object.
        
        """
        # Load the data:
        path_local = os.path.dirname(os.path.abspath(__file__))
        data = np.load(path_local + "/TestData.npz")
        # Extract the pieces:
        self.C0 = data['C0']
        self.Ct = data['Ct']
        self.traj0 = data['traj0']
        self.traj1 = data['traj1']
        # Create test object:
        tau = 500
        output_dimension = 11
        self.estimator = lcor.LaggedCorrelation(output_dimension, tau)
        # Add data to the test object:
        self.estimator.add(self.traj0)
        self.estimator.add(self.traj1)
        
    def test_Ct(self):
        self.assertTrue(np.allclose(self.estimator.GetCt(), self.Ct))
        
    def test_C0(self):
        self.assertTrue(np.allclose(self.estimator.GetC0(), self.C0))
        
    def test_WrongDimension(self):
        with self.assertRaises(Exception):
            self.estimator.add(self.traj0[:,:-1])


if __name__ == '__main__':
    unittest.main()        