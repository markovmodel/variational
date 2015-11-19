from __future__ import absolute_import
from __future__ import print_function
__author__ = 'noe'

import time
import numpy as np
from variational.estimators import moments

def benchmark_moments_XX_sparse(remove_mean=False):
    L = 10000  # number of samples
    N = 10000  # number of dimensions
    S = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]  # non-zero
    X = np.random.rand(L, N)  # random data

    # time for reference calculation
    t1 = time.time()
    s_ref = X.sum(axis=0)  # computation of mean
    if remove_mean:
        X0 = X - s_ref/float(L)
    C_XX_ref = np.dot(X.T, X)  # covariance matrix
    t2 = time.time()
    reftime = t2-t1

    # my time
    times = np.zeros(len(S))
    for k, s in enumerate(S):
        # build sub-matrix
        Xsparse = np.zeros((L, N))
        Xsparse[:, :s] = X[:, :s]
        # time of covariance
        t1 = time.time()
        s, C_XX = moments.moments_XX(Xsparse, remove_mean=remove_mean)
        t2 = time.time()
        times[k] = t2-t1

    # assemble report
    rows = ['L, data points', 'N, dimensions', 'S, nonzeros', 'time trivial', 'time moments_XX', 'speed-up']
    table = np.zeros((6, len(S)))
    table[0, :] = L
    table[1, :] = N
    table[2, :] = S
    table[3, :] = reftime
    table[4, :] = times
    table[5, :] = reftime / times

    # print table
    print('moments_XX\tremove_mean = ' + str(remove_mean))
    print(rows[0] + ('\t%i' * table.shape[1])%tuple(table[0]))
    print(rows[1] + ('\t%i' * table.shape[1])%tuple(table[1]))
    print(rows[2] + ('\t%i' * table.shape[1])%tuple(table[2]))
    print(rows[3] + ('\t%.3f' * table.shape[1])%tuple(table[3]))
    print(rows[4] + ('\t%.3f' * table.shape[1])%tuple(table[4]))
    print(rows[5] + ('\t%.3f' * table.shape[1])%tuple(table[5]))
    print()


def main():
    benchmark_moments_XX_sparse(remove_mean=True)
    benchmark_moments_XX_sparse(remove_mean=False)


if __name__ == "__main__":
    main()