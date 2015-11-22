__author__ = 'noe'

import time
import numpy as np
from variational.estimators import moments

def genS(N):
    """ Generates sparsities given N (number of cols) """
    S = [10, 90, 100, 500, 900, 1000, 2000, 5000, 7500, 9000, 10000, 20000, 50000, 75000, 90000]  # non-zero
    return [s for s in S if s <= N]


def gendata(L, N, n_var=None, const=False):
    X = np.random.rand(L, N)  # random data
    if n_var is not None:
        if const:
            Xsparse = np.ones((L, N))
        else:
            Xsparse = np.zeros((L, N))
        Xsparse[:, :n_var] = X[:, :n_var]
        X = Xsparse
    return X


def reftime_momentsXX(X, remove_mean=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        s_ref = X.sum(axis=0)  # computation of mean
        if remove_mean:
            X = X - s_ref/float(X.shape[0])
        C_XX_ref = np.dot(X.T, X)  # covariance matrix
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def mytime_momentsXX(X, remove_mean=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        s, C_XX = moments.moments_XX(X, remove_mean=remove_mean)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def reftime_momentsXXXY(X, Y, remove_mean=False, symmetrize=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        sx = X.sum(axis=0)  # computation of mean
        sy = Y.sum(axis=0)  # computation of mean
        if symmetrize:
            sx = 0.5*(sx + sy)
            sy = sx
        if remove_mean:
            X = X - sx/float(X.shape[0])
            Y = Y - sy/float(Y.shape[0])
        if symmetrize:
            C_XX_ref = np.dot(X.T, X)
            C_XY_ref = np.dot(X.T, Y)
        else:
            C_XX_ref = np.dot(X.T, X) + np.dot(Y.T, Y)
            C_XY_ref = np.dot(X.T, Y) + np.dot(Y.T, X)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def mytime_momentsXXXY(X, Y, remove_mean=False, symmetrize=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        sx, sy, C_XX, C_XY = moments.moments_XXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def benchmark_moments(L=10000, N=10000, xy=False, remove_mean=False, symmetrize=False, const=False):
    #S = [10, 100, 1000]
    S = genS(N)
    nrep = 3

    # time for reference calculation
    X = gendata(L, N)
    if xy:
        Y = gendata(L, N)
        reftime = reftime_momentsXXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize, nrep=nrep)
    else:
        reftime = reftime_momentsXX(X, remove_mean=remove_mean, nrep=nrep)

    # my time
    times = np.zeros(len(S))
    for k, s in enumerate(S):
        X = gendata(L, N, n_var=s, const=const)
        if xy:
            Y = gendata(L, N, n_var=s, const=const)
            times[k] = mytime_momentsXXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize, nrep=nrep)
        else:
            times[k] = mytime_momentsXX(X, remove_mean=remove_mean, nrep=nrep)

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
    print 'moments_XX\txy = ' + str(xy) + '\tremove_mean = ' + str(remove_mean) + '\tsym = ' + str(symmetrize) + '\tconst = ' + str(const)
    print rows[0] + ('\t%i' * table.shape[1])%tuple(table[0])
    print rows[1] + ('\t%i' * table.shape[1])%tuple(table[1])
    print rows[2] + ('\t%i' * table.shape[1])%tuple(table[2])
    print rows[3] + ('\t%.3f' * table.shape[1])%tuple(table[3])
    print rows[4] + ('\t%.3f' * table.shape[1])%tuple(table[4])
    print rows[5] + ('\t%.3f' * table.shape[1])%tuple(table[5])
    print


def main():
    #X = gendata(10000, 10000, n_var=10, const=True)

    #t1 = time.time()
    #moments.moments_XX(X, remove_mean=False)
    #t2 = time.time()
    #print t2 - t1
    #return

    # TODO: ALERT - xy=True, sym=True und const=True ist langsamer als trivial. Bei const lohnt sich das vorgehen nicht.
    # TODO: --> zero und const in verschiedene Funktionen, um sie anders zu behandeln?
    # TODO: --> bei symmetrisch + const lohnt es sich nicht in der covar-Funktion zu symmetrisieren
    # TODO: --> der Aufwand der nachtraeglichen Addition ist schlimmstenfalls gleich.
    # TODO: --> fuer dense nur einfache Korrelationen (XY - eine Funktion reicht)
    # TODO: --> fuer const nur einfache Korrelationen (XY - eine Funktion reicht)
    # TODO: --> fuer sparse (0) einfache Korrelationen (XY) und symmetrische Korrelationen (XX + YY, XY + YX) - 2 Fn
    # TODO: ALERT - combination remove_mean=False and const=True is slow. However remove_mean=True and const=True is faster than with zeros (ah?)
    # TODO: ALERT - remove_mean in dense mode is a bit slower than trivial computation.
    # 1000000 x 100
    #benchmark_moments(100000, 100, xy=False)
    #benchmark_moments(100000, 100, xy=True)
    # 100000 x 1000
    #benchmark_moments(10000, 1000, xy=False)
    #benchmark_moments(10000, 1000, xy=True)
    # 10000 x 10000, all meaningful combinations
    benchmark_moments(xy=False, remove_mean=False, symmetrize=False, const=True)
    benchmark_moments(xy=True, remove_mean=False, symmetrize=True, const=True)
    return
    benchmark_moments(xy=False, remove_mean=False, symmetrize=False, const=False)
    benchmark_moments(xy=False, remove_mean=False, symmetrize=False, const=True)
    benchmark_moments(xy=False, remove_mean=True, symmetrize=False, const=False)
    benchmark_moments(xy=False, remove_mean=True, symmetrize=False, const=True)
    benchmark_moments(xy=True, remove_mean=False, symmetrize=False, const=False)
    benchmark_moments(xy=True, remove_mean=False, symmetrize=False, const=True)
    benchmark_moments(xy=True, remove_mean=False, symmetrize=True, const=False)
    benchmark_moments(xy=True, remove_mean=False, symmetrize=True, const=True)
    benchmark_moments(xy=True, remove_mean=True, symmetrize=False, const=False)
    benchmark_moments(xy=True, remove_mean=True, symmetrize=False, const=True)
    benchmark_moments(xy=True, remove_mean=True, symmetrize=True, const=False)
    benchmark_moments(xy=True, remove_mean=True, symmetrize=True, const=True)


if __name__ == "__main__":
    main()