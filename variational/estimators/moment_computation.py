__author__ = 'noe'

import math, sys, numbers
import numpy as np
from variational.estimators.covar_c import covartools


def covar_dense(X, Y, symmetrize=False):
    """

    Parameters
    ----------
    C : ndarray (N, M)
        Covariance matrix. If given, the result will be added to this matrix.

    """
    # check input
    if symmetrize and np.shape(X)[1]!=np.shape(Y)[1]:
        raise ValueError('Cannot compute symmetric covariance matrix for differently sized data')
    Craw = np.dot(X.T, Y)
    if symmetrize:
        return 0.5*(Craw + Craw.T)
    else:
        return Craw


def _is_zero(x):
    if isinstance(x, numbers.Number):
        return x == 0.0
    if isinstance(x, np.ndarray):
        return np.all(x==0)
    return False

def covar_sparse(Xvar, mask_X, Yvar, mask_Y,
                 xsum=0, xconst=0, ysum=0, yconst=0, symmetrize=False):
    """
    Computes the covariance matrix C = X' Y when the data matrices can be
    permuted to have the form X = (V, C) with rows (v_t, c) and Y = (W, D)
    with rows (w_t, d). Here, c and d are constant vectors.
    The resulting matrix has the form:

    .. math:
        C &=& [Xvar' Yvar  0 ]
          & & [0     0 ]

    for non zero but constants, we have the form:

    .. math:
        C &=& [Xvar' Yvar  xsum yconst' ]
          & & [xconst ysum'  xsum ysum'  ]

    where c = x0full[mask_X] and y0full[mask_Y].

    Parameters
    ----------
    C : ndarray (N, M)
        Covariance matrix. If given, the result will be added to this matrix.

    """
    print 'in cov sparse'
    sys.stdout.flush()
    # check input
    if symmetrize and len(mask_X) != len(mask_Y):
        raise ValueError('Cannot compute symmetric covariance matrix for differently sized data')
    C = np.zeros((len(mask_X), len(mask_Y)))
    print 'created array'
    sys.stdout.flush()
    # Block 11
    C11 = np.dot(Xvar.T, Yvar)
    print 'C11 ', C11[0,0]
    if symmetrize:
        C[mask_X][:, mask_Y] = 0.5*(C11 + C11.T)
    else:
        C[mask_X][:, mask_Y] = C11
    print 'C11 set ', C[0,0]
    print mask_X[0:20]
    print mask_Y[0:20]
    print 'C11 added'
    sys.stdout.flush()
    # other blocks
    xsum_is_0 = _is_zero(xsum)
    ysum_is_0 = _is_zero(ysum)
    xconst_is_0 = _is_zero(xconst)
    yconst_is_0 = _is_zero(yconst)
    print xsum_is_0, ysum_is_0, xconst_is_0, yconst_is_0
    sys.stdout.flush()
    # Block 12 and 21
    if not (xsum_is_0 or yconst_is_0) or not (ysum_is_0 or xconst_is_0):
        print 'Block 12 and 21'
        print (xsum_is_0 or yconst_is_0)
        print (ysum_is_0 or xconst_is_0)
        print not (xsum_is_0 or yconst_is_0)
        print not (ysum_is_0 or xconst_is_0)
        sys.stdout.flush()
        C12 = np.outer(xsum, yconst)
        C21 = np.outer(xconst, ysum)
        if symmetrize:
            Coff = 0.5*(C12 + C21.T)
            C[mask_X][:, ~mask_Y] = Coff
            C[~mask_X][:, mask_Y] = Coff
        else:
            C[mask_X][:, ~mask_Y] = C12
            C[~mask_X][:, mask_Y] = C21
    print 'off-diag blocks done'
    sys.stdout.flush()
    # Block 22
    if not (xconst_is_0 or yconst_is_0):
        print 'Block 22'
        sys.stdout.flush()
        C22 = np.outer(xconst, yconst)
        if symmetrize:
            C[~mask_X][:, ~mask_Y] = 0.5*(C22 + C22.T)
        else:
            C[~mask_X][:, ~mask_Y] = C22
    print 'finished'
    sys.stdout.flush()
    return C


def sum_sparse(xsum, mask_X, xconst):
    s = np.zeros(len(mask_X))
    s[mask_X] = xsum
    s[~mask_X] = xconst
    return s


# TODO: implement sparse_tol
#    sparse_tol: float
#        Threshold for considering column to be zero in order to save computing
#        effort when the data is sparse or almost sparse.
#        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
#        is not given) of the covariance matrix will be set to zero. If Y is
#        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
#        covariance matrix will be set to zero.


def covar_XX(X, remove_mean=False, min_const_cols_sparse=600):
    """
    Parameters
    ----------
    min_const_cols_sparse=600
    Empirically, it has been found that independent of size, at least 600 columns in both matrices should be zero
    in order for columns striding and computing the smaller-sized matrix product to be faster than the full product.


    """
    # CHECK INPUT
    assert type(X) is np.ndarray, 'X must be an ndarray'
    T, N = X.shape

    print 'here'
    sys.stdout.flush()

    # DETERMINE SPARSITY
    if N > min_const_cols_sparse:
        mask_X = covartools.nonconstant_cols(X)  # 1/0 vector where 0 is constant and 1 is variable
        const_cols_X = np.where(~mask_X)[0]
        var_cols_X = np.where(mask_X)[0]
        sparse_mode = len(const_cols_X) > 600
    else:
        mask_X = None
        const_cols_X = None
        var_cols_X = None
        sparse_mode = False

    print 'sparse mode = ', sparse_mode
    sys.stdout.flush()

    # SUBSELECT DATA
    X0 = X  # X0 is the data matrix we will work with
    print 'buh'
    sys.stdout.flush()
    if sparse_mode:
        print 'subselecting'
        sys.stdout.flush()
        X0 = X0[:, var_cols_X]  # shrink matrix to variable part

    print 'subselected: ', str(X0.shape)
    sys.stdout.flush()

    # CONVERT DATA FORMAT FOR EFFICIENT CALCULATION
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    print 'target type: ', dtype
    sys.stdout.flush()
    if X.dtype.kind == 'b' and T < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    if X0.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        print 'type conversion'
        sys.stdout.flush()
        X0 = X0.astype(dtype)

    print 'colsum and mean'
    sys.stdout.flush()

    # COLUMN SUM AND MEAN
    x0sum_before = X0.sum(axis=0)  # this is the mean before subtracting it.
    print 'colsum done'
    sys.stdout.flush()
    x0sum_after = x0sum_before
    xconst = 0
    if remove_mean:
        covartools.subtract_row(X0, x0sum_before/float(T))  # use fast row subtraction avoiding broadcasting
        print 'mean sub done'
        sys.stdout.flush()
        x0sum_after = 0

    print 'cov'
    sys.stdout.flush()

    # COVARIANCE MATRIX
    if sparse_mode:
        if remove_mean:
            xconst = X[0, const_cols_X].astype(np.float64)
        print 'sum sparse'
        sys.stdout.flush()
        xsum = sum_sparse(x0sum_before, mask_X, xconst)
        print 'cov sparse'
        sys.stdout.flush()
        C = covar_sparse(X0, mask_X, X0, mask_X, xsum=x0sum_after, xconst=xconst, ysum=x0sum_after, yconst=xconst)
        print 'done'
        sys.stdout.flush()
    else:
        xsum = x0sum_before
        C = np.dot(X.T, X)

    return xsum, C


def covar_XXXY(X, Y, remove_mean=False, symmetrize=False, min_const_cols_sparse=600):
    """
    Parameters
    ----------
        Empirically, it has been found that independent of size, at least 600 columns in both matrices should be zero
        in order for columns striding and computing the smaller-sized matrix product to be faster than the full product.

    """
    # CHECK INPUT
    assert (type(X) is np.ndarray) and (type(Y) is np.ndarray), 'X and Y must ndarrays'
    T, M = X.shape
    T2, N = Y.shape
    assert T == T2, 'X and Y must have same number of rows'

    # DETERMINE SPARSITY
    if M > min_const_cols_sparse and N > min_const_cols_sparse:
        mask_X = covartools.nonconstant_cols(X)  # 1/0 vector where 0 is constant and 1 is variable
        const_cols_X = np.where(~mask_X)[0]
        var_cols_X = np.where(mask_X)[0]
        mask_Y = covartools.nonconstant_cols(Y)  # 1/0 vector where 0 is constant and 1 is variable
        const_cols_Y = np.where(~mask_Y)[0]
        var_cols_Y = np.where(mask_Y)[0]
        sparse_mode = math.sqrt(len(const_cols_X)*len(const_cols_X)) > 600
    else:
        mask_X = None
        const_cols_X = None
        var_cols_X = None
        mask_Y = None
        const_cols_Y = None
        var_cols_Y = None
        sparse_mode = False

    # SUBSELECT DATA
    X0 = X  # X0 is the data matrix we will work with
    Y0 = Y  # X0 is the data matrix we will work with
    if sparse_mode:
        X0 = X0[:, var_cols_X]  # shrink matrix to variable part
        Y0 = Y0[:, var_cols_Y]  # shrink matrix to variable part

    # CONVERT DATA FORMAT FOR EFFICIENT CALCULATION
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    if X.dtype.kind == 'b' and T < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    if X0.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        X0 = X0.astype(dtype)
    if Y0.dtype not in (np.float64, dtype):
        Y0 = Y0.astype(np.float64)

    # COLUMN SUM AND MEAN
    x0sum_before = X0.sum(axis=0)  # this is the mean before subtracting it.
    y0sum_before = Y0.sum(axis=0)  # this is the mean before subtracting it.
    x0sum_after = x0sum_before
    y0sum_after = y0sum_before
    xconst = 0
    yconst = 0
    if remove_mean:
        if symmetrize:
            symsum = 0.5*(x0sum_before+y0sum_before)
            symmean = symsum / float(T)
            covartools.subtract_row(X0, symmean)  # use fast row subtraction avoiding broadcasting
            covartools.subtract_row(Y0, symmean)  # use fast row subtraction avoiding broadcasting
            x0sum_after = x0sum_before - symsum
            y0sum_after = y0sum_before - symsum
        else:
            covartools.subtract_row(X0, x0sum_before/float(T))  # use fast row subtraction avoiding broadcasting
            covartools.subtract_row(Y0, y0sum_before/float(T))  # use fast row subtraction avoiding broadcasting
            x0sum_after = 0
            y0sum_after = 0

    # COVARIANCE MATRIX
    if sparse_mode:
        if remove_mean:
            xconst = X[0, const_cols_X].astype(np.float64)
            yconst = Y[0, const_cols_Y].astype(np.float64)
        xsum = sum_sparse(x0sum_before, mask_X, xconst)
        ysum = sum_sparse(y0sum_before, mask_Y, yconst)
        C0 = covar_sparse(X0, mask_X, X0, mask_X,
                          xsum=x0sum_after, xconst=xconst, ysum=x0sum_after, yconst=xconst, symmetrize=False)
        Ct = covar_sparse(X0, mask_X, Y0, mask_Y,
                          xsum=x0sum_after, xconst=xconst, ysum=y0sum_after, yconst=yconst, symmetrize=symmetrize)
    else:
        xsum = x0sum_before
        ysum = y0sum_before
        C0 = covar_dense(X, X, symmetrize=False)
        Ct = covar_dense(X, Y, symmetrize=symmetrize)

    return xsum, ysum, C0, Ct


# TODO: handle type conversion from inefficient to efficient numerical types (bool -> float/float32)
# TODO: this is an essential function. Should be lifted to API (but without moments object that can stay internal)
# TODO: implement symmetric mean
def compute_moments(X, Y=None, weight=1.0, centralize=True, symmetric_mean=True):
    """ Computes the first and second moments
    """
    w = np.shape(X)[0]
    s = np.sum(X, axis=0)
    mean = s / float(w)
    if centralize:
        X = X - mean
    if Y is None:
        M = np.dot(X.T, X)
    else:
        if centralize:
            Y = Y - mean
        M = np.dot(X.T, Y)
    if weight != 1.0:
        return Moments(weight*w, weight*s, weight*M)
    else:
        return Moments(w, s, M)


class Moments(object):

    def __init__(self, w, s, M):
        """
        Parameters
        ----------
        w : float
            statistical weight.
                w = \sum_t w_t
            In most cases, :math:`w_t=1`, and then w is just the number of samples that went into s1, S2.
        s : ndarray(n,)
            sum over samples:
            .. math:
                s = \sum_t w_t x_t
        M : ndarray(n, n)
            .. math:
                M = (X-s)^T (X-s)
        """
        self.w = float(w)
        self.s = s
        self.M = M

    def copy(self):
        return Moments(self.w, self.s.copy(), self.M.copy())

    def combine(self, other):
        w1 = self.w
        w2 = other.w
        w = w1 + w2
        ds = (w2/w1) * self.s - other.s
        # update
        self.w = w1 + w2
        self.s += other.s
        self.M += other.M + (w1 / (w2 * w)) * np.outer(ds, ds)
        return self

    @property
    def mean(self):
        return self.s / self.w

    @property
    def covar(self):
        """ Returns M / (w-1)

        Careful: The normalization w-1 assumes that we have counts as weights.

        """
        return self.M / (self.w-1)


class MomentsStorage:
    """
    """

    def __init__(self, nsave, rtol=1.5):
        self.nsave = nsave
        self.storage = []
        self.rtol = rtol

    def _can_merge_tail(self):
        """ Checks if the two last list elements can be merged
        """
        if len(self.storage) < 2:
            return False
        return self.storage[-2].w <= self.storage[-1].w * self.rtol

    def store(self, moments):
        """ Store object X with weight w
        """
        if len(self.storage) == self.nsave:  # merge if we must
            # print 'must merge'
            self.storage[-1].combine(moments)
        else:  # append otherwise
            # print 'append'
            self.storage.append(moments)
        # merge if possible
        while self._can_merge_tail():
            # print 'merge: ',self.storage
            M = self.storage.pop()
            # print 'pop last: ',self.storage
            self.storage[-1].combine(M)
            # print 'merged: ',self.storage

    @property
    def moments(self):
        """
        """
        # collapse storage if necessary
        while len(self.storage) > 1:
            # print 'collapse'
            M = self.storage.pop()
            self.storage[-1].combine(M)
        # print 'return first element'
        return self.storage[0]
