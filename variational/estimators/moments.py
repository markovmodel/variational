__author__ = 'noe'

import math, sys, numbers
import numpy as np
from variational.estimators.covar_c import covartools
import warnings


def _covar_dense(X, Y, symmetrize=False):
    """ Computes the unnormalized covariance matrix between X and Y

    If symmetrize is False, computes :math:`C = X^\top Y`.
    If symmetrize is True, computes :math:`C = \frac{1}{2} (X^\top Y + Y^\top X)`.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    Y : ndarray (T, N)
        Data matrix

    Returns
    -------
    C : ndarray (M, N)
        Unnormalized covariance matrix

    """
    if symmetrize and np.shape(X)[1]!=np.shape(Y)[1]:
        raise ValueError('Cannot compute symmetric covariance matrix for differently sized data')
    Craw = np.dot(X.T, Y)
    if symmetrize:
        return 0.5*(Craw + Craw.T)
    else:
        return Craw


def _is_zero(x):
    """ Returns True if x is numerically 0 or an array with 0's. """
    if isinstance(x, numbers.Number):
        return x == 0.0
    if isinstance(x, np.ndarray):
        return np.all(x == 0)
    return False


def _covar_sparse(Xvar, mask_X, Yvar, mask_Y,
                  xsum=0, xconst=0, ysum=0, yconst=0, symmetrize=False):
    """ Computes the unnormalized covariance matrix between X and Y, exploiting sparsity

    Computes the unnormalized covariance matrix :math:`C = X^\top Y`
    (for symmetric=False) or :math:`C = \frac{1}{2} (X^\top Y + Y^\top X)`
    (for symmetric=True). Suppose the data matrices can be column-permuted
    to have the form

    .. math:
        X &=& (X_{\mathrm{var}}, X_{\mathrm{const}})
        Y &=& (Y_{\mathrm{var}}, Y_{\mathrm{const}})

    with rows:

    .. math:
        x_t &=& (x_{\mathrm{var},t}, x_{\mathrm{const}})
        y_t &=& (y_{\mathrm{var},t}, y_{\mathrm{const}})

    where :math:`x_{\mathrm{const}},\:y_{\mathrm{const}}` are constant vectors.
    The resulting matrix has the general form:

    .. math:
        C &=& [X_{\mathrm{var}}^\top Y_{\mathrm{var}}  x_{sum} y_{\mathrm{const}}^\top ]
          & & [x_{\mathrm{const}}^\top y_{sum}^\top    x_{sum} x_{sum}^\top            ]

    where :math:`x_{sum} = \sum_t x_{\mathrm{var},t}` and
    :math:`y_{sum} = \sum_t y_{\mathrm{var},t}`.

    Parameters
    ----------
    Xvar : ndarray (T, m)
        Part of the data matrix X with :math:`m \le M` variable columns.
    mask_X : ndarray (M)
        Boolean array of size M of the full columns. False for constant column,
        True for variable column in X.
    Yvar : ndarray (T, n)
        Part of the data matrix Y with :math:`n \le N` variable columns.
    mask_Y : ndarray (N)
        Boolean array of size N of the full columns. False for constant column,
        True for variable column in Y.
    xsum : ndarray (m)
        Column sum of variable part of data matrix X
    xconst : ndarray (M-m)
        Values of the constant part of data matrix X
    ysum : ndarray (n)
        Column sum of variable part of data matrix Y
    yconst : ndarray (N-n)
        Values of the constant part of data matrix Y
    symmetrize : bool
        Compute symmetric mean and covariance matrix.

    Returns
    -------
    C : ndarray (M, N)
        Unnormalized covariance matrix.

    """
    #print 'in cov sparse'
    #sys.stdout.flush()
    # check input
    if symmetrize and len(mask_X) != len(mask_Y):
        raise ValueError('Cannot compute symmetric covariance matrix for differently sized data')
    C = np.zeros((len(mask_X), len(mask_Y)))
    #print 'created array'
    #sys.stdout.flush()
    # Block 11
    C11 = np.dot(Xvar.T, Yvar)
    if symmetrize:
        C[np.ix_(mask_X, mask_Y)] = 0.5*(C11 + C11.T)[:, :]
    else:
        C[np.ix_(mask_X, mask_Y)] = C11[:, :]
    #print 'C11 added'
    #sys.stdout.flush()
    # other blocks
    xsum_is_0 = _is_zero(xsum)
    ysum_is_0 = _is_zero(ysum)
    xconst_is_0 = _is_zero(xconst)
    yconst_is_0 = _is_zero(yconst)
    # Block 12 and 21
    if not (xsum_is_0 or yconst_is_0) or not (ysum_is_0 or xconst_is_0):
        #print 'Block 12 and 21'
        #sys.stdout.flush()
        C12 = np.outer(xsum, yconst)
        C21 = np.outer(xconst, ysum)
        if symmetrize:
            Coff = 0.5*(C12 + C21.T)
            C[np.ix_(mask_X, ~mask_Y)] = Coff
            C[np.ix_(~mask_X, mask_Y)] = Coff
        else:
            C[np.ix_(mask_X, ~mask_Y)] = C12
            C[np.ix_(~mask_X, mask_Y)] = C21
    #print 'off-diag blocks done'
    #sys.stdout.flush()
    # Block 22
    if not (xconst_is_0 or yconst_is_0):
        #print 'Block 22'
        #sys.stdout.flush()
        C22 = np.outer(xconst, yconst)
        if symmetrize:
            C[np.ix_(~mask_X, ~mask_Y)] = 0.5*(C22 + C22.T)
        else:
            C[np.ix_(~mask_X, ~mask_Y)] = C22
    #print 'finished'
    #sys.stdout.flush()
    return C


def _sum_sparse(xsum, mask_X, xconst):
    s = np.zeros(len(mask_X))
    s[mask_X] = xsum
    s[~mask_X] = xconst
    return s


def moments_XX(X, remove_mean=False, modify_data=False, min_const_cols_sparse=600, sparse_tol=0.0):
    """ Computes the first two unnormalized moments of X

    Computes :math:`s = \sum_t x_t` and :math:`C = X^\top X` while exploiting
    zero or constant columns in the data matrix.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    min_const_cols_sparse=600
        Empirically, it has been found that independent of size, at least 600
        columns in both matrices should be zero in order for columns striding
        and computing the smaller-sized matrix product to be faster than the
        full product.
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
        covariance matrix will be set to zero.

    Returns
    -------
    s : ndarray (M)
        sum
    C : ndarray (M, M)
        unnormalized covariance matrix

    """
    # CHECK INPUT
    assert type(X) is np.ndarray, 'X must be an ndarray'
    T, N = X.shape

    # DETERMINE SPARSITY
    if N > min_const_cols_sparse:
        mask_X = covartools.variable_cols(X, tol=sparse_tol, min_constant=min_const_cols_sparse)  # bool vector
        nconst = np.where(~mask_X)[0]
        sparse_mode = len(nconst) > min_const_cols_sparse
    else:
        mask_X = None
        sparse_mode = False

    #print 'sparse mode = ', sparse_mode
    #sys.stdout.flush()

    # SUBSELECT DATA
    X0 = X  # X0 is the data matrix we will work with
    if sparse_mode:
        #print 'subselecting'
        #sys.stdout.flush()
        X0 = X0[:, mask_X]  # shrink matrix to variable part

    #print 'subselected: ', str(X0.shape)
    #sys.stdout.flush()

    # CONVERT DATA FORMAT FOR EFFICIENT CALCULATION
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    working_on_copy = False  # keep track if we have a copy or the input data
    #print 'target type: ', dtype
    #sys.stdout.flush()
    if X.dtype.kind == 'b' and T < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    if X0.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        #print 'type conversion'
        #sys.stdout.flush()
        X0 = X0.astype(dtype, order='C')
        working_on_copy = True  # now we have a true copy

    #print 'colsum and mean'
    #sys.stdout.flush()

    # COLUMN SUM AND MEAN
    x0sum_before = X0.sum(axis=0)  # this is the mean before subtracting it.
    #print 'colsum done'
    #sys.stdout.flush()
    x0sum_after = x0sum_before
    xconst = 0
    if remove_mean:
        mean = x0sum_before/float(T)
        # write to new copy. We can't use or subtract_row function because the sliced data is not C-contiguous
        if sparse_mode and not working_on_copy:
            X0copy = np.ndarray(X0.shape)
            X0 = np.subtract(X0, mean, X0copy)
        else:
            X0 = covartools.subtract_row(X0, mean, inplace=modify_data)  # faster
        # print 'mean sub done with inplace = ', modify_data
        #sys.stdout.flush()
        x0sum_after = 0

    #print 'cov'
    #sys.stdout.flush()

    # COVARIANCE MATRIX
    if sparse_mode:
        if remove_mean:
            xconst = X[0, ~mask_X].astype(dtype)
        #print 'sum sparse'
        #sys.stdout.flush()
        xsum = _sum_sparse(x0sum_before, mask_X, xconst)
        #print 'cov sparse'
        #sys.stdout.flush()
        C = _covar_sparse(X0, mask_X, X0, mask_X, xsum=x0sum_after, xconst=xconst, ysum=x0sum_after, yconst=xconst)
        #print 'done'
        #sys.stdout.flush()
    else:
        xsum = x0sum_before
        C = np.dot(X0.T, X0)

    return xsum, C


def moments_XXXY(X, Y, remove_mean=False, modify_data=False, symmetrize=False, min_const_cols_sparse=600, sparse_tol=0.0):
    """ Computes the first two unnormalized moments of X and Y

    If symmetrize is False, computes

    .. math:
        s_x  &=& \sum_t x_t`
        s_y  &=& \sum_t y_t`
        C_XX &=& X^\top X
        C_XY &=& X^\top Y

    If symmetrize is True, computes

    .. math:
        s_x = s_y &=& \sum_t x_t`
        C_XX      &=& X^\top X
        C_XY      &=& \frac{1}{2} (X^\top Y + Y^\top X)

    while exploiting zero or constant columns in the data matrix.

    Parameters
    ----------
    X : ndarray (T, M)
        Data matrix
    remove_mean : bool
        True: remove column mean from the data, False: don't remove mean.
    modify_data : bool
        If remove_mean=True, the mean will be removed in the data matrix X,
        without creating an independent copy. This option is faster but might
        lead to surprises because your input array is changed.
    min_const_cols_sparse=600
        Empirically, it has been found that independent of size, at least 600
        columns in both matrices should be zero in order for columns striding
        and computing the smaller-sized matrix product to be faster than the
        full product.
    sparse_tol: float
        Threshold for considering column to be zero in order to save computing
        effort when the data is sparse or almost sparse.
        If max(abs(X[:, i])) < sparse_tol, then row i (and also column i if Y
        is not given) of the covariance matrix will be set to zero. If Y is
        given and max(abs(Y[:, i])) < sparse_tol, then column i of the
        covariance matrix will be set to zero.

    Returns
    -------
    s_x : ndarray (M)
        x-sum
    s_y : ndarray (N)
        y-sum
    C_XX : ndarray (M, M)
        unnormalized covariance matrix of X
    C_XY : ndarray (M, N)
        unnormalized covariance matrix of XY

    """
    # CHECK INPUT
    assert (type(X) is np.ndarray) and (type(Y) is np.ndarray), 'X and Y must ndarrays'
    T, M = X.shape
    T2, N = Y.shape
    assert T == T2, 'X and Y must have same number of rows'

    # DETERMINE SPARSITY
    if M > min_const_cols_sparse and N > min_const_cols_sparse:
        mask_X = covartools.variable_cols(X, tol=sparse_tol, min_constant=min_const_cols_sparse)  # bool vector
        nconst_X = np.where(~mask_X)[0]
        mask_Y = covartools.variable_cols(Y, tol=sparse_tol, min_constant=min_const_cols_sparse)  # bool vector
        nconst_Y = np.where(~mask_Y)[0]
        sparse_mode = math.sqrt(nconst_X*nconst_Y) > min_const_cols_sparse
    else:
        mask_X = None
        mask_Y = None
        sparse_mode = False

    # SUBSELECT DATA
    X0 = X  # X0 is the data matrix we will work with
    Y0 = Y  # X0 is the data matrix we will work with
    if sparse_mode:
        X0 = X0[:, mask_X]  # shrink matrix to variable part
        Y0 = Y0[:, mask_Y]  # shrink matrix to variable part

    # CONVERT DATA FORMAT FOR EFFICIENT CALCULATION
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    working_on_X_copy = False  # keep track if we have a copy or the input data
    working_on_Y_copy = False  # keep track if we have a copy or the input data
    if X.dtype.kind == 'b' and T < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    if X0.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        X0 = X0.astype(dtype)
        working_on_X_copy = True
    if Y0.dtype not in (np.float64, dtype):
        Y0 = Y0.astype(np.float64)
        working_on_Y_copy = True

    # COLUMN SUM AND MEAN
    x0sum_before = X0.sum(axis=0)  # this is the mean before subtracting it.
    y0sum_before = Y0.sum(axis=0)  # this is the mean before subtracting it.
    x0sum_after = x0sum_before
    y0sum_after = y0sum_before
    xconst = 0
    yconst = 0
    if remove_mean:
        # determine mean and sum after
        if symmetrize:
            symsum = 0.5*(x0sum_before+y0sum_before)
            xmean = symsum / float(T)
            ymean = xmean
            x0sum_after = x0sum_before - symsum
            y0sum_after = y0sum_before - symsum
        else:
            xmean = x0sum_before/float(T)
            ymean = y0sum_before/float(T)
            x0sum_after = 0
            y0sum_after = 0
        # subtract mean
        if sparse_mode and not working_on_X_copy:
            X0copy = np.ndarray(X0.shape)
            X0 = np.subtract(X0, xmean, X0copy)
        else:
            X0 = covartools.subtract_row(X0, xmean, inplace=modify_data)  # faster
        if sparse_mode and not working_on_Y_copy:
            Y0copy = np.ndarray(Y0.shape)
            Y0 = np.subtract(Y0, ymean, Y0copy)
        else:
            Y0 = covartools.subtract_row(Y0, ymean, inplace=modify_data)

    # COVARIANCE MATRIX
    if sparse_mode:
        if remove_mean:
            xconst = X[0, ~mask_X].astype(dtype)
            yconst = Y[0, ~mask_Y].astype(dtype)
        xsum = _sum_sparse(x0sum_before, mask_X, xconst)
        ysum = _sum_sparse(y0sum_before, mask_Y, yconst)
        C0 = _covar_sparse(X0, mask_X, X0, mask_X,
                           xsum=x0sum_after, xconst=xconst, ysum=x0sum_after, yconst=xconst, symmetrize=False)
        Ct = _covar_sparse(X0, mask_X, Y0, mask_Y,
                           xsum=x0sum_after, xconst=xconst, ysum=y0sum_after, yconst=yconst, symmetrize=symmetrize)
    else:
        xsum = x0sum_before
        ysum = y0sum_before
        C0 = _covar_dense(X0, X0, symmetrize=False)
        Ct = _covar_dense(X0, Y0, symmetrize=symmetrize)

    return xsum, ysum, C0, Ct


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


class MomentsStorage(object):
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


class RunningCovar(object):
    """
    """

    def __init__(self, compute_XX=True, compute_XY=False, compute_YY=False,
                 remove_mean=False, symmetrize=False,
                 nsave=5):
        # check input
        if not compute_XX and not compute_XY:
            raise ValueError('One of compute_XX or compute_XY must be True.')
        if symmetrize and not compute_XY:
            warnings.warn('symmetrize=True has no effect with compute_XY=False.')
        # storage
        self.compute_XX = compute_XX
        if compute_XX:
            self.storage_XX = MomentsStorage(nsave)
        self.compute_XY = compute_XY
        if compute_XY:
            self.storage_XY = MomentsStorage(nsave)
        self.compute_YY = compute_YY
        if compute_YY:
            raise NotImplementedError('Currently not implemented')
        # symmetry
        self.remove_mean = remove_mean
        self.symmetrize = symmetrize


    def add(self, X, Y=None):
        # check input
        T = X.shape[0]
        if Y is not None:
            assert Y.shape[0] == T, 'X and Y must have equal length'
        # estimate and add to storage
        if self.compute_XX and not self.compute_XY:
            s_X, C_XX = moments_XX(X, remove_mean=self.remove_mean)
            self.storage_XX.store(Moments(T, s_X, C_XX))
        elif self.compute_XX and self.compute_XY:
            assert Y is not None
            s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=self.remove_mean, symmetrize=self.symmetrize)
            self.storage_XX.store(Moments(T, s_X, C_XX))
            self.storage_XY.store(Moments(T, s_X, C_XY))

    def sum_X(self):
        return self.storage_XX.moments.s

    def mean_X(self):
        return self.storage_XX.moments.mean

    def moments_XX(self):
        return self.storage_XX.moments.M

    def cov_XX(self):
        return self.storage_XX.moments.covar

    def moments_XY(self):
        return self.storage_XY.moments.M

    def cov_XY(self):
        return self.storage_XY.moments.covar

    def sum_Y(self):
        raise NotImplementedError('Currently not implemented')

    def moments_YY(self):
        raise NotImplementedError('Currently not implemented')


