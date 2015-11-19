"""

Data Types
----------
The standard data type for covariance computations is
float64, because the double precision (but not single precision) is
usually sufficient to compute the long sums involved in covariance
matrix computations. Integer types are avoided even if the data is integer,
because the BLAS matrix multiplication is very fast with floats, but very
slow with integers. If X is of boolean type (0/1), the standard data type
is float32, because this will be sufficient to represent numbers up to 2^23
without rounding error, which is usually sufficient sufficient as the
largest element in np.dot(X.T, X) can then be T, the number of data points.


Sparsification
--------------
We aim at computing covariance matrices. For large (T x N) data matrices X, Y,
the bottleneck of this operation is computing the matrix product np.dot(X.T, X),
or np.dot(X.T, Y), with algorithmic complexity O(N^2 T). If X, Y have zero or
constant columns, we can reduce N and thus reduce the algorithmic complexity.

However, the BLAS matrix product used by np.dot() is highly Cache optimized -
the data is accessed in a way that most operations are done in cache, making the
calculation extremely efficient. Thus, even if X, Y have zero or constant columns,
it does not always pay off to interfere with this operation - one one hand by
spending compute time to determine the sparsity of the matrices, one the other
hand by using slicing operations that reduce the algorithmic complexity, but may
destroy the order of the data and thus produce more cache failures.

In order to make an informed decision, we have compared the runtime of the following
operations using matrices of various different sizes (T x N) and different degrees
of sparsity. (using an Intel Core i7 with OS/X 10.10.1):

    1. Compute np.dot(X.T, X)
    2. Compute np.dot(X[:, sel].T, X[:, sel]) where sel selects the nonzero columns
    3. Make a copy X0 = X[:, sel].copy() and then compute np.dot(X0.T, X0)

It may seem that step 3 is not a good idea because we make the extra effort of
copying the matrix. However, the new copy will have data ordered sequentially in
memory, and therefore better prepared for the algorithmically more expensive but
cache-optimized matrix product.

We have empirically found that:

    * Making a copy before running np.dot (option 3) is in most cases better than
      using the dot product on sliced arrays (option 2). Exceptions are when the
      data is extremely sparse, such that only a few columns are selected.
    * Copying and subselecting columns (option 3) is only faster than the full
      dot product (option 1), if 50% or less columns are selected. This observation
      is roughly independent of N.
    * The observations above are valid for  matrices (T x N) that are sufficiently
      large. We assume that "sufficiently large" means that they don't fully fit
      in the cache. For small matrices, the trends are less clear and different
      rules may apply.

In order to optimize covariance calculation for large matrices, we therefore
take the following actions:

    1. Given matrix size of X (and Y), determine the minimum number of columns
       that need to be constant in order to use sparse computation.
    2. Efficiently determine sparsity of X (and Y). Give up as soon as the
       number of constant column candidates drops below the minimum number, to
       avoid wasting time on the decision.
    3. Subselect the desired columns and copy the data to a new array X0 (Y0).
    4. Run operation on the new array X0 (Y0), including in-place substraction
       of the mean if needed.

"""
from __future__ import absolute_import

__author__ = 'noe'

import math, sys, numbers
import numpy as np
from variational.estimators.covar_c import covartools
import warnings


# TODO: this choice is still a bit pessimistic. In cases in which we have to copy the data anyway,
#       we can probably use more sparsity.
def _sparsify(X, sparse_mode='auto', sparse_tol=0.0):
    """ Determines the sparsity of X and returns a selected sub-matrix

    Only conducts sparsification if the number of constant columns is at least
    max(a N - b, min_const_col_number),

    Parameters
    ----------
    X : ndarray
        data matrix
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic

    Returns
    -------
    X0 : ndarray (view of X)
        Either X itself (if not sufficiently sparse), or a sliced view of X,
        containing only the variable columns
    mask : ndarray(N, dtype=bool) or None
        Bool selection array that indicates which columns of X were selected for
        X0, i.e. X0 = X[:, mask]. mask is None if no sparse selection was made.
    xconst : ndarray(N)
        Constant column values that are outside the sparse selection, i.e.
        X[i, ~mask] = xconst for any row i. xconst=0 if no sparse selection was made.

    """
    if sparse_mode.lower() == 'sparse':
        min_const_col_number = 0  # enforce sparsity. A single constant column will lead to sparse treatment
    elif sparse_mode.lower() == 'dense':
        min_const_col_number = X.shape[1] + 1  # never use sparsity
    else:
        # This is a rough heuristic to choose a minimum column number for which sparsity may pay off.
        # Note: this has been determined for a Intel i7 with MacOS, and may be different for different
        # CPUs / OSes. Moreover this heuristic is good for large number of samples, i.e. it may be
        # inadequate for small matrices X.
        if X.shape[1] < 250:
            min_const_col_number = 0.25 * X.shape[1]
        elif X.shape[1] < 1000:
            min_const_col_number = 0.5 * X.shape[1] - 100
        else:
            min_const_col_number = 0.5 * X.shape[1] - 400

    if X.shape[1] > min_const_col_number:
        mask = covartools.variable_cols(X, tol=sparse_tol, min_constant=min_const_col_number)  # bool vector
        nconst = len(np.where(~mask)[0])
        if nconst > min_const_col_number:
            xconst = X[0, ~mask]
            X = X[:, mask]  # sparsify
        else:
            xconst = 0
            mask = None
    else:
        xconst = 0
        mask = None

    return X, mask, xconst  # None, 0 if not sparse


def _copy_convert(X, copy=True):
    """ Makes a copy or converts the data type if needed

    Copies the data and converts the data type if unsuitable for covariance
    calculation. The standard data type for covariance computations is
    float64, because the double precision (but not single precision) is
    usually sufficient to compute the long sums involved in covariance
    matrix computations. Integer types are avoided even if the data is integer,
    because the BLAS matrix multiplication is very fast with floats, but very
    slow with integers. If X is of boolean type (0/1), the standard data type
    is float32, because this will be sufficient to represent numbers up to 2^23
    without rounding error, which is usually sufficient sufficient as the
    largest element in np.dot(X.T, X) can then be T, the number of data points.

    Parameters
    ----------
    copy : bool
        If True, enforces a copy even if the data type doesn't require it.

    Return
    ------
    X : ndarray
        copy or reference to X if no copy was needed.

    """
    # determine type
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    if X.dtype.kind == 'b' and X.shape[0] < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    # copy/convert if needed
    if X.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        X = X.astype(dtype, order='C')
    elif copy:
        X = X.copy(order='C')

    return X


def _sum_center(X, Y=None, symmetric=False, remove_mean=True, inplace=True):
    """ Sums over rows and centers the data if requested.

    Returns
    -------
    X : ndarray
        X data array, possibly centered
    sx_raw : ndarray
        raw row sum of X
    sx_centered : ndarray
        row sum of X after centering

    optional returns (only if Y is given):

    Y : ndarray
        Y data array, possibly centered
    sy_raw : ndarray
        raw row sum of Y
    sy_centered : ndarray
        row sum of Y after centering

    """
    T = X.shape[0]
    # compute raw sums
    sx_raw = X.sum(axis=0)  # this is the mean before subtracting it.
    sy_raw = 0
    if Y is not None:
        sy_raw = Y.sum(axis=0)
    sx_centered = sx_raw
    sy_centered = sx_raw

    # remove mean if desired
    if remove_mean:
        # determine sum for shifting
        sx = sx_raw
        sy = sy_raw
        if symmetric:
            sx = 0.5*(sx_raw + sy_raw)
            sy = sx

        # fast in-place subtraction, because at this point we have a copy if we want or need one
        X = covartools.subtract_row(X, sx/float(T), inplace=inplace)
        if Y is not None:
            Y = covartools.subtract_row(Y, sy/float(T), inplace=inplace)
        # shift sum after
        if Y is not None and symmetric:  # this is the only case where sx_after, sy_after can be nonzero
            sx_centered = sx_raw - sx
            sy_centered = sy_raw - sy
        else:
            sx_centered = 0
            sy_centered = 0

    if Y is None:
        return X, sx_raw, sx_centered
    else:
        return X, sx_raw, sx_centered, Y, sy_raw, sy_centered


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
    # check input
    if symmetrize and len(mask_X) != len(mask_Y):
        raise ValueError('Cannot compute symmetric covariance matrix for differently sized data')
    C = np.zeros((len(mask_X), len(mask_Y)))
    # Block 11
    C11 = np.dot(Xvar.T, Yvar)
    if symmetrize:
        C[np.ix_(mask_X, mask_Y)] = 0.5*(C11 + C11.T)[:, :]
    else:
        C[np.ix_(mask_X, mask_Y)] = C11[:, :]
    # other blocks
    xsum_is_0 = _is_zero(xsum)
    ysum_is_0 = _is_zero(ysum)
    xconst_is_0 = _is_zero(xconst)
    yconst_is_0 = _is_zero(yconst)
    # Block 12 and 21
    if not (xsum_is_0 or yconst_is_0) or not (ysum_is_0 or xconst_is_0):
        C12 = np.outer(xsum, yconst)
        C21 = np.outer(xconst, ysum)
        if symmetrize:
            Coff = 0.5*(C12 + C21.T)
            C[np.ix_(mask_X, ~mask_Y)] = Coff
            C[np.ix_(~mask_X, mask_Y)] = Coff
        else:
            C[np.ix_(mask_X, ~mask_Y)] = C12
            C[np.ix_(~mask_X, mask_Y)] = C21
    # Block 22
    if not (xconst_is_0 or yconst_is_0):
        C22 = np.outer(xconst, yconst)
        if symmetrize:
            C[np.ix_(~mask_X, ~mask_Y)] = 0.5*(C22 + C22.T)
        else:
            C[np.ix_(~mask_X, ~mask_Y)] = C22
    return C


def _sum_sparse(xsum, mask_X, xconst):
    s = np.zeros(len(mask_X))
    s[mask_X] = xsum
    s[~mask_X] = xconst
    return s


def moments_XX(X, remove_mean=False, modify_data=False, sparse_mode='auto', sparse_tol=0.0):
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
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
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
    # sparsify and copy/convert
    X0, mask_X, xconst = _sparsify(X, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    X0 = _copy_convert(X0, copy=sparse_mode or (remove_mean and not modify_data))
    # fast in-place centering, because at this point we have a copy if we want or need one
    X0, sx_raw, sx_centered = _sum_center(X0, symmetric=False, remove_mean=remove_mean, inplace=True)
    # compute covariance matrix
    if mask_X is None:  # dense
        xsum = sx_raw
        C = np.dot(X0.T, X0)
    else:
        xsum = _sum_sparse(sx_raw, mask_X, xconst)
        C = _covar_sparse(X0, mask_X, X0, mask_X, xsum=sx_centered, xconst=xconst, ysum=sx_centered, yconst=xconst)

    return xsum, C


def moments_XXXY(X, Y, remove_mean=False, modify_data=False, symmetrize=False,
                 sparse_mode='auto', sparse_tol=0.0):
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
    sparse_mode : str
        one of:
            * 'dense' : always use dense mode
            * 'sparse' : always use sparse mode if possible
            * 'auto' : automatic
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
    # sparsify
    X0, mask_X, xconst = _sparsify(X, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    Y0, mask_Y, yconst = _sparsify(Y, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    # copy / convert
    copy = sparse_mode or (remove_mean and not modify_data)
    X0 = _copy_convert(X0, copy=copy)
    Y0 = _copy_convert(Y0, copy=copy)
    # fast in-place centering, because at this point we have a copy if we want or need one
    X0, sx_raw, sx_centered = _sum_center(X0, symmetric=False, remove_mean=remove_mean, inplace=True)
    Y0, sy_raw, sy_centered = _sum_center(Y0, symmetric=False, remove_mean=remove_mean, inplace=True)
    # compute covariance matrix
    if mask_X is None:  # dense
        xsum = sx_raw
        ysum = sy_raw
        C0 = _covar_dense(X0, X0, symmetrize=False)
        Ct = _covar_dense(X0, Y0, symmetrize=symmetrize)
    else:
        xsum = _sum_sparse(sx_raw, mask_X, xconst)
        ysum = _sum_sparse(sy_raw, mask_Y, yconst)

        C0 = _covar_sparse(X0, mask_X, X0, mask_X,
                           xsum=sx_centered, xconst=xconst, ysum=sx_centered, yconst=xconst, symmetrize=False)
        Ct = _covar_sparse(X0, mask_X, Y0, mask_Y,
                           xsum=sx_centered, xconst=xconst, ysum=sy_centered, yconst=yconst, symmetrize=symmetrize)

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
        """
        References
        ----------
        [1] http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        """
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
        """
        Parameters
        ----------
        rtol : float
            To decide when to merge two Moments. Ideally I'd like to merge two
            Moments when they have equal weights (i.e. equally many data points
            went into them). If we always add data chunks with equal weights,
            this can be achieved by using a binary tree, i.e. let M1 be the
            moment estimates from one chunk. Two of them are added to M2, Two
            M2 are added to M4, and so on. This way you need to store log2
            (n_chunks) number of Moment estimates.
            In practice you might get data in chunks of unequal length or weight.
            Therefore we need some heuristic when two Moment estimates should get
            merged. This is the role of rtol.

        """
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
