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

__author__ = 'noe'

import math, sys, numbers
import numpy as np
from variational.estimators.covar_c import covartools
import warnings


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
        # TODO: this choice is still a bit pessimistic.
        # TODO: In cases in which we have to copy the data anyway, we can probably use more sparsity.
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
            xconst = None
            mask = None
    else:
        xconst = None
        mask = None

    return X, mask, xconst  # None, 0 if not sparse


def _copy_convert(X, const=None, copy=True):
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
    const : ndarray or None
        copy or reference to const if no copy was needed.

    """
    # determine type
    dtype = np.float64  # default: convert to float64 in order to avoid cancellation errors
    if X.dtype.kind == 'b' and X.shape[0] < 2**23:
        dtype = np.float32  # convert to float32 if we can represent all numbers
    # copy/convert if needed
    if X.dtype not in (np.float64, dtype):  # leave as float64 (conversion is expensive), otherwise convert to dtype
        X = X.astype(dtype, order='C')
        if const is not None:
            const = const.astype(dtype, order='C')
    elif copy:
        X = X.copy(order='C')
        if const is not None:
            const = const.copy(order='C')

    return X, const


def _sum_sparse(xsum, mask_X, xconst, T):
    s = np.zeros(len(mask_X))
    s[mask_X] = xsum
    s[~mask_X] = T * xconst
    return s


def _sum(X, xmask=None, xconst=None, Y=None, ymask=None, yconst=None, symmetric=False, remove_mean=False):
    """ Computes the column sums and centered column sums.

    If symmetric = False, the sums will be determined as
    .. math:
        sx &=& \frac{1}{2} \sum_t x_t
        sy &=& \frac{1}{2} \sum_t y_t

    If symmetric, the sums will be determined as

    .. math:
        sx = sy = \frac{1}{2T} \sum_t x_t + y_t

    Returns
    -------
    sx : ndarray
        effective row sum of X (including symmetrization if requested)
    sx_raw_centered : ndarray
        centered raw row sum of X

    optional returns (only if Y is given):

    sy : ndarray
        effective row sum of X (including symmetrization if requested)
    sy_raw_centered : ndarray
        centered raw row sum of Y

    """
    T = X.shape[0]
    # compute raw sums on variable data
    sx_raw = X.sum(axis=0)  # this is the mean before subtracting it.
    sy_raw = 0
    if Y is not None:
        sy_raw = Y.sum(axis=0)

    # expand raw sums to full data
    if xmask is not None:
        sx_raw = _sum_sparse(sx_raw, xmask, xconst, T)
    if ymask is not None:
        sy_raw = _sum_sparse(sy_raw, ymask, yconst, T)

    # compute effective sums and centered sums
    if Y is not None and symmetric:
        sx = 0.5*(sx_raw + sy_raw)
        sy = sx
    else:
        sx = sx_raw
        sy = sy_raw

    sx_raw_centered = sx_raw
    sy_raw_centered = sy_raw

    # center mean
    if remove_mean:
        if Y is not None and symmetric:
            sx_raw_centered -= sx
            sy_raw_centered -= sy
        else:
            sx_raw_centered = np.zeros(sx.size)
            if Y is not None:
                sy_raw_centered = np.zeros(sy.size)

    # return
    if Y is not None:
        return sx, sx_raw_centered, sy, sy_raw_centered
    else:
        return sx, sx_raw_centered


def _center(X, s, mask=None, const=None, inplace=True):
    """ Centers the data.

    Parameters
    ----------
    inplace : bool
        center in place

    Returns
    -------
    sx : ndarray
        uncentered row sum of X
    sx_centered : ndarray
        row sum of X after centering

    optional returns (only if Y is given):

    sy_raw : ndarray
        uncentered row sum of Y
    sy_centered : ndarray
        row sum of Y after centering

    """
    T = X.shape[0]
    xmean = s / float(T)
    if mask is None:
        X = covartools.subtract_row(X, xmean, inplace=inplace)
    else:
        X = covartools.subtract_row(X, xmean[mask], inplace=inplace)
        if inplace:
            const = np.subtract(const, xmean[~mask], const)
        else:
            const = np.subtract(const, xmean[~mask])

    return X, const


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
    if x is None:
        return True
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
        if symmetrize:
            Coff = 0.5*(np.outer(xsum, yconst) + np.outer(ysum, xconst))
            C[np.ix_(mask_X, ~mask_Y)] = Coff
            C[np.ix_(~mask_X, mask_Y)] = Coff.T
        else:
            C[np.ix_(mask_X, ~mask_Y)] = np.outer(xsum, yconst)
            C[np.ix_(~mask_X, mask_Y)] = np.outer(xconst, ysum)
    # Block 22
    if not (xconst_is_0 or yconst_is_0):
        if symmetrize:  # 0.5 T (c d' + d c')
            C22 = np.outer(0.5*Xvar.shape[0]*xconst, yconst) + np.outer(0.5*Xvar.shape[0]*yconst, xconst)
            C[np.ix_(~mask_X, ~mask_Y)] = C22
        else:  # T c d'
            C22 = np.outer(Xvar.shape[0]*xconst, yconst)
            C[np.ix_(~mask_X, ~mask_Y)] = C22
    return C


def M2_dense(X, Y):
    """ 2nd moment matrix using dense matrix computations. """
    pass

def M2_const(X, Y):
    """ 2nd moment matrix exploiting constant input columns """
    pass

def M2_sparse(X, Y):
    """ 2nd moment matrix exploiting zero input columns """
    pass

# TODO: maybe combine subsequent two functions?
def M2_sparse_self_sym(X, Y):
    """ 2nd self-symmetric moment matrix exploiting zero input columns

    Computes X'X + Y'Y

    """
    pass

def M2_sparse_other_sym(X, Y):
    """ 2nd self-symmetric moment matrix exploiting zero input columns

    Computes X'Y + Y'X

    """
    pass

#def M2()


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
    # sparsify
    X0, mask_X, xconst = _sparsify(X, sparse_mode=sparse_mode, sparse_tol=sparse_tol)
    # copy / convert
    # TODO: do we need to copy xconst?
    X0, xconst = _copy_convert(X0, const=xconst, copy=sparse_mode or (remove_mean and not modify_data))
    # sum / center
    # TODO: confusing. now sx and sx_centered are on full space
    sx, sx_centered = _sum(X0, xmask=mask_X, xconst=xconst, symmetric=False, remove_mean=remove_mean)
    if remove_mean:
        _center(X0, sx, mask=mask_X, const=xconst, inplace=True)  # fast in-place centering
    # compute covariance matrix
    if mask_X is None:  # dense
        C = np.dot(X0.T, X0)
    else:
        # TODO: using centered data only on X0 would be a bit nicer
        C = _covar_sparse(X0, mask_X, X0, mask_X,
                          xsum=sx_centered[mask_X], xconst=xconst,
                          ysum=sx_centered[mask_X], yconst=xconst)

    return sx, C


def moments_XXXY(X, Y, remove_mean=False, modify_data=False, symmetrize=False,
                 sparse_mode='auto', sparse_tol=0.0):
    """ Computes the first two unnormalized moments of X and Y

    If symmetrize is False, computes

    .. math:
        s_x  &=& \sum_t x_t
        s_y  &=& \sum_t y_t
        C_XX &=& X^\top X
        C_XY &=& X^\top Y

    If symmetrize is True, computes

    .. math:
        s_x = s_y &=& \frac{1}{2} \sum_t(x_t + y_t)
        C_XX      &=& \frac{1}{2} (X^\top X + Y^\top Y)
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
    X0, xconst = _copy_convert(X0, const=xconst, copy=copy)
    Y0, yconst = _copy_convert(Y0, const=yconst, copy=copy)
    # sum / center
    sx, sx_centered, sy, sy_centered = _sum(X0, xmask=mask_X, xconst=xconst, Y=Y0, ymask=mask_Y, yconst=yconst,
                                            symmetric=symmetrize, remove_mean=remove_mean)
    if remove_mean:
        _center(X0, sx, mask=mask_X, const=xconst, inplace=True)  # fast in-place centering
        _center(Y0, sy, mask=mask_Y, const=yconst, inplace=True)  # fast in-place centering

    # compute covariance matrix
    if mask_X is None:  # dense
        Cxx = _covar_dense(X0, X0, symmetrize=False)
        if symmetrize:
            Cxx = 0.5 * (Cxx + _covar_dense(Y0, Y0, symmetrize=False))
        Cxy = _covar_dense(X0, Y0, symmetrize=symmetrize)
    else:
        Cxx = _covar_sparse(X0, mask_X, X0, mask_X, xsum=sx_centered[mask_X], xconst=xconst,
                           ysum=sx_centered[mask_X], yconst=xconst, symmetrize=False)
        if symmetrize:
            Cxx = 0.5 * (Cxx + _covar_sparse(Y0, mask_Y, Y0, mask_Y, xsum=sy_centered[mask_Y], xconst=yconst,
                                           ysum=sy_centered[mask_Y], yconst=yconst, symmetrize=False))
        Cxy = _covar_sparse(X0, mask_X, Y0, mask_Y, xsum=sx_centered[mask_X], xconst=xconst,
                           ysum=sy_centered[mask_Y], yconst=yconst, symmetrize=symmetrize)

    return sx, sy, Cxx, Cxy
