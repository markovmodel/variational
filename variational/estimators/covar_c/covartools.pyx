import numpy
import ctypes
cimport numpy

cdef extern from "_covartools.h":
    void _nonconstant_cols_char(int* cols, char* X, int M, int N)

cdef extern from "_covartools.h":
    void _nonconstant_cols_int(int* cols, int* X, int M, int N)

cdef extern from "_covartools.h":
    void _nonconstant_cols_long(int* cols, long* X, int M, int N)

cdef extern from "_covartools.h":
    void _nonconstant_cols_float(int* cols, float* X, int M, int N)

cdef extern from "_covartools.h":
    void _nonconstant_cols_double(int* cols, double* X, int M, int N)

cdef extern from "_covartools.h":
    void _subtract_row_double(double* X, double* row, int M, int N)

cdef extern from "_covartools.h":
    void _subtract_row_float(float* X, float* row, int M, int N)


def nonconstant_cols_char(cols, X, M, N):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <char*> numpy.PyArray_DATA(X)
    _nonconstant_cols_char(pcols, pX, M, N)

def nonconstant_cols_int(cols, X, M, N):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <int*> numpy.PyArray_DATA(X)
    _nonconstant_cols_int(pcols, pX, M, N)

def nonconstant_cols_long(cols, X, M, N):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <long*> numpy.PyArray_DATA(X)
    _nonconstant_cols_long(pcols, pX, M, N)

def nonconstant_cols_float(cols, X, M, N):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <float*> numpy.PyArray_DATA(X)
    _nonconstant_cols_float(pcols, pX, M, N)

def nonconstant_cols_double(cols, X, M, N):
    pcols = <int*> numpy.PyArray_DATA(cols)
    pX = <double*> numpy.PyArray_DATA(X)
    _nonconstant_cols_double(pcols, pX, M, N)


def nonconstant_cols(X):
    """
    Parameters
    ----------
    X : ndarray

    Returns
    -------
    cols : int-array or None
        Indexes of columns with non-constant elements. None means: all columns
        should be considered variable.
    :param X:
    :param min_const:
    :return:
    """
    if X is None:
        return None
    M, N = X.shape

    # prepare column array
    cdef numpy.ndarray[int, ndim=1, mode="c"] path
    cols = numpy.zeros( (N), dtype=ctypes.c_int, order='C' )
    #pcols = <int*> numpy.PyArray_DATA(cols)

    if X.dtype == numpy.float64:
        nonconstant_cols_double(cols, X, M, N)
        #pX = <double*> numpy.PyArray_DATA(X)
        #_nonconstant_cols_double(pcols, pX, M, N)
    elif X.dtype == numpy.float32:
        nonconstant_cols_float(cols, X, M, N)
        #pX = <float*> numpy.PyArray_DATA(X)
        #_nonconstant_cols_float(pcols, pX, M, N)
    elif X.dtype == numpy.int32:
        nonconstant_cols_int(cols, X, M, N)
    elif X.dtype == numpy.int64:
        nonconstant_cols_long(cols, X, M, N)
    elif X.dtype == numpy.bool:
        nonconstant_cols_char(cols, X, M, N)
    else:
        raise TypeError('unsupported type of X: '+str(X.dtype))
    #if X.dtype == numpy.bool:
        #nonconstant_cols_char(pcols, pX, M, N)
    #elif X.dtype == numpy.int32:
    #    pX = <int*> numpy.PyArray_DATA(X)
    #    nonconstant_cols_int(pcols, pX, M, N)
    #elif X.dtype == numpy.float32:
    #    pX = <float*> numpy.PyArray_DATA(X)
    #    nonconstant_cols_float(pcols, pX, M, N)
    #elif X.dtype == numpy.float64:
    #    pX = <double*> numpy.PyArray_DATA(X)
    #    nonconstant_cols_double(pcols, pX, M, N)
    #else:
    #    raise TypeError('unsupported type of X: '+str(X.dtype))
    # return constant cols
    return numpy.array(cols, dtype=numpy.bool)

def subtract_row_float(X, row, M, N):
    prow = <float*> numpy.PyArray_DATA(row)
    pX = <float*> numpy.PyArray_DATA(X)
    _subtract_row_float(pX, prow, M, N)

def subtract_row_double(X, row, M, N):
    prow = <double*> numpy.PyArray_DATA(row)
    pX = <double*> numpy.PyArray_DATA(X)
    _subtract_row_double(pX, prow, M, N)

def subtract_row(X, row):
    M, N = X.shape

    if X.dtype == numpy.float64 and row.dtype == numpy.float64:
        subtract_row_double(X, row, M, N)
    elif X.dtype == numpy.float32 and row.dtype == numpy.float32:
        subtract_row_float(X, row, M, N)
    else:
        raise TypeError('unsupported or inconsistent types: '+str(X.dtype)+' '+str(row.dtype))

