""" Module storing the kernel of the cythonized calculation of pairwise summation.
"""
import numpy as np # use import to get numpy functions
cimport numpy as cnp # use cimport to get numpy types, naming as cnp for clarity

cimport cython # only necessary for the performance-enhancing decorators

__all__ = ('simple_pairwise_sum_cython_engine', )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def simple_pairwise_sum_cython_engine(arr1, arr2):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Parameters
    -----------
    arr1 : array_like
        1-d array storing *npts1* floats

    arr2 : array_like
        1-d array storing *npts2* floats

    Returns
    -------
    result : array
        1-d array storing *npts1 x npts2* floats determined by the
        pairwise sum of the input ``arr1`` and ``arr2``.

        Element *k* of ``result`` equals *arr1[i] + arr2[j]*, where
        *i = k // npts2* and *j = k % npts2*.

    """
    cdef int npts1 = len(arr1)
    cdef int npts2 = len(arr2)

    cdef cnp.float64_t[:] result = np.zeros(npts1*npts2, dtype=np.float64)

    cdef cnp.float64_t[:] arr1_view = np.ascontiguousarray(arr1, dtype=np.float64)
    cdef cnp.float64_t[:] arr2_view = np.ascontiguousarray(arr2, dtype=np.float64)

    cdef int i, j, idx_result
    cdef cnp.float64_t x, y

    for i in range(0, npts1):
        x = arr1_view[i]

        for j in range(npts2):
            y = arr2_view[j]

            idx_result = i*npts2 + j
            result[idx_result] = x + y

    return np.array(result)
