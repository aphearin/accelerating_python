""" Module storing the kernel of the cythonized calculation of pairwise summation.

This version of the cython engine permits a parallelized calculation through the
use of the arr1_loop_indices argument.
"""
import numpy as np # use import to get numpy functions
cimport numpy as cnp # use cimport to get numpy types, renaming as cnp for clarity

cimport cython # only necessary for the performance-enhancing decorators

__all__ = ('pairwise_sum_cython_engine', )

#  The following three decorators enhance the performance of our Cython function
#  These decorators gain performance by sacrificing some of the niceties of python
@cython.boundscheck(False)  # Assume indexing operations will not cause any IndexErrors to be raised
@cython.wraparound(False)  #  Accessing array elements with negative numbers is not permissible
@cython.nonecheck(False)  #  Never waste time checking whether a variable has been set to None
def pairwise_sum_cython_engine(arr1, arr2, arr1_loop_indices=None):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Parameters
    -----------
    arr1 : array_like
        1-d array storing *npts1* floats

    arr2 : array_like
        1-d array storing *npts2* floats

    arr1_loop_indices : sequence, optional
        Two-element sequence storing the first and last indices
        of the outermost loop. Default is None, in which case sum will
        be conducted over the entire outer loop will be summed.
        Argument is used only for parallelization purposes and can be ignored
        for serial calculations.

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

    cdef int first_idx, last_idx
    try:
        first_idx, last_idx = arr1_loop_indices
    except TypeError:
        first_idx, last_idx = 0, npts1

    cdef int i, j, idx_result
    cdef cnp.float64_t x, y

    for i in range(first_idx, last_idx):
        x = arr1_view[i]
        for j in range(npts2):
            y = arr2_view[j]
            idx_result = i*npts2 + j
            result[idx_result] = x + y

    return np.array(result)
