""" Module storing the kernel of the cythonized calculation of pairwise summation.
"""
import numpy as np # use import to get numpy functions
cimport numpy as cnp # use cimport to get numpy types, renaming as cnp for clarity

cimport cython # only necessary for the performance-enhancing decorators

__all__ = ('simple_pairwise_sum_cython_engine', )


#  The following three decorators enhance the performance of our Cython function
#  These decorators gain performance by sacrificing some of the niceties of python
@cython.boundscheck(False)  # Assume indexing operations will not cause any IndexErrors to be raised
@cython.wraparound(False)  #  Accessing array elements with negative numbers is not permissible
@cython.nonecheck(False)  #  Never waste time checking whether a variable has been set to None
def simple_pairwise_sum_cython_engine(double[:] arr1, double[:] arr2):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Parameters
    -----------
    arr1 : array_like
        1-d array storing *npts1* doubles

        Note that arr1 *must* store doubles, e.g., Numpy dtype='f8',
        or an exception will be raised

    arr2 : array_like
        1-d array storing *npts2* doubles

        Note that arr2 *must* store doubles, e.g., Numpy dtype='f8',
        or an exception will be raised

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

    cdef double[:] result = np.zeros(npts1*npts2, dtype=np.float64)

    cdef int i, j, idx_result
    cdef double x, y

    for i in range(0, npts1):
        x = arr1[i]

        for j in range(npts2):
            y = arr2[j]

            idx_result = i*npts2 + j
            result[idx_result] = x + y

    #  We defined ``result`` as a cython typed memoryview for performance reasons
    #  Typed memoryviews do not support convenient features such as broadcasting,
    #  so we convert to a Numpy array before communicating ``result`` with the outside world
    return np.array(result)
