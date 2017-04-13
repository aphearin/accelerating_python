""" Serial python implementation of a function
calculating the pairwise sum of the elements of two arrays.
"""
import numpy as np

__all__ = ('pairwise_sum_python', )


def pairwise_sum_python(arr1, arr2):
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

    Examples
    ----------
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6, 7]
    >>> result = pairwise_sum_python(x, y)
    """
    npts1, npts2 = len(arr1), len(arr2)
    result = np.zeros(npts1*npts2)

    for i in range(0, npts1):
        x = arr1[i]

        for j in range(npts2):
            y = arr2[j]

            idx_result = i*npts2 + j
            result[idx_result] = x + y

    return result
