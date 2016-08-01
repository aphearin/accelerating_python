""" Serial and parallel python implementations of a function
calculating the pairwise sum of the elements of two arrays.
"""
import numpy as np
import multiprocessing
from parallel_helpers import parallelization_indices
from functools import partial

__all__ = ('serial_pairwise_sum_python', 'parallel_pairwise_sum_python')


def serial_pairwise_sum_python(arr1, arr2, arr1_loop_indices=None):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Implementation is serial and in pure python.

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
    npts1, npts2 = len(arr1), len(arr2)
    result = np.zeros(npts1*npts2)

    if arr1_loop_indices is None:
        first_idx, last_idx = 0, npts1
    else:
        first_idx, last_idx = arr1_loop_indices

    for i in range(first_idx, last_idx):
        x = arr1[i]
        for j in range(npts2):
            y = arr2[j]
            idx_result = i*npts2 + j
            result[idx_result] = x + y

    return result


def parallel_pairwise_sum_python(arr1, arr2, num_threads='max'):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Implementation is pure python and parallelized using multiprocessing.pool.map

    Parameters
    -----------
    arr1 : array_like
        1-d array storing *npts1* floats

    arr2 : array_like
        1-d array storing *npts2* floats

    num_threads : int, optional
        Number of independent processing units to use in the calculation.
        Default behavior is to use all available cores.

    Returns
    -------
    result : array
        1-d array storing *npts1 x npts2* floats determined by the
        pairwise sum of the input ``arr1`` and ``arr2``.

        Element *k* of ``result`` equals *arr1[i] + arr2[j]*, where
        *i = k // npts2* and *j = k % npts2*.

    """
    npts1 = len(arr1)

    if num_threads == 'max':
        num_threads = multiprocessing.cpu_count()
    num_threads, list_of_tuples = parallelization_indices(npts1, num_threads)

    # Create a function object that has a single argument, for parallelization purposes
    engine = partial(serial_pairwise_sum_python, arr1, arr2)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        partial_result = pool.map(engine, list_of_tuples)
        result = np.sum(np.array(partial_result), axis=0)
        pool.close()
    else:
        result = engine(list_of_tuples[0])

    return np.array(result)
