""" Python function calling an underlying cython engine
calculating the pairwise sum of the elements of two arrays.
"""
import multiprocessing
from parallel_helpers import parallelization_indices
from functools import partial
import numpy as np

try:
    from pairwise_sum_cython_engine import pairwise_sum_cython_engine
except ImportError:
    msg = "The cython module must be compiled first via ``python setup.py build_ext --inplace``"
    raise ImportError(msg)

__all__ = ('pairwise_sum_cython', )


def pairwise_sum_cython(arr1, arr2, num_threads='max'):
    """ Function calculates the pairwise sum of all elements in arr1 and arr2.

    Implementation is in cython and parallelized using multiprocessing.pool.map.
    This function simply calls the underlying cython kernel.

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
    engine = partial(pairwise_sum_cython_engine, arr1, arr2)

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        partial_result = pool.map(engine, list_of_tuples)
        result = np.sum(np.array(partial_result), axis=0)
        pool.close()
    else:
        result = np.array(engine(list_of_tuples[0]))

    return result
