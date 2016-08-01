""" Module containing helper function used to parallelize a for loop
using python multiprocessing.
"""
import numpy as np

__all__ = ('parallelization_indices', )


def parallelization_indices(for_loop_length, num_threads):
    """ Return a list of tuples that will be passed to the
    multiprocessing.pool.map to execute a simple for-loop in parallel.
    Each tuple has two entries storing the first and last
    index that will be looped over in a trivially parallelizable for-loop.

    Parameters
    -----------
    for_loop_length : int
        Total number of elements in the for-loop

    num_threads : int
        Number of cores requested

    Returns
    -------
    num_threads : int
        Number of threads to use when counting pairs. Only differs from the
        input value for the case where the input num_threads > for_loop_length,
        in which case num_threads is used.

    list_of_tuples : list
        List of two-element tuples containing the first and last loop indices

    Notes
    ------
    Care is taken to deal with cases where num_threads and for_loop_length are relatively prime,
    and also to avoid the problem of potentially having more threads available than cells.

    In the serial case where num_threads is 1,
    the returned list_of_tuples is a one-element list containing the tuple (0, for_loop_length).

    If there are two cores available, for example, then function returns a two-element list,
    list_of_tuples = [(0, for_loop_length/2), (for_loop_length/2, for_loop_length)]

    """
    if num_threads == 1:
        return 1, [(0, for_loop_length)]
    elif num_threads > for_loop_length:
        return for_loop_length, [(a, a+1) for a in np.arange(for_loop_length)]
    else:
        list_with_possibly_empty_arrays = np.array_split(np.arange(for_loop_length), num_threads)
        list_of_nonempty_arrays = [a for a in list_with_possibly_empty_arrays if len(a) > 0]
        list_of_tuples = [(x[0], x[0] + len(x)) for x in list_of_nonempty_arrays]
        return num_threads, list_of_tuples
