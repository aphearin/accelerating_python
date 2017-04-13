""" Unit-testing module for parallelization_indices function.
"""
import numpy as np

from parallel_helpers import parallelization_indices

__all__ = ('test_parallelization_indices1', )


def test_parallelization_indices1():
    """
    """
    for_loop_length, num_threads_in = 10, 2

    num_threads_out, list_of_tuples = parallelization_indices(
        for_loop_length, num_threads_in)

    assert num_threads_out == num_threads_in
    assert np.all(list_of_tuples == [(0, 5), (5, 10)])


def test_parallelization_indices2():
    """
    """
    for_loop_length, num_threads_in = 10, 1

    num_threads_out, list_of_tuples = parallelization_indices(
        for_loop_length, num_threads_in)

    assert num_threads_out == num_threads_in
    assert np.all(list_of_tuples == [(0, 10)])


def test_parallelization_indices3():
    """
    """
    for_loop_length, num_threads_in = 10, 3

    num_threads_out, list_of_tuples = parallelization_indices(
        for_loop_length, num_threads_in)

    assert num_threads_out == num_threads_in
    assert np.all(list_of_tuples == [(0, 4), (4, 7), (7, 10)])


def test_parallelization_indices4():
    """
    """
    for_loop_length, num_threads_in = 3, 4

    num_threads_out, list_of_tuples = parallelization_indices(
        for_loop_length, num_threads_in)

    assert num_threads_out == for_loop_length
    assert np.all(list_of_tuples == [(0, 1), (1, 2), (2, 3)])
