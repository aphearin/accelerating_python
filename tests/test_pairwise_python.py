""" Unit-testing module for python implementation of pairwise sum calculation
"""
import numpy as np
import multiprocessing
from time import time

from pairwise_python import serial_pairwise_sum_python, parallel_pairwise_sum_python

__all__ = ('test_python_serial1', )


def test_python_serial1():
    """ Hard-coded test that serial python function is correct.
    """
    x = np.arange(1, 5)
    y = np.zeros(7)
    z = serial_pairwise_sum_python(x, y)
    assert np.all(z == np.repeat(x, len(y)))


def test_python_serial2():
    """ Hard-coded test that serial python function is correct.
    """
    x = (1, 2, 3)
    y = (4, 5, 6)
    z = serial_pairwise_sum_python(x, y)
    assert np.all(z == (5, 6, 7, 6, 7, 8, 7, 8, 9))


def test_parallel_pairwise_sum_python1():
    """ Verify equality of parallel and serial calculations
    """
    x = np.arange(0, 10)
    y = np.arange(-5, 15)
    serial_result = serial_pairwise_sum_python(x, y)

    parallel_result1 = parallel_pairwise_sum_python(x, y, num_threads=2)
    assert np.all(serial_result == parallel_result1)

    parallel_result2 = parallel_pairwise_sum_python(x, y, num_threads=5)
    assert np.all(serial_result == parallel_result2)


def test_parallel_pairwise_sum_python2():
    """ Verify equality of parallel and serial calculations for
    edge case where num_cores exceeds length of output array.
    """
    x = np.arange(0, 3)
    y = np.arange(-5, -2)
    serial_result = serial_pairwise_sum_python(x, y)

    parallel_result = parallel_pairwise_sum_python(x, y, num_threads=20)
    assert np.all(serial_result == parallel_result)


def test_speedup():
    """ Verify that running calculation in parallel actually improves runtime.
    """
    num_cores = multiprocessing.cpu_count()
    if num_cores > 1:
        npts = 1000
        x = np.arange(npts)
        y = np.arange(npts)

        start = time()
        __ = serial_pairwise_sum_python(x, y)
        end = time()
        serial_runtime = end-start

        start = time()
        __ = parallel_pairwise_sum_python(x, y, num_threads=num_cores)
        end = time()
        parallel_runtime = end-start

        assert parallel_runtime < serial_runtime
