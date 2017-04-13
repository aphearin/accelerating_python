""" Unit-testing module for python implementation of pairwise sum calculation
"""
import numpy as np

from pairwise_sum_python import pairwise_sum_python

__all__ = ('test_python_serial1', )


def test_python_serial1():
    """ Hard-coded test that serial python function is correct.
    """
    x = np.arange(1, 5)
    y = np.zeros(7)
    z = pairwise_sum_python(x, y)
    assert np.all(z == np.repeat(x, len(y)))


def test_python_serial2():
    """ Hard-coded test that serial python function is correct.
    """
    x = (1, 2, 3)
    y = (4, 5, 6)
    z = pairwise_sum_python(x, y)
    assert np.all(z == (5, 6, 7, 6, 7, 8, 7, 8, 9))
