"""
"""
import numpy as np
from pairwise_sum_cython import pairwise_sum_cython
from pairwise_python import serial_pairwise_sum_python

__all__ = ('test_pairwise_sum_cython1', )


def test_pairwise_sum_cython1():
    x = np.arange(-50, 50)
    y = np.arange(-500, 500, 10)

    python_result = serial_pairwise_sum_python(x, y)
    cython_result = pairwise_sum_cython(x, y)
    assert np.all(cython_result == python_result)


def test_pairwise_sum_cython2():
    np.random.seed(43)
    x = np.random.random(100)
    y = np.random.random(200)

    python_result = serial_pairwise_sum_python(x, y)
    cython_result = pairwise_sum_cython(x, y)
    assert np.all(cython_result == python_result)
