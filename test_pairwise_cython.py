"""
"""
import numpy as np
from simple_pairwise_sum_cython import simple_pairwise_sum_cython
from simple_pairwise_sum_python import simple_pairwise_sum_python

__all__ = ('test_pairwise_sum_cython1', )


def test_pairwise_sum_cython1():
    x = np.arange(-50, 50).astype('f8')
    y = np.arange(-500, 500, 10).astype('f8')

    python_result = simple_pairwise_sum_python(x, y)
    cython_result = simple_pairwise_sum_cython(x, y)
    assert np.all(cython_result == python_result)


def test_pairwise_sum_cython2():
    np.random.seed(43)
    x = np.random.random(100)
    y = np.random.random(200)

    python_result = simple_pairwise_sum_python(x, y)
    cython_result = simple_pairwise_sum_cython(x, y)
    assert np.all(cython_result == python_result)


def test_simple_cython():
    np.random.seed(43)
    x = np.random.random(100)
    y = np.random.random(200)
    result1 = simple_pairwise_sum_cython(x, y)
    result2 = simple_pairwise_sum_cython(x, y)
    assert np.all(result1 == result2)
