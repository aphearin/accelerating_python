""" Python function calling an underlying cython engine
calculating the pairwise sum of the elements of two arrays.
"""
try:
    from simple_pairwise_sum_cython_engine import simple_pairwise_sum_cython_engine
except ImportError:
    msg = ("The cython module must be compiled first via "
        "``python setup.py build_ext --inplace``")
    raise ImportError(msg)

__all__ = ('simple_pairwise_sum_cython', )


def simple_pairwise_sum_cython(arr1, arr2):
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

    """

    # Catch input errors within python layer to make exception handling easier
    try:
        npts1 = len(arr1)
        npts2 = len(arr2)
        assert npts1 > 1
        assert npts2 > 1
    except TypeError:
        msg = "Input ``arr1`` and ``arr2`` must be arrays"
        raise TypeError(msg)
    except AssertionError:
        msg = "Input ``arr1`` and ``arr2`` must have more than one element"
        raise ValueError(msg)

    # Call the underlying cython kernel
    result = simple_pairwise_sum_cython_engine(arr1, arr2)

    return result
