## Accelerating python with cython

This repository contains a simple but non-trivial example of how to speed up a python calculation using a combination of cython and the multiprocessing module. 

The basic calculation is to compute the pairwise sum of all elements of two input arrays. There is a python implementation and a cython implementation, each of which can be run in serial or in parallel. To use the cython function, the code must first be compiled:

$ python setup.py build_ext --inplace

The relative performance of the python vs. cython implementation can be tested by calling the example script:

$ python example_script.py

To run the testing suite: 

$ py.test

To call the pairwise functions from a python session:

```
npts = 1000
x, y = np.arange(npts), np.arange(npts)

from pairwise_python import serial_pairwise_sum_python
serial_python_result = serial_pairwise_sum_python(x, y)

from pairwise_sum_cython import pairwise_sum_cython
serial_cython_result = pairwise_sum_cython(x, y)
parallel_cython_result = pairwise_sum_cython(x, y, num_threads=2)

```