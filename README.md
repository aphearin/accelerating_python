# Accelerating Python 

The purpose of this repository is to provide a few worked examples of how to speed up your python code in two different ways:

* Using the python multiprocessing module
* Using Cython

The approach is to take a simple but non-trivial calculation, and do it repeatedly in different ways to demonstrate some of the tricks involved in writing cythonized parallel code. This repository is intended for people with basic knowledge of python, but no prior experience with either Cython or multiprocessing. If you are only interested in Cython, and not parallelization, you can simply ignore the sections on multiprocessing, and conversely. 

## Getting started 

The place to start is with the `serial_pairwise_sum_python` function defined in the `pairwise_python.py` module. This function calculates the pairwise sum of all elements of two input arrays. The `serial_pairwise_sum_python ` implementation is serial and pure python; all other functions in the repo are just alternative (faster) implementations, so be sure to understand how this function works before moving on. 


## Writing Cython code

In the vast majority of cases, Cython code is just python code with some type declarations added to make your loops run faster. 


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