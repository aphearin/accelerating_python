## Accelerating python with cython

This repository contains a simple but non-trivial example of how to speed up a python calculation using a combination of cython and the multiprocessing module. 

The basic calculation is to compute the pairwise sum of all elements of two input arrays. There is a python implementation and a cython implementation, each of which can be run in serial or in parallel. To use the cython function, the code must first be compiled:

$ python setup.py build_ext --inplace

The relative performance of the python vs. cython implementation can be tested by calling the example script:

$ python example_script.py

To run the testing suite: 

$ py.test

