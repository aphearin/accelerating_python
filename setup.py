""" Module providing Cython compilation instructions for pairwise_sum_cython.pyx.
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(['pairwise_sum_cython_engine.pyx', 'simple_pairwise_sum_cython_engine.pyx']),
    include_dirs=[numpy.get_include()])

# compile instructions:
# python setup.py build_ext --inplace
