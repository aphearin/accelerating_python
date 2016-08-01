""" Executable script used to demonstrate performance benefits of using cython
for calculation of pairwise sums.
"""
import numpy as np
from time import time
from pairwise_python import serial_pairwise_sum_python

try:
    from pairwise_sum_cython_engine import pairwise_sum_cython_engine
except ImportError:
    msg = "The cython module must be compiled first via ``python setup.py build_ext --inplace``"
    raise ImportError(msg)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-npts", help="Number of elements in the dummy input arrays x and y",
    default=int(1e3), type=int)
args = parser.parse_args()

x, y = np.arange(args.npts), np.arange(args.npts)

start = time()
serial_python_result = serial_pairwise_sum_python(x, y)
end = time()
print("\n\nTotal runtime for serial pairwise_sum_python = {0:.1f} ms".format((end-start)*1000.))

start = time()
serial_cython_result = pairwise_sum_cython_engine(x, y)
end = time()
print("Total runtime for serial pairwise_sum_cython = {0:.1f} ms\n\n".format((end-start)*1000.))

error_msg = "Error: python and cython modules do not agree!\a"
assert np.all(serial_python_result == serial_cython_result), error_msg
