""" Executable script used to demonstrate performance benefits of using cython
for calculation of pairwise sums.
"""
import numpy as np
from time import time
from pairwise_sum_python import pairwise_sum_python

# Import cython module, catching the case where you forgot to compile the code
try:
    from pairwise_sum_cython import pairwise_sum_cython
except ImportError:
    msg = "The cython module must be compiled first via ``python setup.py build_ext --inplace``"
    raise ImportError(msg)

# Catch optional npts command-line argument using argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-npts", help="Number of elements in the dummy input arrays x and y",
    default=int(2e3), type=int)
args = parser.parse_args()

# run the timing tests and print the results
x, y = np.arange(args.npts).astype('f8'), np.arange(args.npts).astype('f8')

start = time()
serial_python_result = pairwise_sum_python(x, y)
end = time()
print("\n\nTotal runtime for serial pairwise_sum_python = {0:.1f} ms".format((end-start)*1000.))

start = time()
serial_cython_result = pairwise_sum_cython(x, y)
end = time()
print("Total runtime for serial pairwise_sum_cython = {0:.1f} ms\n\n".format((end-start)*1000.))
