# Accelerating Python with Cython

The purpose of this repository is to provide a few worked examples of how to speed up your python code using Cython. The approach is to take a simple but non-trivial calculation of the pairwise sum of two arrays,  and just demonstrate some of the tricks involved in writing cythonized code. This repository is intended for people with basic knowledge of python, but no prior experience with Cython. 

## Getting started 

Cython is a tool that transforms your python code into compiled C. This means there is an extra step involved between code development and code use: you must compile your cython code before you can call it from python. 

This repository demonstrates two different ways to compile Cython code: first using a Jupyter Notebook, and second using a `setup.py`. Using a `setup.py` file is more powerful, since it makes it easier to integrate your Cython code throughout your python modules, but using a Notebook is simpler for quick prototyping. To get started, we'll use the Notebook way of doing things since compiling Cython in a notebook is so easy, this will let us focus straight away on how to write cython in the next section. In the following section, we'll return show the `setup.py` way of doing things. 

Either way, make sure you `pip install cython` before moving on. 

## Basics of Writing Cython

If you know some python, you already know most Cython, because Cython is a formal superset of python, so __all valid python is also valid cython.__ 
In the vast majority of cases, Cython code is just python code with some type declarations added to make your loops run faster. 

To see that in action, open up the Jupyter notebook and have a look at the code for computing pairwise sums. The python and cython algorithms are identical: it's just a double for loop over the elements of the two arrays. The main difference is just that in Cython, you can declare the types of the variables used in your loops. This saves the python interpreter from needlessly type-checking your loop variable at each iteration, which can result in dramatic performance enhancements. 

Note that you don't *have* to declare those types. If you delete those lines of code, it still compiles and runs just fine (try it!). But when you do include the type declarations, it just makes your code run faster. 

Floats and integers and longs and doubles in Cython are declared like this:

```
cdef float some_float
cdef int some_int
cdef double some_double
cdef long some_long
```

The modern way to declare an array in Cython is as follows:

```
cdef float[:] some_float_array
```
The fancy term for an array declared in this way is a "Cython typed memoryview". See the `cython_declaration_experimentation.ipynb` notebook for how memoryviews are different from Numpy arrays. 

In many cases, you want to initialize your arrays to have some values. In that case, you will need to take care that the initialized values have a consistent type with the declared type, or Cython will raise an exception:

```
cdef double[:] some_double_array = np.zeros(5, dtype='f8')
cdef float[:] some_float_array = np.zeros(5, dtype='f4')
```

See the `cython_declaration_experimentation.ipynb` notebook to learn a little more about variable declarations, and to play around with things for yourself. 



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

There are in fact many other ways to compile Cython, but these are two of the most common ways, so just refer to the online documentation if you want other options. 
