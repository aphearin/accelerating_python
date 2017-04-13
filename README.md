# Accelerating Python with Cython

The purpose of this repository is to provide a few worked examples of how to speed up your python code using Cython. The approach is to take a simple but non-trivial calculation of the pairwise sum of two arrays,  and just demonstrate some of the tricks involved in writing cythonized code. 

This repository is intended for people with basic knowledge of python, but no prior experience with Cython. I make no attempt whatsoever at being comprehensive. In fact, here I do the opposite: I give something that can be read quickly to help get your feet moving. 

## Getting started 

Cython is a tool that transforms your python code into compiled C. This means there is an extra step involved between code development and code use: you must compile your cython code before you can call it from python. 

This repository demonstrates two different ways to compile Cython code: first using a Jupyter Notebook, and second using a `setup.py`. Using a `setup.py` file is more powerful, since it makes it easier to integrate your Cython code throughout your python modules, but using a Notebook is simpler for quick prototyping. To get started, we'll use the Notebook way of doing things since compiling Cython in a notebook is so easy, this will let us focus straight away on how to write cython in the next section. In the subsequent section, we'll return show the `setup.py` way of doing things. 

Either way, make sure you `pip install cython` before moving on. 

These notes provide some flesh to the highly recommended series of blog posts on Numba vs. Cython written by Jake VanderPlas. 

## Basic Cython Syntax

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

## Writing Cython Modules

At some point, you'll want to use some cython code coherently with your python modules. To do that, it can be helpful to move your cython functions out of notebooks and into dedicated modules. This section describes a simple pattern to match for this purpose. Note, however, that there are many ways to skin this cat, and that you can find plenty of documentation online describing alternative approaches.

Open up `pairwise_sum_cython.py` and `pairwise_sum_cython_engine.pyx` with a text editor. The `pairwise_sum_cython` function defined in the `pairwise_sum_cython.py` module is the function the "outside world" will interact with. The `pairwise_sum_cython` function is just ordinary python code like you are used to. It just so happens that the `pairwise_sum_cython` function calls another function, `pairwise_sum_cython_engine`, that is written in Cython. 

This illustrates a very useful design pattern: use python to write code that faces the "outside world", and only use Cython to handle the tiny little expensive operation that dominates the runtime. This way, you can do all your normal bounds-checking and error-handling in pure python, and then only pass in to Cython a set of arguments that you are confident it can properly handle. 

This leaves us with the need to actually compile the code that gets written in a Cython module. The code responsible for that is written in `setup.py`. This is utterly boilerplate code that can be copied-and-pasted over and over again. This repo makes no attempt at all to explain this machinery - see the online documentation if you are curious. The only thing that will change from user to user is the filename of the Cython module. If you have more than one cython module, just add an additional string to the list that currently contains `['pairwise_sum_cython_engine.pyx']`, e.g., `['pairwise_sum_cython_engine.pyx', 'another_module.pyx']`

Once you have written your `setup.py` file, you can compile it as follows:

$ python setup.py build_ext --inplace

Now you can import and use the `pairwise_sum_cython` module from anywhere in your python code. 

To see an example of how you can call your cythonized python function from within another python module, have a look at `example_script.py`, or just run it:

```
$ python example_script.py
```

