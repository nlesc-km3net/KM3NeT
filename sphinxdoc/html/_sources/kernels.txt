.. toctree::
   :maxdepth: 2


Kernels documentation
=====================

This page shows how to call the GPU kernels from Python.

To simplify the process of using the GPU kernels we've created
a couple of Python classes that wrap the GPU kernels and take care
compiling the kernels and managing GPU memory.

Currently the kernels module consists of three classes: QuadraticDifferenceSparse, Match3BSparse, and PurgingSparse.
QuadraticDifferenceSparse and Match3BSparse take as input the hits in a timeslice and produce a sparsely stored correlation matrix.
PurgingSparse processes the correlation matrix to approximate the largest group of hits that are all correlated with each other.


km3net.kernels.QuadraticDifferenceSparse
----------------------------------------
.. autoclass:: km3net.kernels.QuadraticDifferenceSparse
    :members:

km3net.kernels.Match3BSparse
----------------------------------------
.. autoclass:: km3net.kernels.Match3BSparse
    :members:

km3net.kernels.PurgingSparse
----------------------------
.. autoclass:: km3net.kernels.PurgingSparse
    :members:
