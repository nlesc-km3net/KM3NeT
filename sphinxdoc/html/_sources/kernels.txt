.. toctree::
   :maxdepth: 2


Kernels documentation
=====================

This page shows how to call the GPU kernels from Python.

To simplify the process of using the GPU kernels we've created
two Python classes that wrap the GPU kernels and take care
compiling the kernels and managing GPU memory.

Currently the kernels module consists of two classes: QuadraticDifferenceSparse and PurgingSparse.
QuadraticDifferenceSparse takes as input the hits in a timeslice and produces a sparsely stored correlation matrix.
PurgingSparse processes the correlation matrix to approximate the largest group of hits that are all correlated with eachother.


km3net.kernels.QuadraticDifferenceSparse
----------------------------------------
.. autoclass:: km3net.kernels.QuadraticDifferenceSparse
    :members:

km3net.kernels.PurgingSparse
----------------------------
.. autoclass:: km3net.kernels.PurgingSparse
    :members:
