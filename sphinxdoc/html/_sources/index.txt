.. km3net documentation master file, created by
   sphinx-quickstart on Tue Mar 29 15:46:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The KM3Net documentation
========================

Contents:

.. toctree::
   :maxdepth: 1

   Introduction <self>
   kernels
   utils

Introduction
============
The software for the real-time detection of neutrinos in the new generation neutrino telescope.

This software package covers the implementation of a pipeline that
analyzes the hits that come out of the detector. The algorithm looks
for correlated hits and analyzes them to detect neutrinos passing
through the detector.


Installation
------------
| Clone the repository  
|     ``git clone git@github.com:benvanwerkhoven/KM3Net.git``  
| Change into the top-level directory  
|     ``cd KM3Net``  
| Install using  
|     ``pip install -r requirements.txt``  
|     ``pip install .``


Dependencies
------------
 * Python 3 (http://conda.pydata.org/miniconda.html)
 * PyCuda (https://mathema.tician.de/software/)
 * Pandas
 * Scipy
 * Numpy
 * Sphinx
 * Sphinx readthedocs theme
 * Kernel Tuner (https://github.com/benvanwerkhoven/kernel_tuner)


Example usage
-------------
Example on how to use the software has been documented in a Jupyter Notebook,
which can be viewed `here <https://github.com/benvanwerkhoven/KM3Net/blob/master/notebooks/Example.ipynb>`_.


Contributors
------------
- Daniela Remenska
- Hanno Spreeuw
- Ben van Werkhoven




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

