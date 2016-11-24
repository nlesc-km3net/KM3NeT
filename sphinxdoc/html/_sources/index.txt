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
   internal

Introduction
============

The software for the real-time detection of neutrinos in the new generation neutrino telescope.

This software package covers the implementation of a pipeline that
analyzes the hits that come out of the detector. The algorithm looks
for correlated hits and analyzes them to detect neutrinos passing
through the detector.

Installation
------------
Installation is currently not required, if needed instructions will follow later.


Dependencies
------------
 * Python 3 
 * PyCuda (https://mathema.tician.de/software/)
 * Pandas
 * Scipy
 * Numpy
 * Sphinx
 * Sphinx readthedocs theme

Dependencies can be installed with `pip install -r requirements.txt`.

Example usage
-------------
Example on how to use the software will go here.


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

