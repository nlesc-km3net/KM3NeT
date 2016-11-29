
KM3Net - Real-time detection of neutrinos
=========================================

This repository contains the software for the real-time detection of
neutrinos in the new generation neutrino telescope.

This software package covers the implementation of a pipeline that
analyzes the hits that come out of the detector. The algorithm looks
for correlated hits and analyzes them to detect neutrinos passing
through the detector.


Documentation
-------------
Documentation can be found [here](https://benvanwerkhoven.github.io/KM3Net/sphinxdoc/html/index.html).


Installation
------------
Clone the repository
    ``git clone git@github.com:benvanwerkhoven/KM3Net.git``  
Change into the top-level directory  
    ``cd KM3Net``  
Install dependencies
    ``pip install -r requirements.txt``  
    ``pip install .``


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
which can be viewed [here](https://github.com/benvanwerkhoven/KM3Net/blob/master/notebooks/Example.ipynb).


Contributors
------------

* Daniela Remenska
* Hanno Spreeuw
* Ben van Werkhoven


