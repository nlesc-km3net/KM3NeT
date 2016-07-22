import os

from nose import SkipTest
from nose.tools import nottest
import numpy as np
from numba import jit
@nottest
def skip_if_no_cuda_device():
    try:
        from pycuda.autoinit import context
    except (ImportError, Exception):
        raise SkipTest("PyCuda not installed or no CUDA device detected")

@nottest
def get_kernel_path():
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

@nottest
def create_plot(corr1, corr2):
    try:
        from matplotlib import pyplot
        f, (ax1, ax2, ax3) = pyplot.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        ax1.imshow(corr1, cmap=pyplot.cm.bone)
        ax2.imshow(corr2, cmap=pyplot.cm.bone)
        ax3.imshow(corr2-corr1, cmap=pyplot.cm.jet)
        pyplot.show()
    except:
        pass

@jit
def degrees_dense_cpu(correlations):
    in_degrees = np.zeros(correlations.shape[1],'int')
    for i in range(correlations.shape[1]):
        in_degree = 0
        for j in range(correlations.shape[0]):
            col = i-j-1
            if col>=0:
                in_degree += correlations[j, col]
        in_degrees[i] = in_degree

    out_degrees = np.sum(correlations, axis=0).astype(np.int32)
    return in_degrees + out_degrees

#function for computing the reference answer
@jit
def correlations_cpu(check, x, y, z, ct):
    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                  if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   check[j - i - 1, i] = 1
    return check
