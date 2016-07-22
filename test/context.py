import os

from nose import SkipTest
from nose.tools import nottest
import numpy as np

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
        ax1.imshow(corr1, cmap=pyplot.cm.bone, interpolation='nearest')
        ax2.imshow(corr2, cmap=pyplot.cm.bone, interpolation='nearest')
        ax3.imshow(corr2-corr1, cmap=pyplot.cm.jet, interpolation='nearest')
        pyplot.show()
    except:
        pass

@nottest
def generate_correlations_table(N, sliding_window_width, cutoff=2.87):
    #generate input data with an expected density of correlated hits
    correlations = np.random.randn(sliding_window_width, N)
    correlations[correlations <= cutoff] = 0
    correlations[correlations > cutoff] = 1
    correlations = np.array(correlations.reshape(sliding_window_width, N), dtype=np.uint8)

    #zero the triangle at the end of correlations table that cannot contain any ones
    for j in range(correlations.shape[0]):
        for i in range(N-j-1, N):
            correlations[j,i] = 0

    return correlations

@nottest
def get_full_matrix(correlations):
    #obtain a true correlation matrix from the correlations table
    n = correlations.shape[1]
    matrix = np.zeros((n,n), dtype=np.uint8)
    for i in range(n):
        for j in range(correlations.shape[0]):
            if correlations[j,i] == 1:
                col = i+j+1
                if col < n and col >= 0:
                    matrix[i,col] = 1
                    matrix[col,i] = 1
    return matrix

@nottest
#function for computing the reference answer
def correlations_cpu(check, x, y, z, ct):
    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   check[j - i - 1, i] = 1
    return check

@nottest
def generate_input_data(N):
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)
    return x,y,z,ct

