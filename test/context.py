import os

from nose import SkipTest
from nose.tools import nottest

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
