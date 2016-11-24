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
def correlations_cpu(check, x, y, z, ct):
    """function for computing the reference answer"""
    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   check[j - i - 1, i] = 1
    return check

@nottest
def correlations_cpu_3B(check, x, y, z, ct, roadwidth=100.0, tmax=100.0):
    """function for computing the reference answer using only the 3B condition"""
    index_of_refrac = 1.3800851282
    tan_theta_c     = np.sqrt((index_of_refrac-1.0) * (index_of_refrac+1.0) )
    cos_theta_c     = 1.0 / index_of_refrac
    sin_theta_c     = tan_theta_c * cos_theta_c
    c               = 0.299792458  # m/ns
    inverse_c       = 1.0/c

    TMaxExtra = tmax         #another wild guess

    tt2 = tan_theta_c**2
    D0 = roadwidth
    D1 = roadwidth*2.0
    D2 = roadwidth * 0.5 * np.sqrt(tt2 + 10.0 + 9.0/tt2)

    D02 = D0**2
    D12 = D1**2
    D22 = D2**2

    R = roadwidth
    Rs = R * sin_theta_c

    R2 = R**2
    Rs2 = Rs**2
    Rst = Rs * tan_theta_c
    Rt = R * tan_theta_c

    #our ct is in meters, convert back to nanoseconds
    t = ct * inverse_c

    def test_3B_condition(t1,x1,y1,z1, t2,x2,y2,z2):
        diffx = x1-x2
        diffy = y2-y2
        diffz = z2-z2
        d2 = diffx**2 + diffy**2 + diffz**2
        difft = np.absolute(t1 - t2)

        if d2 < D02:
            dmax = np.sqrt(d2) * index_of_refrac
        else:
            dmax = np.sqrt(d2 - Rs2) + Rst

        if difft > dmax * inverse_c + TMaxExtra:
            return False

        if d2 > D22:
            dmin = np.sqrt(d2 - R2) - Rt
        elif d2 > D12:
            dmin = np.sqrt(d2 - D12)
        else:
            return True

        return difft >= dmin * inverse_c - TMaxExtra

    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                if test_3B_condition(t[i],x[i],y[i],z[i],t[j],x[j],y[j],z[j]):
                   check[j - i - 1, i] = 1

                #if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                #   check[j - i - 1, i] = 1
    return check

@nottest
def generate_input_data(N):
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)
    return x,y,z,ct

