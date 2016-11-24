from __future__ import print_function
import os
import pandas
import numpy as np
import scipy.constants

import pycuda.driver as drv

data_dir = '/var/scratch/bwn200/KM3Net/'

def get_kernel_path():
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

def get_slice(x_all, y_all, z_all, ct_all, N, shift):
    x   = x_all[shift:shift+N]
    y   = y_all[shift:shift+N]
    z   = z_all[shift:shift+N]
    ct  = ct_all[shift:shift+N]
    return x,y,z,ct

def init_pycuda():
    drv.init()
    context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])
    return context, cc

def allocate_and_copy(arg):
    gpu_arg = drv.mem_alloc(arg.nbytes)
    drv.memcpy_htod(gpu_arg, arg)
    return gpu_arg

from kernel_tuner import run_kernel

data_dir = '/var/scratch/bwn200/KM3Net/'

def get_kernel_path():
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

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


def generate_large_correlations_table(N, sliding_window_width):
    #generating a very large correlations table takes hours on the CPU
    #reconstruct input data on the GPU
    x,y,z,ct = generate_input_data(N)
    problem_size = (N,1)
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]
    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        qd_string = f.read()
    data = run_kernel("quadratic_difference_linear", qd_string, problem_size, args, {"block_size_x": 512, "write_sums": 1})
    correlations = data[0]
    sums = data[1]

    #now I cant compute the node degrees on the CPU anymore, so using another GPU kernel
    with open(get_kernel_path()+'degrees.cu', 'r') as f:
        degrees_string = f.read()
    args = [sums, correlations, N]
    data = run_kernel("degrees_dense", degrees_string, problem_size, args, {"block_size_x": 512})
    sums = data[0]

    print("generated large correlations table")
    print("N", N)
    print("total_correlated_hits", sums.sum())
    print("density", sums.sum() / (float(N)*sliding_window_width) )

    return correlations, sums


def create_sparse_matrix(correlations, sums):
    N = np.int32(correlations.shape[0])
    prefix_sums = np.cumsum(sums).astype(np.int32)
    total_correlated_hits = np.sum(sums.sum())
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)

    with open(get_kernel_path()+'dense2sparse.cu', 'r') as f:
        kernel_string = f.read()

    args = [row_idx, col_idx, prefix_sums, correlations, N]
    data = run_kernel("dense2sparse_kernel", kernel_string, (N,1), args, {"block_size_x": 256})

    return data[0], data[1], prefix_sums

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

def generate_input_data(N):
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = 2000*np.random.normal(0.5, 0.06, N).astype(np.float32)  #replace 2000 with 5 for the new density

    #print("x", x[:10])
    #print("y", y[:10])
    #print("z", z[:10])
    #print("ct", ct[:10])
    return x,y,z,ct


def get_real_input_data(filename):
    data = pandas.read_csv(data_dir + filename, sep=' ', header=None)

    t = np.array(data[0]).astype(np.float32)
    t = t / 1e9 # convert from nano seconds to seconds
    ct = t * scipy.constants.c

    #x,y,z positions of the hits, assuming these are in meters
    x = np.array(data[1]).astype(np.float32)
    y = np.array(data[2]).astype(np.float32)
    z = np.array(data[3]).astype(np.float32)

    #print("x", x[:10], x.min(), x.max())
    #print("y", y[:10], y.min(), y.max())
    #print("z", z[:10], z.min(), z.max())
    #print("ct", ct[:10], ct.min(), ct.max())
    #print("ct diff", np.diff(ct[:11]))

    N = np.int32(x.size)
    return N,x,y,z,ct


def correlations_cpu_3B(check, x, y, z, ct, roadwidth=100.0, tmax=100.0):
    """function for computing the reference answer using only the 3B condition"""
    index_of_refrac = 1.3800851282
    tan_theta_c     = np.sqrt((index_of_refrac-1.0) * (index_of_refrac+1.0) )
    cos_theta_c     = 1.0 / index_of_refrac
    sin_theta_c     = tan_theta_c * cos_theta_c
    c               = 0.299792458  # m/ns
    inverse_c       = 1.0/c

    TMaxExtra = tmax        #what is the purpose of tmax exactly?

    tt2 = tan_theta_c**2
    D0 = roadwidth          # in meters
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


def correlations_cpu(check, x, y, z, ct):
    """function for computing the reference answer"""
    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   check[j - i - 1, i] = 1
    return check

def correlations_cpu_2(check, x, y, z, ct, roadwidth=None, tmax=None):
    """trying the obvious"""
    for i in range(check.shape[1]):
        for j in range(i + 1, i + check.shape[0] + 1):
            if j < check.shape[1]:
                d_squared = (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2
                ct_squared = (ct[i]-ct[j])**2
                condition_1 = ct_squared < d_squared        #needs to be True
                condition_2 = ct_squared*2 > d_squared      #needs to be True as well

                if condition_1 and condition_2:
                   check[j - i - 1, i] = 1
    return check



