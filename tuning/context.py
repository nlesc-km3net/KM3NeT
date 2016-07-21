from __future__ import print_function

import os

import numpy as np

from kernel_tuner import run_kernel

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
    ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)
    return x,y,z,ct
