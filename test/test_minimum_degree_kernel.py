from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np
import os
from nose.tools import nottest

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot

from kernel_tuner import run_kernel

def test_minimum_degree_kernel():

    skip_if_no_cuda_device()

    #obtain a true correlation matrix from the correlations table
    def get_full_matrix(correlations):
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

    with open(get_kernel_path()+'minimum_degree.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(300)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)
    params = { "block_size_x": 128 }
    max_blocks = int(np.ceil(N / float(params["block_size_x"])))

    #generate input data with an expected density of correlated hits
    correlations = np.random.randn(sliding_window_width, N)
    correlations[correlations <= 2.2] = 0
    correlations[correlations > 2.2] = 1
    correlations = np.array(correlations.reshape(sliding_window_width, N), dtype=np.uint8)

    #zero the triangle at the end of correlations table that can not contain any ones
    for j in range(correlations.shape[0]):
        for i in range(N-j-1, N):
            correlations[j,i] = 0

    #obtain full correlation matrix for reference
    dense_matrix = get_full_matrix(correlations)
    sparse_matrix = csr_matrix(dense_matrix)

    #setup all kernel inputs
    degrees = dense_matrix.sum(axis=0).astype(np.int32)
    prefix_sums = np.cumsum(degrees).astype(np.int32)
    row_idx = (sparse_matrix.nonzero()[0]).astype(np.int32)
    col_idx = (sparse_matrix.nonzero()[1]).astype(np.int32)
    minimum = np.zeros(max_blocks).astype(np.int32)
    num_nodes = np.zeros(max_blocks).astype(np.int32)
    input_degrees = degrees + (np.random.rand(degrees.size)*10.0).astype(np.int32)

    #call the CUDA kernel
    args = [minimum, num_nodes, input_degrees, row_idx, col_idx, prefix_sums, N]
    answer = run_kernel("minimum_degree", kernel_string, problem_size, args, params)

    #verify all kernel outputs
    #minimum
    print ("block-based minimum")
    print (answer[0])
    print ("minimum computed reference")
    min_answer = np.ma.masked_equal(answer[0], 0).min()
    min_reference = np.ma.masked_equal(degrees, 0).min()
    print (min_answer)
    print (min_reference)
    assert min_answer == min_reference

    #num_nodes
    print ("block-based num_nodes")
    print (answer[1])
    print ("num_nodes computed reference")
    num_answer = answer[1].sum()
    print (num_answer)
    count = np.zeros_like(degrees)
    count[degrees > 0] = 1
    num_reference = np.ma.masked_equal(count, 0).sum()
    print (num_reference)
    assert num_answer == num_reference

    #degrees
    print ("degrees computed")
    print (answer[2])
    print ("degrees reference")
    print (degrees)
    assert all(answer[2] - degrees == 0)

