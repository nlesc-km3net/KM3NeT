from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np
import os
from nose.tools import nottest

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot

from kernel_tuner import run_kernel

def test_dense2sparse_kernel():

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

    with open(get_kernel_path()+'dense2sparse.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(300)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    correlations = np.random.randn(sliding_window_width, N)
    correlations[correlations <= 2.87] = 0
    correlations[correlations > 2.87] = 1
    correlations = np.array(correlations.reshape(sliding_window_width, N), dtype=np.uint8)

    #zero the triangle at the end of correlations table that can not contain any ones
    for j in range(correlations.shape[0]):
        for i in range(N-j-1, N):
            correlations[j,i] = 0

    #obtain full correlation matrix for reference
    dense_matrix = get_full_matrix(correlations)

    #setup all kernel inputs
    node_degrees = dense_matrix.sum(axis=0)
    prefix_sums = np.cumsum(node_degrees).astype(np.int32)
    total_correlated_hits = np.sum(node_degrees.sum())
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)

    #call the CUDA kernel
    args = [row_idx, col_idx, prefix_sums, correlations, N]
    params = { "block_size_x": 256, 'window_width': sliding_window_width }
    answer = run_kernel("dense2sparse_kernel", kernel_string, problem_size, args, params)

    row_idx = answer[0]
    col_idx = answer[1]

    print("computed")
    print("row_idx", row_idx)
    print("col_idx", col_idx)

    #obtain Python objects for the sparse representations of both matrices
    answer = csr_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(N,N))
    reference = csr_matrix(dense_matrix)

    print("reference")
    print("row_idx", reference.nonzero()[0])
    print("col_idx", reference.nonzero()[1])

    #subtract both sparse matrices and test
    #if number of non zero elements is zero, i.e. matrix is empty
    diff = reference - answer
    test_result = diff.nnz == 0

    #verify
    if not test_result == True:
        print("test dense2sparse FAILED, attempting to create a plot for visual comparison")
        create_plot(answer.todense(), reference.todense())

    assert test_result
