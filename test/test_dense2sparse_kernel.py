from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot, get_full_matrix, generate_correlations_table

from kernel_tuner import run_kernel

def test_dense2sparse_kernel():

    skip_if_no_cuda_device()

    with open(get_kernel_path()+'dense2sparse.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(300)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    correlations = generate_correlations_table(N, sliding_window_width, cutoff=2.87)

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
