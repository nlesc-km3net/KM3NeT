from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np

from .context import skip_if_no_cuda_device, get_kernel_path, get_full_matrix, generate_correlations_table

from kernel_tuner import run_kernel

def test_remove_nodes_kernel():

    skip_if_no_cuda_device()

    with open(get_kernel_path()+'remove_nodes.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(300)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)
    params = { "block_size_x": 128 }

    #generate input data with an expected density of correlated hits
    correlations = generate_correlations_table(N, sliding_window_width, cutoff=2.2)

    #obtain full correlation matrix for reference
    dense_matrix = get_full_matrix(correlations)
    sparse_matrix = csr_matrix(dense_matrix)

    #setup all kernel inputs
    degrees = dense_matrix.sum(axis=0).astype(np.int32)
    prefix_sums = np.cumsum(degrees).astype(np.int32)
    row_idx = (sparse_matrix.nonzero()[0]).astype(np.int32)
    col_idx = (sparse_matrix.nonzero()[1]).astype(np.int32)
    minimum = np.zeros(2).astype(np.int32)
    minimum[0] = 5 #remove nodes with degree 5 or less, as a test

    #call the CUDA kernel
    args = [degrees, row_idx, col_idx, prefix_sums, minimum, N]
    answer = run_kernel("remove_nodes", kernel_string, problem_size, args, params)

    #verify all kernel outputs
    #updated degrees array
    print("degrees")
    print(answer[0])
    reference = degrees.copy()
    reference[degrees <= minimum[0]] = 0
    assert all(answer[0] == reference)

    #updated col_idx array
    #this array will contain -1 for removed edges of non-removed nodes
    print("col_idx")
    print(answer[2])
    col_idx = answer[2]

    #construct sparse matrix without the removed edges
    row_idx = row_idx[col_idx!=-1]
    col_idx = col_idx[col_idx!=-1]
    answer_matrix = csr_matrix((np.ones_like(row_idx),(row_idx,col_idx)), shape=(N,N))

    #reconstruct the same matrix from the reference answer
    for i in range(dense_matrix.shape[0]):
        for j in range(dense_matrix.shape[1]):
            #for non-removed nodes remove the edge to removed nodes
            if reference[i] > 0 and reference[j] == 0:
                dense_matrix[i,j] = 0
    sparse_ref = csr_matrix(dense_matrix)
    print(sparse_ref.nnz)

    #verify that the two sparse matrices are the same
    assert answer_matrix.nnz == sparse_ref.nnz
    assert all(sparse_ref.nonzero()[0] == answer_matrix.nonzero()[0])
    assert all(sparse_ref.nonzero()[1] == answer_matrix.nonzero()[1])

