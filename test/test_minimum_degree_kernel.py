from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np
from kernel_tuner import run_kernel

from .context import skip_if_no_cuda_device
from km3net.util import get_kernel_path, get_full_matrix, generate_correlations_table

def test_minimum_degree_kernel():
    skip_if_no_cuda_device()

    with open(get_kernel_path()+'minimum_degree.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(3000)
    sliding_window_width = np.int32(1500)
    problem_size = (N, 1)
    params = { "block_size_x": 128, "threshold": 1 }
    max_blocks = int(np.ceil(N / float(params["block_size_x"])))

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
    minimum = np.zeros(max_blocks).astype(np.int32)
    num_nodes = np.zeros(max_blocks).astype(np.int32)
    input_degrees = degrees + (np.random.rand(degrees.size)*10.0).astype(np.int32)

    #call the CUDA kernel
    args = [minimum, num_nodes, degrees, row_idx, col_idx, prefix_sums, N]
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


def test_combine_blocked_min_num():
    skip_if_no_cuda_device()

    with open(get_kernel_path()+'minimum_degree.cu', 'r') as f:
        kernel_string = f.read()

    size = 487
    N = np.int32(size)

    minimum = (np.random.rand(size)*10.0).astype(np.int32)
    num_nodes = (np.random.rand(size)*10.0).astype(np.int32)

    args = [minimum, num_nodes, N]

    params = { "block_size_x": 512, "threshold": 1 }
    answer = run_kernel("combine_blocked_min_num", kernel_string, (1,1), args, params)

    print(answer[0])
    print(minimum)

    assert answer[0][0] == np.ma.masked_equal(minimum, 0).min()

    print(answer[1][0])
    print(num_nodes.sum())

    assert answer[1][0] == num_nodes.sum()

