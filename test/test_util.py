from .context import skip_if_no_cuda_device

import numpy as np
import os
from km3net.util import *

#this test verifies that we are testing
#the current repository package rather than the installed package
def test_get_kernel_path():
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    reference = path+'/km3net/kernels/'
    print(reference)
    answer = get_kernel_path()
    print(answer)
    assert reference == answer

#reading or generating input
def test_get_real_input_data():
    pass

def test_generate_input_data():
    pass

def test_generate_correlations_table():
    pass

def test_generate_large_correlations_table():
    pass

def test_get_slice():
    pass

def test_insert_clique():
    pass


#handling matrices
def test_create_sparse_matrix():
    skip_if_no_cuda_device()

    correlations = np.zeros((4,6), dtype=np.uint8)
    correlations[0,1] = 1
    correlations[1,2] = 1
    correlations[2,1] = 1
    dense_matrix = get_full_matrix(correlations)
    sums = np.sum(dense_matrix, axis=1)

    print("correlations table:")
    print(correlations)
    print("matrix:")
    print(dense_matrix)
    print("sums:")
    print(sums)
    print(np.sum(sums.sum()))

    row_idx, col_idx, prefix_sums = create_sparse_matrix(correlations.T, sums)

    reference = csr_matrix(dense_matrix)
    ref_row_idx = reference.nonzero()[0]
    ref_col_idx = reference.nonzero()[1]
    ref_prefix_sums = np.cumsum(sums)

    print(row_idx)
    print(ref_row_idx)
    print(col_idx)
    print(ref_col_idx)
    print(prefix_sums)
    print(ref_prefix_sums)

    assert all([a==b for a,b in zip(row_idx, ref_row_idx)])
    assert all([a==b for a,b in zip(col_idx, ref_col_idx)])
    assert all([a==b for a,b in zip(prefix_sums, ref_prefix_sums)])

def test_get_full_matrix():
    correlations = np.zeros((3,5), dtype=np.uint8)
    correlations[0,1] = 1
    correlations[1,2] = 1
    correlations[2,1] = 1
    answer = get_full_matrix(correlations)

    reference = np.zeros((5,5), dtype=np.uint8)
    row_idx = [1, 1, 2, 2, 4, 4]
    col_idx = [2, 4, 1, 4, 1, 2]
    reference[row_idx,col_idx] = 1

    print(answer)
    print(reference)

    assert all([a==b for a,b in zip(answer.flatten(),reference.flatten())])

def test_sparse_to_dense():
    pass

def test_dense_to_sparse():

    dense_matrix = np.zeros((5,5), dtype=np.uint8)
    dense_matrix[1,1] = 1
    dense_matrix[1,2] = 1
    dense_matrix[2,3] = 1
    dense_matrix[3,4] = 1
    dense_matrix[2,1] = 1

    col_idx, prefix_sum, degrees = dense_to_sparse(dense_matrix)

    print(dense_matrix)
    print(col_idx)
    print(prefix_sum)
    print(degrees)

    ref_col_idx = [1, 2, 1, 3, 4]
    ref_prefix_sum = [0, 2, 4, 5, 5]
    ref_degrees = [0, 2, 2, 1, 0]

    assert all([a==b for a,b in zip(col_idx, ref_col_idx)])
    assert all([a==b for a,b in zip(prefix_sum, ref_prefix_sum)])
    assert all([a==b for a,b in zip(degrees, ref_degrees)])


#cpu versions of algorithms
def test_correlations_cpu_3B():
    pass

def test_correlations_cpu():
    pass


#cuda helper functions
def test_init_pycuda():
    pass

def test_allocate_and_copy():
    pass

def test_ready_input():
    pass

def test_memcpy_dtoh():
    pass

