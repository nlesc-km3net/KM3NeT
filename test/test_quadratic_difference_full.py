from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np
from nose.tools import nottest
from kernel_tuner import run_kernel

from .context import skip_if_no_cuda_device, create_plot
from km3net.util import get_kernel_path, get_full_matrix, correlations_cpu, generate_input_data, sparse_to_dense

def test_quadratic_difference_full_sums_impl1():
    test_quadratic_difference_full_sums("quadratic_difference_full")

@nottest
def test_quadratic_difference_full_sums_impl2():
    test_quadratic_difference_full_sums("quadratic_difference_full_shfl")

@nottest
def test_quadratic_difference_full_sums(kernel_name):
    skip_if_no_cuda_device()

    with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(600)
    sliding_window_width = np.int32(300)
    problem_size = (N, 1)

    x,y,z,ct = generate_input_data(N)

    correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    row_idx = np.zeros(10).astype(np.int32)         #not used in this test
    col_idx = np.zeros(10).astype(np.int32)         #not used in this test
    prefix_sums = np.zeros(10).astype(np.int32)     #not used in this test

    args = [row_idx, col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    #call the CUDA kernel
    params = { "block_size_x": 128, "write_sums": 1, 'window_width': sliding_window_width, 'tile_size_x': 1 }
    answer = run_kernel(kernel_name, kernel_string, problem_size, args, params)

    #compute reference answer
    correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)
    corr_matrix = get_full_matrix(correlations_ref)

    sums_ref = np.sum(corr_matrix, axis=0)
    #sums_ref = np.sum(correlations_ref, axis=0)
    print("reference", sums_ref.sum())
    print(sums_ref)

    sums = answer[3]
    print("answer", sums.sum())
    print(sums)

    diff = (sums_ref - sums).astype(np.int8)
    print("diff")
    print(diff)

    assert all(diff == 0)


def test_quadratic_difference_full_sparse_matrix_impl1():
    test_quadratic_difference_full_sparse_matrix("quadratic_difference_full")

@nottest
def test_quadratic_difference_full_sparse_matrix_impl2():
    test_quadratic_difference_full_sparse_matrix("quadratic_difference_full_shfl")


@nottest
def test_quadratic_difference_full_sparse_matrix(kernel_name):
    skip_if_no_cuda_device()

    with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(600)
    sliding_window_width = np.int32(300)
    problem_size = (N, 1)

    x,y,z,ct = generate_input_data(N)

    #compute reference answer
    correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
    correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)
    corr_matrix = get_full_matrix(correlations_ref)
    sums_ref = np.sum(corr_matrix, axis=0)
    total_correlated_hits = corr_matrix.sum()

    sums = sums_ref.astype(np.int32)
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)
    prefix_sums = np.cumsum(sums_ref).astype(np.int32)

    args = [row_idx, col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    #call the CUDA kernel
    params = { "block_size_x": 128, "write_spm": 1, 'write_rows': 1, 'window_width': sliding_window_width, 'tile_size_x': 1 }
    answer = run_kernel(kernel_name, kernel_string, problem_size, args, params)

    reference = csr_matrix(corr_matrix)
    col_idx_ref = reference.nonzero()[1]

    row_idx = answer[0]
    print("row_idx")
    print(row_idx)
    col_idx = answer[1]
    print("col_idx")
    print(col_idx)

    col_idx = answer[1]
    answer = csr_matrix((np.ones_like(row_idx), (row_idx, col_idx)), shape=(N,N))

    print("reference")
    print(list(zip(reference.nonzero()[0], reference.nonzero()[1])))

    print("answer")
    print(list(zip(answer.nonzero()[0], answer.nonzero()[1])))

    diff = reference - answer

    print("diff")
    print(list(zip(diff.nonzero()[0], diff.nonzero()[1])))
    print("diff.nnz", diff.nnz)

    answer2 = csr_matrix(sparse_to_dense(prefix_sums, col_idx), shape=(N,N))
    diff2 = reference - answer2
    print("diff2")
    print(list(zip(diff2.nonzero()[0], diff2.nonzero()[1])))
    print("diff2.nnz", diff2.nnz)

    if False:
        create_plot(get_full_matrix(reference), get_full_matrix(answer))

    assert diff.nnz == 0
    assert diff2.nnz == 0
