from __future__ import print_function

from scipy.sparse import csr_matrix
import numpy as np

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot, get_full_matrix, correlations_cpu, generate_input_data

from kernel_tuner import run_kernel

def test_quadratic_difference_full_sums():

    skip_if_no_cuda_device()

    with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(600)
    sliding_window_width = np.int32(300)
    problem_size = (N, 1)

    x,y,z,ct = generate_input_data(N)

    correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    col_idx = np.zeros(10).astype(np.int32)         #not used in this test
    prefix_sums = np.zeros(10).astype(np.int32)     #not used in this test

    args = [col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    #call the CUDA kernel
    params = { "block_size_x": 128, "write_sums": 1, 'window_width': sliding_window_width, 'tile_size_x': 2 }
    answer = run_kernel("quadratic_difference_full", kernel_string, problem_size, args, params)

    #compute reference answer
    correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)
    corr_matrix = get_full_matrix(correlations_ref)

    sums_ref = np.sum(corr_matrix, axis=0)
    #sums_ref = np.sum(correlations_ref, axis=0)
    print("reference", sums_ref.sum())
    print(sums_ref)

    sums = answer[2]
    print("answer", sums.sum())
    print(sums)

    diff = (sums_ref - sums).astype(np.int8)
    print("diff")
    print(diff)

    assert all(diff == 0)


def test_quadratic_difference_full_sparse_matrix():

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
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)
    prefix_sums = np.cumsum(sums_ref).astype(np.int32)

    args = [col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    #call the CUDA kernel
    params = { "block_size_x": 128, "write_spm": 1, 'window_width': sliding_window_width, 'tile_size_x': 1 }
    answer = run_kernel("quadratic_difference_full", kernel_string, problem_size, args, params)

    reference = csr_matrix(corr_matrix)
    col_idx_ref = reference.nonzero()[1]

    print("reference")
    print(col_idx_ref)

    print("answer")
    print(answer[0])

    print("diff")
    diff = col_idx_ref - answer[0]
    print(diff)

    assert all(diff == 0)

