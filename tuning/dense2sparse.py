#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np

from context import get_kernel_path, get_full_matrix, generate_large_correlations_table

from kernel_tuner import tune_kernel, run_kernel

def tune_dense2sparse():

    with open(get_kernel_path()+'dense2sparse.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(4.5e6)
    sliding_window_width = np.int32(1500)
    problem_size = (N, 1)

    #generate input
    correlations, sums = generate_large_correlations_table(N, sliding_window_width)

    #setup all kernel inputs
    prefix_sums = np.cumsum(sums).astype(np.int32)
    total_correlated_hits = np.sum(sums.sum())
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)] #factors of 32 up to 1024
    tune_params["window_width"] = [sliding_window_width]
    tune_params["f_unroll"] = [i for i in range(1,7) if 1500/float(i) == 1500//i] #divisors of 1500

    #call the tuner
    args = [row_idx, col_idx, prefix_sums, correlations, N]
    return tune_kernel("dense2sparse_kernel", kernel_string, problem_size, args, tune_params, verbose=True)

if __name__ == "__main__":
    tune_dense2sparse()
