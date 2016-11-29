#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel

from context import get_kernel_path
from km3net.util import generate_input_data

def tune_quadratic_difference_kernel():

    with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
        kernel_string = f.read()
    kernel_name = "quadratic_difference_full"

    N = np.int32(4.5e6)
    sliding_window_width = np.int32(1500)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x,y,z,ct = generate_input_data(N)

    #setup kernel arguments
    row_idx = np.zeros(10).astype(np.int32)         #not used in first kernel
    col_idx = np.zeros(10).astype(np.int32)         #not used in first kernel
    prefix_sums = np.zeros(10).astype(np.int32)     #not used in first kernel
    sums = np.zeros(N).astype(np.int32)
    args = [row_idx, col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    #run the sums kernel once
    params = {"block_size_x": 256, "write_sums": 1}
    answer = run_kernel(kernel_name, kernel_string, problem_size, args, params)
    reference = [None for _ in range(len(args))]
    reference[3] = answer[3]
    sums = reference[3].astype(np.int32)

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)] #multiples of 32
    tune_params["f_unroll"] = [i for i in range(1,20) if 1500/float(i) == 1500//i] #divisors of 1500
    tune_params["tile_size_x"] = [2**i for i in range(5)] #powers of 2
    tune_params["write_sums"] = [1]
    tune_params["write_spm"] = [0]

    kernel_1 = tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params, verbose=True)

    #tune kernel #2
    total_correlated_hits = sums.sum()
    print("total_correlated_hits", total_correlated_hits)
    print("density", total_correlated_hits/(float(N)*sliding_window_width))

    col_idx = np.zeros(total_correlated_hits).astype(np.int32)
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    prefix_sums = np.cumsum(sums).astype(np.int32)
    args = [col_idx, prefix_sums, sums, N, sliding_window_width, x, y, z, ct]

    tune_params["write_sums"] = [0]
    tune_params["write_spm"] = [1]

    kernel_2 = tune_kernel(kernel_name, kernel_string, problem_size, args, tune_params, verbose=True)

    return kernel_1, kernel_2


if __name__ == "__main__":
    tune_quadratic_difference_kernel()
