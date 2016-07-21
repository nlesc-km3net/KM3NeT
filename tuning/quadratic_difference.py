#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np

from context import get_kernel_path, generate_input_data

from kernel_tuner import tune_kernel

def tune_quadratic_difference_kernel():

    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(4.5e6)
    sliding_window_width = np.int32(1500)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x,y,z,ct = generate_input_data(N)

    #setup kernel arguments
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)] #multiples of 32
    tune_params["f_unroll"] = [i for i in range(1,20) if 1500/float(i) == 1500//i] #divisors of 1500
    tune_params["tile_size_x"] = [2**i for i in range(5)] #powers of 2
    tune_params["write_sums"] = [1]

    return tune_kernel("quadratic_difference_linear", kernel_string, problem_size, args, tune_params, verbose=True)

if __name__ == "__main__":
    tune_quadratic_difference_kernel()
