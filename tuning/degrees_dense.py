#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np

from context import get_kernel_path, generate_input_data

from kernel_tuner import run_kernel, tune_kernel

def tune_degrees_dense():

    with open(get_kernel_path()+'degrees.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(4.5e6)
    sliding_window_width = np.int32(1500)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x,y,z,ct = generate_input_data(N)
    problem_size = (N,1)
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]
    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        qd_string = f.read()
    data = run_kernel("quadratic_difference_linear", qd_string, problem_size, args, {"block_size_x": 512, "write_sums": 1})
    correlations = data[0]
    sums = data[1]  #partial sum of the # of correlated hits to hits later in time

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["window_width"] = [sliding_window_width]

    args = [sums, correlations, N]
    return tune_kernel("degrees_dense", kernel_string, problem_size, args, tune_params, verbose=True)

if __name__ == "__main__":
    tune_degrees_dense()
