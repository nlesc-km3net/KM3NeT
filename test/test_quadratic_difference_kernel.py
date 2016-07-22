from __future__ import print_function

import numpy as np
from nose.tools import nottest

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot
from numba import jit
from kernel_tuner import run_kernel

def test_quadratic_difference_kernel():

    skip_if_no_cuda_device()

    #function for computing the reference answer
    @jit
    def correlations_cpu(check, x, y, z, ct):
        for i in range(check.shape[1]):
            for j in range(i + 1, i + check.shape[0] + 1):
                if j < check.shape[1]:
                      if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                       check[j - i - 1, i] = 1
        return check

    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(4.5e3)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)

    correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)

    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]

    #call the CUDA kernel
    params = { "block_size_x": 256, "write_sums": 1, 'window_width': sliding_window_width }
    answer = run_kernel("quadratic_difference_linear", kernel_string, problem_size, args, params)

    #compute reference answer
    correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)

    test_result = np.sum(answer[0] - correlations_ref) == 0
    if not test_result == True:
        print("test quadratic_difference_linear FAILED, attempting to create a plot for visual comparison")
        create_plot(correlations_ref, answer[0])

    assert test_result
