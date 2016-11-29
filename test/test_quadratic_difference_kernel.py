from __future__ import print_function

import numpy as np
from kernel_tuner import run_kernel

from .context import skip_if_no_cuda_device, create_plot
from km3net.util import get_kernel_path, correlations_cpu, generate_input_data

def test_quadratic_difference_kernel():
    skip_if_no_cuda_device()

    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(300)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x,y,z,ct = generate_input_data(N)

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
