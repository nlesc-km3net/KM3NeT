from __future__ import print_function

import numpy as np
from kernel_tuner import run_kernel

from .context import skip_if_no_cuda_device, create_plot
from km3net.util import get_kernel_path, generate_correlations_table

def test_degrees_kernel():
    skip_if_no_cuda_device()

    def in_degrees(correlations):
        degrees = np.zeros(correlations.shape[1])
        for i in range(correlations.shape[1]):
            in_degree = 0
            for j in range(correlations.shape[0]):
                col = i-j-1
                if col>=0:
                    in_degree += correlations[j, col]
            degrees[i] = in_degree
        return degrees

    with open(get_kernel_path()+'degrees.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(400)
    sliding_window_width = np.int32(150)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    correlations = generate_correlations_table(N, sliding_window_width, cutoff=2.87)

    #compute reference answer
    in_degree = in_degrees(correlations)
    out_degree = np.sum(correlations, axis=0).astype(np.int32)
    reference = (in_degree+out_degree)

    #call the CUDA kernel
    args = [out_degree, correlations, N]
    params = { "block_size_x": 256, 'window_width': sliding_window_width }
    answer = run_kernel("degrees_dense", kernel_string, problem_size, args, params)

    print("answer", answer[0])
    print("reference", reference)

    #verify
    test_result = np.sum(answer[0] - reference) == 0
    if not test_result == True:
        print("test degrees_dense FAILED, attempting to create a plot for visual comparison")
        create_plot(reference.reshape(20,20), answer[0].reshape(20,20))

    assert test_result
