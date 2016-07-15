from __future__ import print_function

import numpy as np
from nose.tools import nottest

from .context import skip_if_no_cuda_device, get_kernel_path, create_plot

from kernel_tuner import run_kernel

def test_quadratic_difference_kernel():

    skip_if_no_cuda_device()

    #function for computing the reference answer
    def correlations_cpu(check, x, y, z, ct):
        for i in range(check.shape[1]):
            for j in range(i + 1, i + check.shape[0] + 1):
                if j < check.shape[1]:
                    if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                       check[j - i - 1, i] = 1
        return check

    # function for testing the in_degree reference answer
    def in_degrees(correlations):
        degrees = np.zeros(correlations.shape[1], 'uint8')
        for i in range(correlations.shape[1]):
            in_degree = 0
            for j in range(correlations.shape[0]):
                col = i-j-1
                if col>=0:
                    in_degree += correlations[j, col]
            degrees[i] = in_degree
        return degrees

    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(27)
    sliding_window_width = np.int32(9)
    problem_size = (N, 1)

    #generate input data with an expected density of correlated hits
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = 10*np.random.normal(0.5, 0.06, N).astype(np.float32)

    correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)

    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]

    #call the first kernel; don't compute the sums
    params = { "block_size_x": 9, "write_sums": 0, 'window_width': sliding_window_width }
    answer = run_kernel("quadratic_difference_linear", kernel_string, problem_size, args, params)

    #compute reference answer for correlations
    correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)

    # test correlations result from first kernel
    test_result_correlations = np.sum(answer[0] - correlations_ref) == 0
    if not test_result_correlations == True:
        print("test quadratic_difference_linear FAILED, attempting to create a plot for visual comparison")
        create_plot(correlations_ref, answer[0])


    # add second kernel for computing degrees
    with open(get_kernel_path()+'degrees.cu', 'r') as f:
        kernel_string = f.read()

    #    = [out_degree, correlations, N]
    args = [answer[1], answer[0], N]
    params = { "block_size_x": 9, 'window_width': sliding_window_width }
    answer1 = run_kernel("degrees_dense", kernel_string, problem_size, args, params)

    # Test also out_degree as a result of second kernel
    in_degree = in_degrees(correlations_ref)
    out_degree = np.sum(correlations_ref, axis=0).astype(np.int32)
    reference = (in_degree+out_degree)
    test_result_degree = np.sum(answer1[0] - reference) == 0

    print("full_degree:\n", answer1[0])
    print("reference:\n", reference)
    print("correlations:\n", answer[0])

    assert test_result_correlations and test_result_degree
