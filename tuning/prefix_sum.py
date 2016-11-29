#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel

from context import get_kernel_path

def tune_prefix_sum_kernel():

    with open(get_kernel_path()+'prefixsum.cu', 'r') as f:
        kernel_string = f.read()

    N = np.int32(4.5e6)
    problem_size = (N, 1)

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]

    max_blocks = np.ceil(N/float(max(tune_params["block_size_x"]))).astype(np.int32)
    x = np.ones(N).astype(np.int32)

    #setup kernel arguments
    prefix_sums = np.zeros(N).astype(np.int32)
    block_carry = np.zeros(max_blocks).astype(np.int32)
    args = [prefix_sums, block_carry, x, N]

    #tune only the first kernel that computes the thread block-wide prefix sums
    #and outputs the block carry values
    return tune_kernel("prefix_sum_block", kernel_string, problem_size, args, tune_params, verbose=True)

if __name__ == "__main__":
    tune_prefix_sum_kernel()

