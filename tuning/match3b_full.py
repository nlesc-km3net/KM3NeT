#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel

from context import get_kernel_path
from km3net.util import generate_input_data

from correlate_full import tune_correlate_full_kernel

if __name__ == "__main__":
    tune_correlate_full_kernel("match3b_full")
