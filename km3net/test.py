#!/usr/bin/env python

import numpy as np

from kernel_tuner.util import prepare_kernel_string, get_grid_dimensions, get_thread_block_dimensions

from pycuda.compiler import SourceModule

N = np.int32(4.5e6)
sliding_window_width = 1500

x = np.random.normal(0.2, 0.1, N).astype(np.float32)
y = np.random.normal(0.2, 0.1, N).astype(np.float32)
z = np.random.normal(0.2, 0.1, N).astype(np.float32)
ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)

correlations = np.zeros((sliding_window_width, N), 'uint8')

class QuadraticDifference(object):
    """object that provides a Python interface to calling the quadratic difference linear CUDA kernel"""

    def __init__(self, N, sliding_window_width=1500):
        self.N = N
        self.sliding_window_width = sliding_window_width

        #parameters obtained from tuning the kernel
        self.params = params = dict()
        params["block_size_x"] = 384
        params["tile_size_x"] = 4
        params["funroll"] = 1
        params["write_sums"] = 1
        params["window_width"] = sliding_window_width

        #add parameters to the source code, could also do this through setting defaults in the code
        with open('kernels/quadratic_difference_linear.cu', 'r') as f:
            kernel_string = f.read()
        kernel_string = _prepare_kernel_string(kernel_string, params)

        #get compute capability
        self.context = #INSERT PYCUDA CONTEXT REF HERE
        devprops = { str(k): v for (k, v) in self.context.get_device().get_attributes().items() }
        self.cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

        #compile kernel
        self.kernel = SourceModule(kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + self.cc, code='sm_' + self.cc,
                    cache_dir=False).get_function('quadratic_difference_linear')

        #setup grid and thread block dimensions for later kernel calls
        self.threads = _get_thread_block_dimensions(params)
        self.grid = _get_grid_dimensions((N,1), params, [], ["block_size_x", "tile_size_x"])


#decide on whether to get the compute capability here or from above


#try to decide on how or where to allocate and move data to the GPU
#perhaps letting this object allocate memory is not the nicest approach

args = [correlations, x, y, z, ct, N]

gpu_args = dev.ready_argument_list(args)


    

dev.run_kernel(qd_kernel, gpu_args, (params["block_size_x"], 1, 1), grid)

    def 

    def set_args(self, *args):
        pass


    def run(self, gpu_args):
        """run the kernel given a list of arguments containing GPU memory allocations"""
        self.kernel(*gpu_args, block=self.threads, grid=self.grid)



