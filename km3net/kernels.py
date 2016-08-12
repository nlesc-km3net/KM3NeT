from __future__ import print_function

import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from km3net.util import *

class QuadraticDifferenceSparse(object):

    def __init__(self, N, sliding_window_width, cc):
        self.N = N
        self.sliding_window_width = sliding_window_width

        with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
            kernel_string = f.read()
        block_size_x = 256
        prefix = "#define block_size_x " + str(block_size_x) + "\n"
        kernel_string = prefix + kernel_string

        self.quadratic_difference_sums = SourceModule("#define write_sums 1\n" + kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("quadratic_difference_full_shfl")
        self.quadratic_difference_sparse_matrix = SourceModule("#define write_spm 1\n" + kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("quadratic_difference_full_shfl")

        self.threads = (block_size_x, 1, 1)
        self.grid = (int(np.ceil(N/float(block_size_x))), 1)

    def compute(self, d_x, d_y, d_z, d_ct):

        #run the first kernel
        row_idx = np.zeros(10).astype(np.int32)
        col_idx = np.zeros(10).astype(np.int32)
        prefix_sums = np.zeros(10).astype(np.int32)
        degrees = np.zeros(self.N).astype(np.int32)

        d_degrees = allocate_and_copy(degrees)
        d_row_idx = allocate_and_copy(row_idx)
        d_col_idx = allocate_and_copy(col_idx)
        d_prefix_sums = allocate_and_copy(prefix_sums)

        args_list = [d_row_idx, d_col_idx, d_prefix_sums, d_degrees, self.N, self.sliding_window_width, d_x, d_y, d_z, d_ct]
        self.quadratic_difference_sums(*args_list, block=self.threads, grid=self.grid, stream=None, shared=0)

        #allocate space to store sparse matrix
        drv.memcpy_dtoh(degrees, d_degrees)
        total_correlated_hits = degrees.sum()
        col_idx = np.zeros(total_correlated_hits).astype(np.int32)
        prefix_sums = np.cumsum(degrees).astype(np.int32)

        d_col_idx = allocate_and_copy(col_idx)
        d_prefix_sums = allocate_and_copy(prefix_sums)

        args_list2 = [d_row_idx, d_col_idx, d_prefix_sums, d_degrees, self.N, self.sliding_window_width, d_x, d_y, d_z, d_ct]
        self.quadratic_difference_sparse_matrix(*args_list2, block=self.threads, grid=self.grid, stream=None, shared=0)

        return d_col_idx, d_prefix_sums, d_degrees


class PurgingSparse(object):

    def __init__(self, N, cc):
        self.N = N

        with open(get_kernel_path()+'remove_nodes.cu', 'r') as f:
            remove_nodes_string = f.read()
        with open(get_kernel_path()+'minimum_degree.cu', 'r') as f:
            minimum_string = f.read()
        self.minimum_degree = SourceModule(minimum_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("minimum_degree")
        self.combine_blocked_min_num = SourceModule(minimum_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("combine_blocked_min_num")
        self.remove_nodes = SourceModule(remove_nodes_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("remove_nodes")

        block_size_x = 128
        self.max_blocks = (np.ceil(N / float(block_size_x))).astype(np.int32)
        self.threads = (block_size_x, 1, 1)
        self.grid = (int(self.max_blocks), 1)



    def compute(self, d_col_idx, d_prefix_sums, d_degrees, shift=0):

        minimum = np.zeros(self.max_blocks).astype(np.int32)
        num_nodes = np.zeros(self.max_blocks).astype(np.int32)

        #setup GPU memory
        d_row_idx = allocate_and_copy(np.zeros(10).astype(np.int32))
        d_minimum = allocate_and_copy(minimum)
        d_num_nodes = allocate_and_copy(num_nodes)

        args_minimum = [d_minimum, d_num_nodes, d_degrees, d_row_idx, d_col_idx, d_prefix_sums, self.N]
        self.minimum_degree(*args_minimum, block=self.threads, grid=self.grid)

        #call the helper kernel to combine these values
        args_combine = [d_minimum, d_num_nodes, self.max_blocks]
        self.combine_blocked_min_num(*args_combine, block=self.threads, grid=(1,1))

        current_minimum = np.array([0]).astype(np.int32)
        current_num_nodes = np.array([0]).astype(np.int32)
        drv.memcpy_dtoh(current_minimum, d_minimum)
        drv.memcpy_dtoh(current_num_nodes, d_num_nodes)
        #print("current_minimum", current_minimum)
        #print("current_num_nodes", current_num_nodes)

        args_remove = [d_degrees, d_row_idx, d_col_idx, d_prefix_sums, d_minimum, self.N]

        counter = 0
        while current_minimum+1 < current_num_nodes:

            counter += 1
            self.remove_nodes(*args_remove, block=self.threads, grid=self.grid)
            self.minimum_degree(*args_minimum, block=self.threads, grid=self.grid)
            self.combine_blocked_min_num(*args_combine, block=self.threads, grid=(1,1))
            drv.memcpy_dtoh(current_minimum, d_minimum)
            drv.memcpy_dtoh(current_num_nodes, d_num_nodes)

            #print("current_minimum", current_minimum)
            #print("current_num_nodes", current_num_nodes)

        print("finished purging, iterations = ", counter)

        degrees = np.zeros(self.N).astype(np.int32)
        drv.memcpy_dtoh(degrees, d_degrees)
        if (current_num_nodes > 0):
            print("found clique of size=", current_num_nodes)
            indices = np.array(range(degrees.size))
            found_indices = indices[degrees >= current_minimum]
            print(found_indices + shift)
            return found_indices.size

        return 0


