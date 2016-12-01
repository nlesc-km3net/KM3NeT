from __future__ import print_function

import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from km3net.util import *

class QuadraticDifferenceSparse(object):
    """ class that provides an interface to the Quadratic Difference GPU Kernel and maintains GPU state"""

    def __init__(self, N, sliding_window_width=1500, cc='52'):
        """instantiate QuadraticDifferenceSparse

        Create the object that provides an interface to the GPU kernel for performing the
        Quadratic Difference algorithm. This implementation of the algorithm stores the
        correlation table in a sparse manner, using CSR notation.

        When this object is instantiated the CUDA kernel
        code is compiled and some of the GPU memory is allocated.

        :param N: The largest number of hits that are to be processed by one iteration
                of the quadratic difference algorithm.
        :type N: int

        :param sliding_window_width: The width of the 'window' in which we look for correlated
                hits. This is related to the size of the detector and the expected rate of background
                induced hits. The value we currently assume is 1500.
        :type sliding_window_width: int

        :param cc: The CUDA compute capability of the target device as a string, consisting
                of the major and minor number concatenated without any separators.
        :type cc: string

        """

        self.N = np.int32(N)
        self.sliding_window_width = np.int32(sliding_window_width)

        with open(get_kernel_path()+'quadratic_difference_full.cu', 'r') as f:
            kernel_string = f.read()
        block_size_x = 128
        prefix = "#define block_size_x " + str(block_size_x) + "\n" + "#define window_width " + str(sliding_window_width) + "\n"
        kernel_string = prefix + kernel_string

        self.quadratic_difference_sums = SourceModule("#define write_sums 1\n" + kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("quadratic_difference_full")
        self.quadratic_difference_sparse_matrix = SourceModule("#define write_spm 1\n" + kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + cc, code='sm_' + cc,
                    cache_dir=False).get_function("quadratic_difference_full")

        self.threads = (block_size_x, 1, 1)
        self.grid = (int(np.ceil(N/float(block_size_x))), 1)

    def compute(self, x, y, z, ct):
        """ perform a computation of the quadratic difference algorithm


        :param d_x: an array storing the x-coordinates of the hits,
            either a numpy ndarray or an array stored on the GPU
        :type d_x: numpy ndarray or pycuda.driver.DeviceAllocation

        :param d_y: an array storing the y-coordinates of the hits,
            either a numpy ndarray or an array stored on the GPU
        :type d_x: numpy ndarray or pycuda.driver.DeviceAllocation

        :param d_z: an array storing the z-coordinates of the hits,
            either a numpy ndarray or an array stored on the GPU
        :type d_x: numpy ndarray or pycuda.driver.DeviceAllocation

        :param d_ct: an array storing the 'ct' value of the hits,
            either a numpy ndarray or an array stored on the GPU
            This is the time in nano seconds multiplied with the speed of light.
        :type d_x: numpy ndarray or pycuda.driver.DeviceAllocation


        :returns: d_col_idx, d_prefix_sums, d_degrees
            d_col_idx, d_prefix_sums: The sparse matrix in CSR notation. d_col_idx stores the column indices,
            the size equals the number of correlations (or edges in the graph).
            d_prefix_sums stores per row, the start index of the row within the column index array. The size of d_prefix_sums is equal to the number of hits.
            d_degrees: The number of correlated hits per hit, stored as an array of size equal to the number of hits.

        :rtype: tuple( pycuda.driver.DeviceAllocation )

        """
        d_x = ready_input(x)
        d_y = ready_input(y)
        d_z = ready_input(z)
        d_ct = ready_input(ct)

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

        return d_col_idx, d_prefix_sums, d_degrees, total_correlated_hits


class PurgingSparse(object):
    """ class that provides an interface to the GPU Kernels used for Purging and maintains GPU state"""

    def __init__(self, N, cc):
        """instantiate PurgingSparse

        Create the object that provides an interface to the GPU kernel for performing the
        Purging algorithm on a sparse correlation table. This implementation of the
        algorithm uses the correlation table in a sparse manner, produced by the
        Quadratic Difference Sparse kernel. When this object is instantiated the
        CUDA kernel codes are compiled.

        :param N: The largest number of hits that are to be processed by one iteration
                of the quadratic difference algorithm.
        :type N: int

        :param cc: The CUDA compute capability of the target device as a string, consisting
                of the major and minor number concatenated without any separators.
        :type cc: string

        """

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



    def compute(self, col_idx, prefix_sums, degrees, shift=0):
        """ perform purging on a sparse matrix

        :param col_idx: A device allocation storing the column indices of the sparse matrix.
            The size of col_idx equals the number of correlations.
        :type col_idx: numpy.ndarray or pycuda.driver.DeviceAllocation

        :param prefix_sums: The start index of each row within the column index array.
            The size of prefix_sums is equal to the number of hits.
        :type prefix_sums: numpy.ndarray or pycuda.driver.DeviceAllocation

        :param degrees: The number of correlated hits per hit, stored as an array of size equal to the number of hits.
        :type degrees: numpy.ndarray or pycuda.driver.DeviceAllocation

        :param shift: Optional parameter that can be used to shift the indices of the nodes
            that remain after purging. This could be used when sliding through a larger time
            slice to convert the indices from within the current slice to a global index.
        :type shift: int

        :returns: The list of node indices of the nodes that remain after purging.
        :rtype: list ( int )

        """
        d_col_idx = ready_input(col_idx)
        d_prefix_sums = ready_input(prefix_sums)
        d_degrees = ready_input(degrees)

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

        #print("finished purging, iterations = ", counter)

        degrees = np.zeros(self.N).astype(np.int32)
        drv.memcpy_dtoh(degrees, d_degrees)
        if (current_num_nodes > 0):
            #print("found clique of size=", current_num_nodes)
            indices = np.array(range(degrees.size))
            found_indices = indices[degrees >= current_minimum]
            #print(found_indices + shift)
            return found_indices + shift

        return []


