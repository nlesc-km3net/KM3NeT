#!/usr/bin/env python

from __future__ import print_function
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
from numba import jit

from pycuda.compiler import SourceModule
import pycuda.driver
from test.context import skip_if_no_cuda_device, get_kernel_path
from test.context import degrees_dense_cpu, correlations_cpu

skip_if_no_cuda_device()

N = np.int32(4500000)
sliding_window_width = np.int32(1500)
block_size_x = 450
block_size_y = 1
# generate input data with an expected density of correlated hits
x = np.random.normal(0.2, 0.1, N).astype(np.float32)
y = np.random.normal(0.2, 0.1, N).astype(np.float32)
z = np.random.normal(0.2, 0.1, N).astype(np.float32)
ct = 0.001 * np.random.normal(0.0005, 0.06, N).astype(np.float32)
correlations_ref = np.zeros((sliding_window_width, N), 'uint8')
correlations = np.zeros((sliding_window_width, N), 'uint8')
sums = np.zeros(N).astype(np.int32)

#### LOAD KERNEL STRINGS #### 
with open(get_kernel_path() + 'headers.cu', 'r') as f:
    kernels_string = f.read()

kernels_string = kernels_string % \
                {'tile_size_x_qd':1,
                 'block_size_x_qd': block_size_x,
                 'block_size_y_qd':block_size_y,
                 'window_width':sliding_window_width,
                 'write_sums_qd':0,
                 'block_size_x_d':block_size_x}

with open(get_kernel_path() + 'quadratic_difference_linear.cu', 'r') as f:
    kernels_string += "\n" + f.read()
with open(get_kernel_path() + 'degrees.cu', 'r') as f:
    kernels_string += "\n" + f.read()

kernel_mod = SourceModule(kernels_string)
quadratic_difference_linear= kernel_mod.get_function("quadratic_difference_linear")

#### MEMORY ALLOCATION for: x, y, z, ct, sums arrays, and dense correlations matrix. ####
start_malloc = time.time()
pycuda.driver.start_profiler()
x_gpu = drv.mem_alloc(x.nbytes)
y_gpu = drv.mem_alloc(y.nbytes)
z_gpu = drv.mem_alloc(z.nbytes)
ct_gpu = drv.mem_alloc(ct.nbytes)
sums_gpu = drv.mem_alloc(sums.nbytes)
correlations_gpu = drv.mem_alloc(correlations.nbytes)
# memory you allocate on the GPU is not clean, and may contain results from previous runs. Therefore must MEMSET.
# Correlations matrix is too large to be set in one go, API (PyCuda or also Cuda) expects 'unsigned int' parameter
# https://github.com/inducer/pycuda/blob/master/src/wrapper/wrap_cudadrv.cpp
# so maximum value expected should be 4294967295. For 4.5e6x1500=6750000000, it will overflow.
# We probably need to set it piece by piece.
# drv.memset(destination, data, count) # count in number of elements, not bytes
# drv.memset_d8(correlations_gpu, 0, correlations.shape[0]*correlations.shape[1])
# How about instead of d8,, we do d32 on 4 times less data.

drv.memset_d32(correlations_gpu, 0, (correlations.shape[0]*correlations.shape[1])/4) # this works!
drv.memset_d32(sums_gpu, 0, sums.shape[0])
end_malloc = time.time()

print('Memory allocation on device took {0:.2e}s.\n'.format(end_malloc -start_malloc))
print("Number of bytes needed for the correlation matrix = {0:.3e} \n".format(correlations.nbytes))


#### MEMORY TRANFER ####
start_transfer = time.time()
drv.memcpy_htod(x_gpu, x)
drv.memcpy_htod(y_gpu, y)
drv.memcpy_htod(z_gpu, z)
drv.memcpy_htod(ct_gpu, ct)
#drv.memcpy_htod(correlations_gpu, correlations)
#drv.memcpy_htod(sums_gpu, sums)
end_transfer = time.time()
print('Data transfer from host to device took {0:.2e}s.\n'.format(end_transfer -start_transfer))


#### RUN KERNEL quadratic_difference_linear ####
gridx = int(np.ceil(correlations.shape[1]/block_size_x))
gridy = int(1)
print("gridx:", gridx)

start = drv.Event()
end = drv.Event()

pycuda.autoinit.context.synchronize()

start.record() # start timing
quadratic_difference_linear(
        correlations_gpu, sums_gpu, N, sliding_window_width, x_gpu, y_gpu, z_gpu, ct_gpu, 
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

pycuda.autoinit.context.synchronize()

end.record() # end timing
end.synchronize()

secs = start.time_till(end)*1e-3
print('Time taken for GPU computations of first kernel is {0:.2e}s.\n'.format(secs))


#### TRANSFER BACK (we won't need this later) ####
start_transfer = time.time()
drv.memcpy_dtoh(correlations, correlations_gpu)
pycuda.driver.stop_profiler()
end_transfer = time.time()
print('Data transfer from device to host took {0:.2e}s.\n'.format(end_transfer -start_transfer))

print("Computing ground truth on CPU....")
correlations_ref = correlations_cpu(correlations_ref, x, y, z, ct)
check = np.sum(correlations - correlations_ref)
print("Done.")
print('np.sum(correlations - correlations_ref) = ', check)
print('np.sum(correlations) = ', correlations.sum())
print('np.sum(correlations_ref) = ', correlations_ref.sum())
## SECOND KERNEL ####
degrees_dense = kernel_mod.get_function("degrees_dense")
#### RUN KERNEL degrees ####

pycuda.autoinit.context.synchronize()
start.record() # start timing
degrees_dense(
        sums_gpu, correlations_gpu, N,
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

pycuda.autoinit.context.synchronize()

end.record() # end timing
end.synchronize()

secs = start.time_till(end)*1e-3
print('Time taken for GPU computations of second kernel is {0:.2e}s.\n'.format(secs))

#### TRANSFER BACK (we won't need this later) ####
start_transfer = time.time()
drv.memcpy_dtoh(sums, sums_gpu)
pycuda.driver.stop_profiler()
end_transfer = time.time()

print('Data transfer from device to host took {0:.2e}s.\n'.format(end_transfer -start_transfer))

print("Computing ground truth on CPU....")
#sums_ref = np.zeros_like(sums)
sums_ref = degrees_dense_cpu(correlations_ref)
print("Done.")
print('np.sum(sums - sums_ref) = ', np.sum(sums - sums_ref))
print('np.sum(sums) = ', sums.sum())
print('np.sum(sums_ref) = ', sums_ref.sum())

print('correlations = \n', correlations)
# print('sums = \n', sums)
print('correlations_ref\n', correlations_ref)
# print("sums_ref = \n", sums_ref)