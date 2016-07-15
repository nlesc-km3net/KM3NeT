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

skip_if_no_cuda_device()
N = np.int32(27)
sliding_window_width = np.int32(9)
problem_size = (N, 1)

#generate input data with an expected density of correlated hits
x = np.random.normal(0.2, 0.1, N).astype(np.float32)
y = np.random.normal(0.2, 0.1, N).astype(np.float32)
z = np.random.normal(0.2, 0.1, N).astype(np.float32)
ct = 10*np.random.normal(0.5, 0.06, N).astype(np.float32)
correlations = np.zeros((sliding_window_width, N), 'uint8')
sums = np.zeros(N).astype(np.int32) # TODO: do we really need int32? 

with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
    kernel_string = f.read()

kernel_mod = SourceModule(kernel_string)
# TODO: append other kernels to the string
quadratic_difference_linear= kernel_mod.get_function("quadratic_difference_linear")

#### MEMORY ALLOCATION ####
start_malloc = time.time()

pycuda.driver.start_profiler()
x_gpu = drv.mem_alloc(x.nbytes)
y_gpu = drv.mem_alloc(y.nbytes)
z_gpu = drv.mem_alloc(z.nbytes)
ct_gpu = drv.mem_alloc(ct.nbytes)
sums_gpu = drv.mem_alloc(sums.nbytes)
correlations_gpu = drv.mem_alloc(correlations.nbytes)
# memory you allocate on the GPU is not clean, and may contain results from previous runs. Therefore:
drv.memset_d8(correlations_gpu, 0, correlations.shape[0]*correlations.shape[1])
drv.memset_d32(sums_gpu, 0, sums.shape[0]) # TODO: do we really need int32
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



block_size_x = 9
block_size_y = 1
gridx = int(np.ceil(correlations.shape[1]/block_size_x))
gridy = int(1)


#### RUN KERNEL quadratic_difference_linear ####
# create two timers so we can speed-test each approach
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
# drv.memcpy_dtoh(sums, sums_gpu)
pycuda.driver.stop_profiler()
end_transfer = time.time()

print('Data transfer from device to host took {0:.2e}s.\n'.format(end_transfer -start_transfer))
print('correlations = \n', correlations)

### SECOND KERNEL ####
with open(get_kernel_path()+'degrees.cu', 'r') as f:
    kernel_string = f.read()

degrees_mod = SourceModule(kernel_string)
degrees_dense = degrees_mod.get_function("degrees_dense")

#### RUN KERNEL degrees ####
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
print('sums = \n', sums)
