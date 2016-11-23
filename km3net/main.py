#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import pycuda.driver as drv
from pycuda.compiler import SourceModule

from km3net.util import *
from km3net.kernels import *

def main():

    #problem size
    N = np.int32(3000)
    problem_size = (N, 1)
    sliding_window_width = np.int32(1500)

    #compile GPU functions
    context, cc = init_pycuda()
    qd_kernel = QuadraticDifferenceSparse(N, sliding_window_width, cc)
    purging = PurgingSparse(N, cc)

    #get input data
    print("Reading input data")
    N_all,x_all,y_all,z_all,ct_all = get_real_input_data('event0.txt')

    shift = 0
    x,y,z,ct = get_slice(x_all, y_all, z_all, ct_all, N, shift)

    #allocate GPU input arrays
    d_x = drv.mem_alloc(x.nbytes)
    d_y = drv.mem_alloc(y.nbytes)
    d_z = drv.mem_alloc(z.nbytes)
    d_ct = drv.mem_alloc(ct.nbytes)

    step_size = N-sliding_window_width
    counter = 0
    found_cliques = 0
    total_clique_size = 0

    for i in range(N_all//step_size):
        counter += 1
        shift = i * step_size
        x,y,z,ct = get_slice(x_all, y_all, z_all, ct_all, N, shift)

        #copy data to the gpu
        drv.memcpy_htod(d_x, x)
        drv.memcpy_htod(d_x, y)
        drv.memcpy_htod(d_x, z)
        drv.memcpy_htod(d_x, ct)

        #call the quadratic difference kernel twice to build sparse matrix
        d_col_idx, d_prefix_sums, d_degrees = qd_kernel.compute(d_x, d_y, d_z, d_ct)

        #use purging algorithm to find clique
        clique = purging.compute(d_col_idx, d_prefix_sums, d_degrees, shift)
        if clique.size > 0:
            found_cliques += 1
            total_clique_size += clique.size

    print("processing timeslice finished")
    print("found", found_cliques, "cliques in", counter, "slices")
    print("average clique size", total_clique_size/float(found_cliques))

    context.pop()


if __name__ == "__main__":
    main()


