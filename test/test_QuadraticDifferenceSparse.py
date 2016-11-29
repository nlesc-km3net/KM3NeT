import pycuda.driver as drv
import numpy as np

from scipy.sparse import csr_matrix
from km3net.kernels import QuadraticDifferenceSparse
import km3net.util as util

from .context import create_plot

def test_QuadraticDifferenceSparse():

    N = 500
    window_width = 150
    x,y,z,ct = util.generate_input_data(N)

    try:
        context, cc = util.init_pycuda()

        qd_kernel = QuadraticDifferenceSparse(N, window_width, cc)

        d_col_idx, d_prefix_sums, d_degrees, total_hits = qd_kernel.compute(x, y, z, ct)

        #copy GPU data to the host
        prefix_sums = np.zeros(N, dtype=np.int32)
        drv.memcpy_dtoh(prefix_sums, d_prefix_sums)

        print(prefix_sums.size)
        print(prefix_sums)

        col_idx = np.zeros(total_hits, dtype=np.int32)
        drv.memcpy_dtoh(col_idx, d_col_idx)

        print(col_idx.size)
        print(col_idx)

    finally:
        context.pop()

    correlations = np.zeros((window_width, N), dtype=np.uint8)
    correlations = util.correlations_cpu(correlations, x, y, z, ct)

    reference = csr_matrix(util.get_full_matrix(correlations), shape=(N,N))
    answer = csr_matrix(util.sparse_to_dense(prefix_sums, col_idx), shape=(N,N))

    print(total_hits)
    print(reference.sum())
    print(answer.sum())

    diff = reference - answer

    print("diff")
    print(list(zip(diff.nonzero()[0], diff.nonzero()[1])))

    assert diff.nnz == 0
