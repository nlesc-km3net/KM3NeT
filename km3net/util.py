from __future__ import print_function
import os
import pandas
import numpy as np
import scipy.constants
from kernel_tuner import run_kernel

import pycuda.driver as drv

def get_kernel_path():
    """ function that returns the location of the CUDA kernels on disk

    :returns: the location of the CUDA kernels
    :rtype: string
    """
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

def get_slice(x_all, y_all, z_all, ct_all, N, shift):
    """ return a smaller slice of the whole timeslice

    :param x_all: The x-coordinates of the hits for the whole timeslice
    :type x_all: numpy ndarray of type numpy.float32

    :param y_all: The y-coordinates of the hits for the whole timeslice
    :type y_all: numpy ndarray of type numpy.float32

    :param z_all: The z-coordinates of the hits for the whole timeslice
    :type z_all: numpy ndarray of type numpy.float32

    :param ct_all: The ct values of the hits for the whole timeslice
    :type ct_all: numpy ndarray of type numpy.float32

    :param N: The number of hits that this smaller slice should contain
    :type N: int

    :param shift: The offset into the whole timeslice where this slice should start
    :type shift: int

    :returns: x,y,z,ct for the smaller slice
    :rtype: tuple(numpy ndarray of type numpy.float32)
    """
    x   = x_all[shift:shift+N]
    y   = y_all[shift:shift+N]
    z   = z_all[shift:shift+N]
    ct  = ct_all[shift:shift+N]
    return x,y,z,ct

def init_pycuda():
    """ helper func to init PyCuda

    :returns: The PyCuda context and a string containing the major and minor compute capability for the device
    :rtype: pycuda.driver.Context, string
    """
    drv.init()
    context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])
    return context, cc

def allocate_and_copy(arg):
    """ helper func to allocate and copy GPU memory

    :param arg: A numpy array that should be moved to GPU memory. This function
            will allocate GPU memory equal to the size of this array and copy the
            entire array into the newly allocated GPU memory.
    :type arg: numpy ndarray

    :returns: A PyCuda device allocation that represents the GPU memory allocation
    :rtype: pycuda.driver.DeviceAllocation
    """
    gpu_arg = drv.mem_alloc(arg.nbytes)
    drv.memcpy_htod(gpu_arg, arg)
    return gpu_arg

def generate_correlations_table(N, sliding_window_width, cutoff=2.87):
    """ generate input data with an expected density of correlated hits

    This function is for testing purposes. It generates a correlations
    table of size N by sliding_window_width, which is filled with zeros
    or ones when two hits are considered correlated.

    :param N: The number of hits to be considerd by this correlation table
    :type N: int

    :param sliding_window_width: The sliding window width used for this
            correlation table.
    :type sliding_window_width: int

    :param cutoff: The cutoff used for considering two hits correlated. This
            is actually the sigma of gaussian distribution, only values that are
            above the cutoff are considered a hit. Default value is 2.87, which
            should fill a correlations table with a density of roughly 0.0015.
    :type cutoff: float

    :returns: correlations table of size N by sliding_window_width
    :rtype: numpy ndarray of type numpy.uint8

    """
    correlations = np.random.randn(sliding_window_width, N)
    correlations[correlations <= cutoff] = 0
    correlations[correlations > cutoff] = 1
    correlations = np.array(correlations.reshape(sliding_window_width, N), dtype=np.uint8)
    #zero the triangle at the end of correlations table that cannot contain any ones
    for j in range(correlations.shape[0]):
        for i in range(N-j-1, N):
            correlations[j,i] = 0
    return correlations


def generate_large_correlations_table(N, sliding_window_width):
    """ generate a larget set of input data with an expected density of correlated hits

    This function is for testing purposes. It generates a large correlations
    table of size N by sliding_window_width, which is filled with zeros
    or ones when two hits are considered correlated. This function has no cutoff
    parameter but uses generate_input_data() to get input data. The correlations
    table is reconstructed on the GPU, for which a kernel is compiled and ran
    on the fly.

    :param N: The number of hits to be considerd by this correlation table
    :type N: int

    :param sliding_window_width: The sliding window width used for this
            correlation table.
    :type sliding_window_width: int

    :returns: correlations table of size N by sliding_window_width and an array
            storing the number of correlated hits per hit of size N.
    :rtype: numpy ndarray of type numpy.uint8, a numpy array of type numpy.int32

    """
    #generating a very large correlations table takes hours on the CPU
    #reconstruct input data on the GPU
    x,y,z,ct = generate_input_data(N)
    problem_size = (N,1)
    correlations = np.zeros((sliding_window_width, N), 'uint8')
    sums = np.zeros(N).astype(np.int32)
    args = [correlations, sums, N, sliding_window_width, x, y, z, ct]
    with open(get_kernel_path()+'quadratic_difference_linear.cu', 'r') as f:
        qd_string = f.read()
    data = run_kernel("quadratic_difference_linear", qd_string, problem_size, args, {"block_size_x": 512, "write_sums": 1})
    correlations = data[0]
    sums = data[1]

    #now I cant compute the node degrees on the CPU anymore, so using another GPU kernel
    with open(get_kernel_path()+'degrees.cu', 'r') as f:
        degrees_string = f.read()
    args = [sums, correlations, N]
    data = run_kernel("degrees_dense", degrees_string, problem_size, args, {"block_size_x": 512})
    sums = data[0]

    return correlations, sums

def create_sparse_matrix(correlations, sums):
    """ call GPU kernel to transform a correlations table into a spare matrix

    This function compiles the dense2sparse GPU kernel and calls it convert a
    densely stored correlations table into a sparsely stored correlation matrix.
    The sparse notation used is CSR.

    :param correlations: A correlations table of size N by sliding_window_width
    :type correlations: a 2d numpy array of type numpy.uint8

    :param sums: An array with the number of correlated hits per hit
    :type sums: numpy array of type numpy.int32

    :returns: This function returns three arrays that together form the sparse matrix

        * row_idx: the row index of each entry in the column index array
        * col_idx: the column index of each correlation in the sparse matrix
        * prefix_sums: the offset into the column index array for each row

    :rtype: numpy ndarray of type numpy.int32
    """
    N = np.int32(correlations.shape[0])
    prefix_sums = np.cumsum(sums).astype(np.int32)
    total_correlated_hits = np.sum(sums.sum())
    row_idx = np.zeros(total_correlated_hits).astype(np.int32)
    col_idx = np.zeros(total_correlated_hits).astype(np.int32)
    with open(get_kernel_path()+'dense2sparse.cu', 'r') as f:
        kernel_string = f.read()
    args = [row_idx, col_idx, prefix_sums, correlations, N]
    data = run_kernel("dense2sparse_kernel", kernel_string, (N,1), args, {"block_size_x": 256})
    return data[0], data[1], prefix_sums

def get_full_matrix(correlations):
    """ obtain a full correlation matrix from the correlations table

    This function should only be used for testing purposes on small
    correlations tables as the full correlations matrix is typically
    huge and nearly empty.

    :param correlations: A correlations table of size N by sliding_window_width
    :type correlations: a 2d numpy array of type numpy.uint8

    :returns: A full, densely stored, N by N correlations matrix
    :rtype: a 2d numpy array of type numpy.uint8
    """
    n = correlations.shape[1]
    matrix = np.zeros((n,n), dtype=np.uint8)
    for i in range(n):
        for j in range(correlations.shape[0]):
            if correlations[j,i] == 1:
                col = i+j+1
                if col < n and col >= 0:
                    matrix[i,col] = 1
                    matrix[col,i] = 1
    return matrix

def memcpy_dtoh(d_x, N, dtype):
    """ helper func to copy data from the GPU

    :param d_x: Array on the GPU
    :type d_x: pycuda.driver.DeviceAllocation

    :param N: Number of elements in GPU array
    :type N: int

    :param dtype: Type of the array
    :type dtype: numpy.dtype

    :returns: A numpy array copy of the GPU array of size N, and data type dtype.
    :rtype: numpy.ndarray

    """
    temp = np.zeros(N, dtype=dtype)
    drv.memcpy_dtoh(temp, d_x)
    return temp

def sparse_to_dense(prefix_sums, col_idx, N=None, hits=None):
    """ Convert a sparse matrix to a dense matrix

    Helper function to convert a sparse matrix in CSR format to a
    dense 2d matrix of size N by N.
    This function should only be used for testing purposes on small
    correlations tables as the full correlations matrix is typically
    huge and nearly empty.

    :param prefix_sums: This is the row_start array of the CSR format.
        It contains the start index of each row within the col_idx array. Note that
        row 0 starts on position 0 implicitly, so the the first number in the
        prefix_sums array is actually the start index of row 1, which is also
        the number of elements on row 0. The size of this array is equal to
        the number of rows.
    :type prefix_sums: numpy ndarray or pycuda.driver.DeviceAllocation

    :param col_idx: This array stores the column indices of the CSR format.
        The size of this array is equal to the number of elements in the matrix.
    :type col_idx: numpy ndarray or pycuda.driver.DeviceAllocation

    :param N: The number of hits, only needs to be passed when prefix_sums is
        passed as a pycuda.driver.DeviceAllocation instead of a numpy array
    :type N: int

    :param hits: The number of correlated hits, only needs to be passed when col_idx is
        passed as a pycuda.driver.DeviceAllocation instead of a numpy array
    :type hits: int

    :returns: A full densely stored correlation matrix of size N by N.
    :rtype: numpy ndarray
    """
    if isinstance(prefix_sums, drv.DeviceAllocation):
        prefix_sums = memcpy_dtoh(prefix_sums, N, np.int32)
    if isinstance(col_idx, drv.DeviceAllocation):
        col_idx = memcpy_dtoh(col_idx, hits, np.int32)

    N = np.int32(prefix_sums.size)
    matrix = np.zeros((N,N), dtype=np.uint8)
    start = 0
    for row in range(N):
        end = prefix_sums[row]
        for i in range(start,end):
            matrix[row,col_idx[i]] = 1
        start = end
    return matrix

def generate_input_data(N, factor=2000.0):
    """ generate input data

    This function generates hits stored as x,y,z-coordinates
    and ct values from random noise. The default factor should
    result in a density of about 0.002 when using a sliding
    window width of 1500, where density is defined
    as the total number of correlated hits / (N*sliding_window_width).

    :param N: The number of hits to generate
    :type N: int

    :param factor: Optionally specify a factor to modify the correlation
            density of the hits. Default=2000.0
    :type factor: float

    :returns: N hits stored as x,y,z,ct
    :rtype: tuple(numpy ndarray of type numpy.float32)

    """
    x = np.random.normal(0.2, 0.1, N).astype(np.float32)
    y = np.random.normal(0.2, 0.1, N).astype(np.float32)
    z = np.random.normal(0.2, 0.1, N).astype(np.float32)
    ct = (factor*np.random.normal(0.5, 0.06, N)).astype(np.float32)  #replace 2000 with 5 for the new density
    return x,y,z,ct


def get_real_input_data(filename):
    """ Read input data from disk

    Read a timeslice of input data from a file stored on disk.
    The file format to be used is a text file that stores one
    hit per row in a text file. The first column stores the
    time the hit occured in nanoseconds. Followed by three
    columns that store the x,y,z coordinates of where the
    hit was measured in meters. The hits are assumed to be
    stored in ascending order by the time the hit occured, so
    earliest hit first.

    This routine also multiplies the time values with the speed of light.
    These values are therefore called ct and are stored in meters.

    :param filename: The path and the filename of the file that contains the input data.
    :type filename: string

    :returns: N,x,y,z,ct. N is the number of hits that were retrieved from the file.
            x,y,z are the coordinates of the hit in meters and ct the time multiplied
            by the speed of light, also in meters.
    :rtype: tuple(int, numpy ndarray of type numpy.float32)
    """
    data = pandas.read_csv(filename, sep=' ', header=None)

    t = np.array(data[0]).astype(np.float32)
    t = t / 1e9 # convert from nano seconds to seconds
    ct = t * scipy.constants.c

    #x,y,z positions of the hits, assuming these are in meters
    x = np.array(data[1]).astype(np.float32)
    y = np.array(data[2]).astype(np.float32)
    z = np.array(data[3]).astype(np.float32)

    N = np.int32(x.size)
    return N,x,y,z,ct


def correlations_cpu_3B(correlations, x, y, z, ct, roadwidth=90.0, tmax=0.0):
    """ function for computing the reference answer using only the 3B condition

    This function computes the Match 3B criterion instead of the quadratic
    difference criterion. The 3B criterion is similar to the quadratic difference
    criterion, but is also considers a maximum distance for two hits to be
    correlated. This distance is based on the parameter 'roadwidth', which
    is the assumed maximum distance a photon can travel through seawater.

    :param correlations: A correlations table of size N by sliding_window_width
        used for storing the result, which is also returned.
    :type correlations: a 2d numpy array of type numpy.uint8

    :param x: The x-coordinates of the hits
    :type x: numpy ndarray of type numpy.float32

    :param y: The y-coordinates of the hits
    :type y: numpy ndarray of type numpy.float32

    :param z: The z-coordinates of the hits
    :type z: numpy ndarray of type numpy.float32

    :param ct: The ct values of the hits
    :type ct: numpy ndarray of type numpy.float32

    :param roadwidth: The roadwidth used in the 3B criterion, the assumed
        distance a photon can travel through seawater. Default is 90.0.
    :type roadwidth: float

    :param tmax: The maximum time between two hits for them to always be
        considered correlated. By default 0.0.
    :type tmax: float

    :returns: correlations table of size N by sliding_window_width.
    :rtype: numpy 2d array of type numpy.uint8
    """
    index_of_refrac = 1.3800851282 #also known as theta, angle of emitted Cherenkov light
    tan_theta_c     = np.sqrt((index_of_refrac-1.0) * (index_of_refrac+1.0) )
    cos_theta_c     = 1.0 / index_of_refrac
    sin_theta_c     = tan_theta_c * cos_theta_c
    c               = 0.299792458  # m/ns, use scipy.constants.c instead
    inverse_c       = 1.0/c

    TMaxExtra = tmax

    tt2 = tan_theta_c**2
    D0 = roadwidth          # in meters
    D1 = roadwidth*2.0
    D2 = roadwidth * 0.5 * np.sqrt(tt2 + 10.0 + 9.0/tt2)

    D02 = D0**2
    D12 = D1**2
    D22 = D2**2

    R = roadwidth
    Rs = R * sin_theta_c

    R2 = R**2
    Rs2 = Rs**2
    Rst = Rs * tan_theta_c
    Rt = R * tan_theta_c

    #our ct is in meters, convert back to nanoseconds
    t = ct * inverse_c
    def test_3B_condition(t1,x1,y1,z1, t2,x2,y2,z2):
        diffx = x1-x2
        diffy = y2-y2
        diffz = z2-z2
        d2 = diffx**2 + diffy**2 + diffz**2
        difft = np.absolute(t1 - t2)

        if d2 < D02:
            dmax = np.sqrt(d2) * index_of_refrac
        else:
            dmax = np.sqrt(d2 - Rs2) + Rst

        if difft > dmax * inverse_c + TMaxExtra:
            return False

        if d2 > D22:
            dmin = np.sqrt(d2 - R2) - Rt
        elif d2 > D12:
            dmin = np.sqrt(d2 - D12)
        else:
            return True

        return difft >= dmin * inverse_c - TMaxExtra

    for i in range(correlations.shape[1]):
        for j in range(i + 1, i + correlations.shape[0] + 1):
            if j < correlations.shape[1]:
                if test_3B_condition(t[i],x[i],y[i],z[i],t[j],x[j],y[j],z[j]):
                   correlations[j - i - 1, i] = 1
    return correlations


def correlations_cpu(correlations, x, y, z, ct):
    """ function for computing the reference answer

    This function is the CPU version of the quadratic difference algorithm.
    It computes the correlations based on the quadratic difference criterion.
    This function is mainly for testing and verification, for large datasets
    use the GPU kernel.

    :param correlations: A correlations table of size sliding_window_width by N
        used for storing the result, which is also returned.
    :type correlations: a 2d numpy array of type numpy.uint8

    :param x: The x-coordinates of the hits
    :type x: numpy ndarray of type numpy.float32

    :param y: The y-coordinates of the hits
    :type y: numpy ndarray of type numpy.float32

    :param z: The z-coordinates of the hits
    :type z: numpy ndarray of type numpy.float32

    :param ct: The ct values of the hits
    :type ct: numpy ndarray of type numpy.float32

    :returns: correlations table of size sliding_window_width by N.
    :rtype: numpy 2d array
    """
    for i in range(correlations.shape[1]):
        for j in range(i + 1, i + correlations.shape[0] + 1):
            if j < correlations.shape[1]:
                if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   correlations[j - i - 1, i] = 1
    return correlations

def insert_clique(dense_matrix, sliding_window_width=1500, clique_size=10):
    """ generate clique indices and insert into dense matrix

    Insert a clique into a dense matrix for testing purposes. This function attemps to
    generate a clique of size clique_size, but may generate a clique that is slightly
    smaller.

    :param dense_matrix: A densly stored correlation matrix, size N by N.
    :type dense_matrix: 2d numpy array

    :param sliding_window_width: the sliding window width, 1500 by default.
    :type sliding_window_width: int

    :param clique_size: The size of the clique that is to be insered into the data, 10 by default.
    :type clique_size: int

    :returns: The dense matrix, a list of clique indices, and the size of the inserted clique.
    :rtype: tuple(numpy.ndarray, list, int)
    """
    #generate clique indices at most sliding_window_width apart
    clique_indices = sorted((np.random.rand(clique_size) * float(sliding_window_width)).astype(np.int))
    #shift it to somewhere in the middle
    clique_indices += sliding_window_width
    clique_indices = np.unique(clique_indices)
    #may contain the same index multiple times, reduce clique_size if needed
    clique_size = len(clique_indices)
    for i in clique_indices:
        for j in clique_indices:
            if not i == j:
                dense_matrix[i,j] = 1
                dense_matrix[j,i] = 1
    return (dense_matrix, clique_indices, clique_size)
