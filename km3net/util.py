import os
import pandas
import numpy as np
import scipy.constants

import pycuda.driver as drv

data_dir = '/var/scratch/bwn200/KM3Net/'

def get_kernel_path():
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

def get_real_input_data(filename):
    data = pandas.read_csv(data_dir + filename, sep=' ', header=None)

    t = np.array(data[0]).astype(np.float32)
    t = t / 1e9 # convert from nano seconds to seconds
    ct = t * scipy.constants.c

    #x,y,z positions of the hits, assuming these are in meters
    x = np.array(data[1]).astype(np.float32)
    y = np.array(data[2]).astype(np.float32)
    z = np.array(data[3]).astype(np.float32)

    #print("x", x[:10], x.min(), x.max())
    #print("y", y[:10], y.min(), y.max())
    #print("z", z[:10], z.min(), z.max())
    #print("ct", ct[:10], ct.min(), ct.max())
    #print("ct diff", np.diff(ct[:11]))

    N = np.int32(x.size)
    return N,x,y,z,ct

def get_slice(x_all, y_all, z_all, ct_all, N, shift):
    x   = x_all[shift:shift+N]
    y   = y_all[shift:shift+N]
    z   = z_all[shift:shift+N]
    ct  = ct_all[shift:shift+N]
    return x,y,z,ct

def init_pycuda():
    drv.init()
    context = drv.Device(0).make_context()
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])
    return context, cc

def allocate_and_copy(arg):
    gpu_arg = drv.mem_alloc(arg.nbytes)
    drv.memcpy_htod(gpu_arg, arg)
    return gpu_arg




