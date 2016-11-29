import os

def get_kernel_path():
    """ get path to the kernels as a string

    This function is a duplicate of get_kernel_path in km3net.util. The reason
    for this is that we want the tuning scripts to use the kernel located in the
    directory tree of the git repository, rather than the kernels that were
    installed with the install script when the km3net package was installed.
    """
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/km3net/kernels/'

