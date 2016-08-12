import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "km3net",
    version = "0.0.1",
    author = "Ben van Werkhoven",
    author_email = "b.vanwerkhoven@esciencecenter.nl",
    description = ("Software for the real-time detection of neutrinos"),
    license = "Apache 2.0",
    keywords = "",
    packages=['km3net'],
    package_dir={'km3net': 'km3net'},
    package_data={"km3net": ['kernels/*.cu']},
    long_description=read('README.md'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Development Status :: 2 - Pre-Alpha',
    ],
)
