from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

import numpy as np
import os

# name the module
MODULE = 'calibration'
LANG_LEVEL = 3

# special cases
SPECIAL = {}


def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith('.pyx'):
            files.append(path.replace(os.path.sep, '.')[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def clean_path(path):
    if path.startswith('.'):
        return clean_path(path[1:])
    return path


def make_ext(ext):
    path = clean_path(ext).replace('.', os.path.sep) + '.pyx'
    name = MODULE + '.' + os.path.basename(clean_path(ext).replace('.', os.path.sep)).replace(os.path.sep, '')
    special = SPECIAL[name] if name in SPECIAL else {}
    return Extension(
        clean_path(ext) if special.get('name', None) is None else special.get('name', None),
        [path] + special.get('sources', []),
        include_dirs=['.', np.get_include()] + special.get('includes', []),
        extra_compile_args=['-O3', '-Wall'] + special.get('compile', []),
        extra_link_args=['-g'],
        libraries=[]
    )


# this is just for the cython files and is not designed to actually build the all-in-one application yet
setup(
    name='em-calibration',
    version='1.0',
    description='Najafian Lab Image Electron Microscopy Image Calibration',
    author='David Smerkous, Yu Fang',
    author_email='smerkd@uw.edu, fangy35@uw.edu',
    include_package_data=True,
    packages=find_packages(include=['calibration', 'calibration.*']),
    install_requires=[
        'numpy>=1.14.5',
        'opencv-contrib-python>=4.3.0.36',
        'scikit-image>=0.18.1',
        'scikit-learn>=0.24.0',
        'cython>=0.29.0',
        'six'
    ],
    ext_modules=cythonize(
        [make_ext(ext) for ext in scandir('.')],
        language_level=LANG_LEVEL,
        # nthreads=1# ,
        # gdb_debug=True
    )
)