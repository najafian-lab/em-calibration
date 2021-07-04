import numpy as np

from calibration.types cimport *

cdef class DArrayList(object):
    cdef np.ndarray data
    cdef int capacity
    cdef int size
    
    # funcs
    cpdef update(self, NPDOUBLE_t[:] row)
    cpdef add(self, NPDOUBLE_t x)
    cpdef np.ndarray[NPDOUBLE_t, mode='c'] finalize(self)