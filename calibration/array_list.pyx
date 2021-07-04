# distutils: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np

# cython imports
from calibration.types cimport *
cimport cython
cimport numpy as np
np.import_array()


cdef class DArrayList(object):
    def __cinit__(self, int initial_size):
        self.data = np.zeros((initial_size,))
        self.capacity = initial_size
        self.size = 0

    cpdef update(self, NPDOUBLE_t[:] row):
        for r in row:
            self.add(r)

    cpdef add(self, NPDOUBLE_t x):
        cdef np.ndarray newdata 
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity,))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    cpdef np.ndarray[NPDOUBLE_t, mode='c'] finalize(self):
        cdef np.ndarray data = self.data[:self.size]
        return np.ascontiguousarray(data)
