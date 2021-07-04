""" Similar ctypes as defined in C/C++ code and used just for how much smaller/cleaner it is for me """
cimport numpy as np
np.import_array()

# define types (ctypedefs are iffy)
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef signed int int32_t
ctypedef unsigned char bool_t
ctypedef unsigned long long uint64_t

# create the type definition
ctypedef np.npy_bool NPBOOL_t
ctypedef np.uint8_t NPUINT_t
ctypedef np.int32_t NPINT32_t
ctypedef np.uint32_t NPUINT32_t
ctypedef np.longlong_t NPLONGLONG_t
ctypedef np.float32_t NPFLOAT_t
ctypedef np.double_t NPDOUBLE_t