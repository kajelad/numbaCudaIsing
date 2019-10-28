import numpy as np
import numba


@numba.vectorize([numba.int32(numba.int32)])
def log2(n):
    """
    fast integer floor log 2

    :param n: input integer, must be positive
    :return:
    """
    result = 0
    for i in range(1, 32):
        if not n >> i:
            result = i - 1
            break
    return result


# Nearest neighbor kernel, for standard cubic lattice interactions
def nearest_neighbor_kernel(ndim):
    """
    nearest neighbor interaction kernel for "standard" ferromagnetic
    nearest neighbor interactions with coupling 1.0

    :param ndim: number of diensions
    :type ndim: np.float32
    :return: np.ndarray[np.float64] nearest neighbor kernel
    """
    return -(
        (np.sum(np.indices((3,) * ndim) != 1, axis=0) == 1)
        .astype(np.float64)
    )



