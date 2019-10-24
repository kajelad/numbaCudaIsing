import numpy as np
from numba import cuda

nearest_neighbor_kernel = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, -1.0],
    [0.0, -1.0, 0.0]
])


class IsingModel():
    """
    Ising model istance
    """
    

    def __init__(self, shape, coupling=None, field=0):
        """
        initializer

        args:
            shape (tuple of int32): number of spins in each dimension
            coupling (ndarray of float64): Coupling constants, encoded
                as a (2*r+1)x(2*r+1)x...x(2*r+1) array, where r is a
                fixed maximum range. element [r,r,...,r] denotes zero
                distance. Zero elements automatically detected.
            field (float64): applied field.
        """
        self.num_dim = len(shape)
        self.shape = shape
        self.num_spins = np.prod(self.shape)
        if self.num_spins % 8 != 0:
            raise ValueError("Size must be multiple of 8")
        self.spins = cuda.device_array((self.num_spins // 8,), np.uint8)

        if coupling is None:
            coupling = np.empty(np.zeros(self.num_dim), dtype=np.float64)
        self.coupling_indices = cuda.to_device(np.vstack(np.where(coupling != 0.0)).T)
        self.coupling_constants = cuda.to_device(coupling[np.where(coupling != 0.0)])
    



