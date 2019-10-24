import numpy as np
from numba import cuda


cuda.jit(
    numba.float64(numba.int32, numba.uint8[:], numba.int32[:],  numba.int32[:,:], numba.float64[:]),
    device=True, inline=True
)
def calc_single_interaction_energy(index, spins, shape, coupling_indices, coupling_constants):
    energy = 0.0
    num_dim = shape.size
    num_neighbors = coupling_constants.size
    for n_index in range(num_neighbors):
        delta = coupling_indices[n_index, 0]
        for a in range(1, num_dim):
            delta *= shape[a-1]
            delta += coupling_indices[n_index, a]
        this_spin = (spins[index // 8] >> (index % 8)) & 1
        other_index = index + delta
        other_spin = (spins[other_index // 8] >> (other_index % 8)) & 1
        energy += coupling_constants[n_index] * this_spin * other_spin
    return spin
