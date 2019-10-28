import numpy as np
import numba
from numba import cuda


@cuda.jit(numba.uint8(numba.uint8), device=True, inline=True)
def hamming(n):
    """
    Hamming weight, i.e. number of set bits

    :param n: np.uint8
    :return:  np.uint8
    """
    # recursively divide in two, combinig sums by bit shifting and adding
    n = (n & np.uint8(85)) + ((n >> 1) & np.uint8(85))  # 85=01010101b
    n = (n & np.uint8(51)) + ((n >> 2) & np.uint8(51))  # 51=00110011b
    n = (n & np.uint8(15)) + ((n >> 4) & np.uint8(15))  # 15=00001111b
    return n


@cuda.jit(
    numba.float64(
        numba.int32, numba.uint8[:], numba.int32[:],
        numba.int32[:, :], numba.float64[:]
    ),
    device=True, inline=True
)
def calc_single_interaction_energy(
        index, spins, shape_shifts,
        coupling_indices, coupling_constants
):
    energy = 0.0
    num_dim = shape_shifts.size
    num_neighbors = coupling_constants.size
    for n_index in range(num_neighbors):
        other_index = 0
        for a in range(num_dim):
            other_index <<= shape_shifts[a]
            other_index += (
                    coupling_indices[n_index, a] +
                    (index & ((1 << shape_shifts[a]) - 1)) &
                    ((1 << shape_shifts[a]) - 1)
            )
        this_spin = (spins[index >> 3] >> (index & 7)) & 1
        other_spin = (spins[other_index >> 3] >> (other_index & 7)) & 1
        energy += coupling_constants[n_index] * this_spin * other_spin
    return energy


@cuda.jit(
    numba.float64(
        numba.uint8[:], numba.int32[:], numba.float64, numba.float64,
        numba.int32[:, :], numba.float64[:],
        numba.int32[:], numba.bool_[:]
    ),
)
def metropolis_step(
        spins, shape_shifts, temperature, field,
        coupling_indices, coupling_constants,
        block_shifts, offsets
):
    """
    Multi-spin metropolis algorthm

    :param spins: spin configuration; stored as np.uint8 bytes
    :param shape_shifts: shape of lattice, as power of 2
    :param temperature: unitless temperature
    :param field: unitless applied field
    :param coupling_indices:
    :param coupling_constants:
    :param block_shifts: shape of subsubdivisions, as power of 2
    :param offsets: offsets for subdivisions
    :return:
    """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    bindex = bx * bw + tx
    num_dim = shape_shifts.size
    shift = 0
    index = 0
    for a in range(num_dim-1, -1, -1):
        multi_index = (
                bindex & ((1 << (shape_shifts[a] - block_shifts[a] - 1)) - 1)
        )
        multi_index <<= block_shifts[a] + 1
        multi_index += (1 << block_shifts[a]) * offsets[a]
        multi_index += np.random.randint(0, 1 << block_shifts[a])
        index += multi_index << shift
        shift += shape_shifts[a]
        bindex >>= shape_shifts[a] - block_shifts[a] - 1
    delta_E = -2 * calc_single_interaction_energy(
        index, spins, shape_shifts, coupling_indices, coupling_constants
    )
    this_spin = (spins[index >> 3] >> (index & 7)) & 1
    delta_E -= 2 * field * this_spin
    if np.random.rand() < np.exp(-delta_E / temperature):
        spins[index >> 3] ^= 1 << (index & 7)
