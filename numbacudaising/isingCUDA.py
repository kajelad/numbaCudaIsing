import math
import numpy as np
import numba
from numba import cuda
from numba.cuda import random as ncrand


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
        this_spin = 2 * ((spins[index >> 3] >> (index & 7)) & 1) - 1
        other_spin = 2 * (
                    (spins[other_index >> 3] >> (other_index & 7)) & 1) - 1
        energy += coupling_constants[n_index] * this_spin * other_spin
    return energy


@cuda.jit
def calc_energy(spins, shape_shifts, coupling_indices, coupling_constants,
                energies):
    index = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    energies[index] += calc_single_interaction_energy(
        index, spins, shape_shifts, coupling_indices, coupling_constants
    )
    energies[index] += pm * ncrand.xoroshiro128p_uniform_float64(rng_states,
                                                                 index)


@cuda.jit
def random_flip(
        spins, shape_shifts, rng_states
):
    tindex = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    index = 0
    num_dim = shape_shifts.size
    for a in range(num_dim - 1, -1, -1):
        index <<= shape_shifts[a]
        index += np.int64(
            ncrand.xoroshiro128p_uniform_float64(rng_states, tindex) *
            (1 << shape_shifts[a])
        )
    spins[index >> 3] ^= 1 << (index & 7)


@cuda.jit
def metropolis_step(
        spins, shape_shifts, temperature, field,
        coupling_indices, coupling_constants,
        block_shifts, offsets, rng_states
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
    :param rng_states: numba.cuda.random rng states
    :return:
    """
    thread_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    temp_index = thread_index
    num_dim = shape_shifts.size
    shift = 0
    spin_index = 0
    for a in range(num_dim - 1, -1, -1):
        spin_index <<= shape_shifts[a]
        spin_index += (
                (temp_index & (
                            (1 << shape_shifts[a] - block_shifts[a] - 1) - 1))
                << (block_shifts[a] + 1)
        )
        spin_index += np.int64(offsets[a]) << block_shifts[a]
        spin_index += np.int64(
            ncrand.xoroshiro128p_uniform_float64(rng_states, thread_index) *
            (1 << block_shifts[a])
        )
        temp_index >>= block_shifts[a] + 1

    delta_E = -2 * calc_single_interaction_energy(
        spin_index, spins, shape_shifts, coupling_indices, coupling_constants
    )
    this_spin = (spins[spin_index >> 3] >> (spin_index & 7)) & 1
    delta_E -= 2 * field * this_spin
    """
    if (
            ncrand.xoroshiro128p_uniform_float64(rng_states, thread_index) <
            math.exp(-delta_E / temperature)
    ):
    """
    if True:
        spins[spin_index >> 3] ^= 1 << (spin_index & 7)
