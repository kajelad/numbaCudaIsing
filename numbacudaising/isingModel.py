import numpy as np
from numba import cuda
from numba.cuda import random as ncrand
from matplotlib import pyplot as plt

from . import numbaCudaIsingLib as ncil
from . import isingCUDA as ic


class IsingModel():
    """
    Ising model istance
    """

    def __init__(self, shape, blocks, coupling=None, temperature=0, field=0):
        """

        :param shape: number of spins along each axis
        :param blocks: size of subdivisions along each axis, for parallel
            computation
        :param coupling: coupling array, must be (2*r+1)x...x(2*r+1) with
            (r,...,r) denoting the center.
        :param temperature: unitless temperature
        :param field: unitless applied field
        """
        self.num_dim = len(shape)
        self.shape = np.array(shape, dtype=np.int32)
        self.num_spins = np.prod(self.shape)
        self.blocks = np.array(blocks, dtype=np.int32)

        if np.any(self.shape & (self.shape - 1)):
            raise ValueError("Shape must consist of powers of 2,")
        if self.shape[-1] % 8:
            raise ValueError("Last shape must be multiple of 8")
        if np.any(self.shape % self.blocks):
            raise ValueError("Blocks do not evenly divide shape")

        self.shape_shifts = cuda.to_device(ncil.log2(self.shape))
        self.block_shifts = cuda.to_device(ncil.log2(self.blocks))
        self.spins = cuda.device_array((self.num_spins // 8,), np.uint8)
        self.spins[:] = np.random.randint(
            0, 256, size=self.num_spins >> 3, dtype=np.uint8
        )

        if coupling is None:
            coupling = np.empty(np.zeros(self.num_dim), dtype=np.float64)
        self.coupling_indices = cuda.to_device(
            np.vstack(np.where(coupling != 0.0)).T - (coupling.shape[0] // 2)
        )
        self.coupling_constants = cuda.to_device(
            coupling[np.where(coupling != 0.0)]
        )
        self.temperature = temperature
        self.field = field

    def draw_2D_spins(self, ax=None, **kwargs):
        """
        draws the spins of a 2D lattice

        :param ax:
        :param kwargs:
        :return:
        """
        if self.num_dim != 2:
            raise RuntimeError("Cannot draw non-2D configuration")
        if ax is None:
            ax = plt.gca()
        bool_spins = np.empty(self.shape, dtype=np.bool)
        for i in range(8):
            bool_spins[:, i::8] = (
                ((self.spins.copy_to_host() >> i) & 1)
                    .reshape(self.shape[0], self.shape[1] // 8)
            )
        ax.imshow(bool_spins, cmap="Greys")

    def calc_all_energies(self):
        energies = cuda.device_array((self.num_spins,), dtype=np.float64)
        energies[:] = 0
        ic.calc_energy[self.num_spins >> 3, 8](
            self.spins, self.shape_shifts, self.coupling_indices,
            self.coupling_constants, energies
        )
        return energies.copy_to_host()

    def random_flip(self, num):
        num = (num >> 3) << 3
        rng_states = ncrand.create_xoroshiro128p_states((num,), 1)
        ic.random_flip[num >> 3, 8](self.spins, self.shape_shifts, rng_states)

    def metropolis(self, iterations, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2 ** 64, dtype=np.uint64)
        offsets = np.zeros((1 << self.num_dim, self.num_dim), dtype=np.bool)
        for n in range(1 << self.num_dim):
            for a in range(self.num_dim):
                offsets[n, a] = (n >> a) & 1
        offsets = cuda.to_device(offsets)
        reduced_blocks = (
                np.ones(self.num_dim, dtype=np.int32) <<
                (
                            self.shape_shifts.copy_to_host() - self.block_shifts.copy_to_host() - 1)
        )
        total_blocks = np.prod(reduced_blocks)
        rng_states = ncrand.create_xoroshiro128p_states((total_blocks,), seed)
        for iteration in range(iterations):
            for i in range(1 << self.num_dim):
                ic.metropolis_step[total_blocks >> 6, 64](
                    self.spins, self.shape_shifts,
                    self.temperature, self.field,
                    self.coupling_indices, self.coupling_constants,
                    self.block_shifts, offsets[i], rng_states
                )


