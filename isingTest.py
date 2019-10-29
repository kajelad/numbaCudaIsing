import numpy as np
import numbacudaising as nci
from matplotlib import pyplot as plt

coupling = np.ones((15, 15), dtype=np.float64)

coupling[8, 8] = 0

ising = nci.IsingModel(
    (2**9, 2**9), (8, 8),
    coupling=nci.nearest_neighbor_kernel(2), temperature = 0.5
)

print("spins:")
print(ising.spins.copy_to_host())
print("coupling indices:")
print(ising.coupling_indices.copy_to_host())
print("coupling constants:")
print(ising.coupling_constants.copy_to_host())
print("shape_shifts")
print(ising.shape_shifts)
print("block_shifts")
print(ising.block_shifts)

initial_spins = ising.spins.copy_to_host()

#for i in range(16):
    # ising.draw_2D_spins()
    # plt.show()

    #energies = ising.calc_all_energies()
    #ising.metropolis(1024)

rand_indices = np.arange(64)
for i in range(1, 21):
    print(i)
    ising.metropolis(2**i)
    print(np.array([initial_spins[ri] ^ ising.spins[ri] for ri in rand_indices]))
    print(np.array([initial_spins[ri] ^ ising.spins[ri] for ri in rand_indices+2**14]))
