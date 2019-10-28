import numbacudaising as nci
from matplotlib import pyplot as plt

ising = nci.IsingModel(
    (32, 32), (4, 4),
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

for i in range(16):
    ising.draw_2D_spins()
    plt.show()

    ising.metropolis(1)

