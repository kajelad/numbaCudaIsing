import isingModel as im

ising = im.IsingModel((16, 16), coupling=im.nearest_neighbor_kernel)

print("spins:")
print(ising.spins.copy_to_host())
print("coupling indices:")
print(ising.coupling_indices.copy_to_host())
print("coupling constants:")
print(ising.coupling_constants.copy_to_host())


