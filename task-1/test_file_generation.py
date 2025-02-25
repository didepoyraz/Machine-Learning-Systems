# Modified test_file_creation.py
import numpy as np

# Dimension 2
N = 1000
D_small = 2
A_2d = np.random.randn(N, D_small).astype(np.float32)
X_2d = np.random.randn(D_small).astype(np.float32)
np.savetxt("A_2d.txt", A_2d)
np.savetxt("X_2d.txt", X_2d)

# Dimension 1024 or should this be 2^15
# D_large = 1024
# A_1024d = np.random.randn(N, D_large).astype(np.float32)
# X_1024d = np.random.randn(D_large).astype(np.float32)
# Use binary format for large dimensions
# np.save("A_1024d.npy", A_1024d)
# np.save("X_1024d.npy", X_1024d)

print("Data generation complete.")