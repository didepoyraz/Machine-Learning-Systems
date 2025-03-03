import numpy as np
import json

def save_metadata(filename, n, d, a_file, x_file, k=10):
    """Saves metadata as a JSON file."""
    metadata = {
        "n": n,
        "d": d,
        "a_file": a_file,
        "x_file": x_file,
        "k": k
    }
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=4)

def generate_and_save(n, d, prefix):
    """Generates data and saves it in .npy format along with metadata."""
    A = np.random.randn(n, d).astype(np.float32)
    X = np.random.randn(d).astype(np.float32)

    a_file = f"{prefix}_A.npy"
    x_file = f"{prefix}_X.npy"
    meta_file = f"{prefix}_meta.json"

    np.save(a_file, A)
    np.save(x_file, X)
    save_metadata(meta_file, n, d, a_file, x_file)

# Generate and save datasets
generate_and_save(1000, 2, "2d")
generate_and_save(1000, 2**15, "215")
generate_and_save(4000, 100, "4k")
generate_and_save(4000000, 100, "4m")

print("Data generation complete!")
