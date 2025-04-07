import numpy as np
import json
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

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

def save_metadata_kmeans(filename, n, d, k, a_file):
    """Saves metadata as a JSON file."""
    metadata = {
        "n": n,
        "d": d,
        "a_file": a_file,
        "k": k
    }
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=4)

def generate_and_save_kmeans(n, d,k, prefix):
    """Generates data and saves it in .npy format along with metadata."""
    print(f"Generating data with k={k}")
    
    A, labels = make_blobs(n_samples=n, centers=k, n_features=d,random_state=42)
    A = A.astype(np.float32)
    plt.clf()
    print(f"Generated data shape: {A.shape}")
    print(f"Unique cluster labels: {np.unique(labels)}")    
    plt.scatter(A[:, 0], A[:, 1], s=2, alpha=0.5)
    plt.savefig(f"figures/TEST_{prefix}.png")
      
    a_file = f"{prefix}_A.npy"
    meta_file = f"{prefix}_meta.json"
    np.save(a_file, A)
    
    save_metadata_kmeans(meta_file, n, d, k, a_file)


def save_metadata_ann(filename, n, d, k, a_file, x_file):
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
        

def generate_and_save_ann(n, d,k1, prefix, k2=10):
    #this assumes that the wanted num of top vectors are 10 always
    """Generates data and saves it in .npy format along with metadata."""
    print(f"Generating data with k={k1}")
    
    A, labels = make_blobs(n_samples=n, centers=k1, n_features=d,random_state=42)
    A = A.astype(np.float32)
    X = np.random.randn(d).astype(np.float32)
    
    plt.clf()
    print(f"Generated data shape: {A.shape}")
    print(f"Unique cluster labels: {np.unique(labels)}")    
    plt.scatter(A[:, 0], A[:, 1], s=2, alpha=0.5)
    plt.savefig(f"figures/TEST_{prefix}.png")
      
    a_file = f"{prefix}_A.npy"
    x_file = f"{prefix}_X.npy"
    meta_file = f"{prefix}_meta.json"
    
    np.save(x_file, X)
    np.save(a_file, A)
    
    save_metadata_ann(meta_file, n, d, k2, a_file, x_file)





# generate_and_save_kmeans(1000,2,30, "1000_2")

# generate_and_save_kmeans(1000,1024,30, "1000_1024")

generate_and_save_kmeans(100000,2,7, "100k_2")

generate_and_save_kmeans(100000,1024,7, "100k_1024")



# generate_and_save_kmeans(1000000,100,4, "1m")

# generate_and_save_kmeans(1000000,10,10, "1m_10_K10")

# generate_and_save_kmeans(1000000,100,10, "1m_100_K10")

# generate_and_save_kmeans(4000000,100,10, "4m_K10")



# Generate and save datasets
# generate_and_save(1000, 2, "2d")
# generate_and_save(1000, 2**15, "215")
# generate_and_save(4000, 100, "4k")
# generate_and_save(40000, 100, "40k")
# generate_and_save(4000000, 100, "4m")

# generate_and_save_ann(1000, 2, 3, "2d")
# generate_and_save_ann(1000, 2**15, 3, "215")
# generate_and_save_ann(4000, 100, 3, "4k")
# generate_and_save_ann(40000, 100, 4, "40k")
# generate_and_save_ann(4000000, 100, 10, "4m")

print("Data generation complete!")
