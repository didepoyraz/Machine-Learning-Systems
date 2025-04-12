import numpy as np
import json
from sklearn.datasets import make_blobs
import argparse

# add arguments
parser = argparse.ArgumentParser(description="KNN implementation with GPU and CPU")
parser.add_argument("--test", choices=["dist", "knn", "kmeans", "ann"], default="knn",
                    help="Choose test type (default: knn)")
args = parser.parse_args()

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

def generate_and_save_knn(n, d, prefix):
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
    #plt.clf()
    print(f"Generated data shape: {A.shape}")
    print(f"Unique cluster labels: {np.unique(labels)}")    
    #plt.scatter(A[:, 0], A[:, 1], s=2, alpha=0.5)
    #plt.savefig(f"figures/TEST_{prefix}.png")
      
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
    
    #plt.clf()
    print(f"Generated data shape: {A.shape}")
    print(f"Unique cluster labels: {np.unique(labels)}")    
    #plt.scatter(A[:, 0], A[:, 1], s=2, alpha=0.5)
    #plt.savefig(f"figures/TEST_{prefix}.png")
      
    a_file = f"{prefix}_A.npy"
    x_file = f"{prefix}_X.npy"
    meta_file = f"{prefix}_meta.json"
    
    np.save(x_file, X)
    np.save(a_file, A)
    
    save_metadata_ann(meta_file, n, d, k2, a_file, x_file)


if args.test == "knn":
    # Generate and save datasets
    generate_and_save_knn(1000, 2, "2d")
    generate_and_save_knn(1000, 2**15, "215")
    generate_and_save_knn(4000, 100, "4k")
    generate_and_save_knn(40000, 100, "40k")
    generate_and_save_knn(4000000, 100, "4m")

elif args.test == "kmeans":
    generate_and_save_kmeans(1000,10,3, "1000_2")
    generate_and_save_kmeans(1000,1024,3, "1000_1024")
    generate_and_save_kmeans(10000,10,3, "10k_2")
    generate_and_save_kmeans(10000,1024,3, "10k_1024")
    generate_and_save_kmeans(1000000,10,10, "1m_10_K10")
elif args.test == 'ann':
    generate_and_save_ann(1000, 2, 3, "2d")
    generate_and_save_ann(1000, 2**15, 5, "215")
    generate_and_save_ann(100000, 1024, "100k")
    generate_and_save_ann(4000, 100, 3, "4k")
    generate_and_save_ann(40000, 100, 5, "40k")
    generate_and_save_ann(4000000, 100, 200, "4m")

print("Data generation complete!")
