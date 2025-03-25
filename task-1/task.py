import cupy as cp
import torch
import triton
import numpy as np
import time
import json
from test import read_data, testdata_kmeans, testdata_knn, testdata_ann
import argparse

# add arguments
parser = argparse.ArgumentParser(description="KNN implementation with GPU and CPU")
parser.add_argument("--distance", choices=["cosine", "l2", "dot", "manhattan"], default="manhattan",
                    help="Choose distance metric (default: manhattan)")
parser.add_argument("--test", choices=["knn", "kmeans", "ann"], default="knn",
                    help="Choose test type (default: knn)")
args = parser.parse_args()

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    # Compute cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(X, Y, dim=0)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    # Compute L2 distance (Euclidean)
    return torch.norm(X - Y)

def distance_dot(X, Y):
    # Compute dot product
    return torch.dot(X, Y)

def distance_manhattan(X, Y):
    # Compute Manhattan distance (L1 norm)
    return torch.sum(torch.abs(X - Y))

def get_distance_function():
    return {
        "cosine": distance_cosine,
        "l2": distance_l2,
        "dot": distance_dot,
        "manhattan": distance_manhattan
    }[args.distance]

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------
# Global variable (set by argparse)
dist_metric = None  # Placeholder for the distance metric to be set dynamically

def set_distance_metric(metric):
    global dist_metric
    dist_metric = metric

def our_knn(N, D, A, X, K):
    if dist_metric is None:
        raise ValueError("Distance metric not set. Please specify one via command-line arguments.")
    
    X_tensor = torch.from_numpy(X).cuda()
    
    if N <= 100000:
        A_tensor = torch.from_numpy(A).cuda(non_blocking=True)  
    
        if dist_metric == "l2":
            dists = torch.norm(A_tensor - X_tensor, dim=1)
        elif dist_metric == "cosine":
            A_norm = torch.nn.functional.normalize(A_tensor, p=2, dim=1)
            X_norm = torch.nn.functional.normalize(X_tensor, p=2, dim=0)
            similarities = torch.matmul(A_norm, X_norm)
            dists = 1 - similarities
        elif dist_metric == "dot":
            dists = -torch.matmul(A_tensor, X_tensor)  # Negate so smaller is better
        elif dist_metric == "manhattan":
            dists = torch.sum(torch.abs(A_tensor - X_tensor), dim=1)
        else:
            raise ValueError("Invalid distance metric")
     
        # Get top-K indices in this batch
        _, sorted_indices = torch.topk(dists, k=K, largest=False)
        sorted_indices = sorted_indices.cpu().numpy()
        return sorted_indices

    # Find the best batch size according to the available memory in the GPU
    MAX_FRACTION = 0.8
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory
    usable_memory = available_memory * MAX_FRACTION
    
    bytes_per_vec_element = 8
    bytes_per_vec = D * bytes_per_vec_element
    batch_size = int(usable_memory // bytes_per_vec)
    
    distances_list = []
    indices_list = []

    num_batches = (N + batch_size - 1) // batch_size
    
    stream1 = torch.cuda.Stream()  # For memory transfer
    stream2 = torch.cuda.Stream()  # For computation
            
    A_batch = torch.from_numpy(A[:batch_size]).cuda(non_blocking=True)  # pretransfer the first block 

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        with torch.cuda.stream(stream1):
            # Transfer next batch to the GPU
            if i < num_batches - 1: 
                A_batch = torch.from_numpy(A[end_idx : end_idx + batch_size]).cuda(non_blocking=True)

        with torch.cuda.stream(stream2):
            stream2.wait_stream(stream1)  # Ensure batch is available before computing
        
            # Compute distances
            if dist_metric == "l2":
                dists = torch.norm(A_batch - X_tensor, dim=1)
            elif dist_metric == "cosine":
                A_norm = torch.nn.functional.normalize(A_batch, p=2, dim=1)
                X_norm = torch.nn.functional.normalize(X_tensor, p=2, dim=0)
                similarities = torch.matmul(A_norm, X_norm)
                dists = 1 - similarities
            elif dist_metric == "dot":
                dists = -torch.matmul(A_batch, X_tensor)  # Negate so smaller is better
            elif dist_metric == "manhattan":
                dists = torch.sum(torch.abs(A_batch - X_tensor), dim=1)
            else:
                raise ValueError("Invalid distance metric")

            # Get top-K indices
            batch_dists, batch_indices = torch.topk(dists, k=K, largest=False)

            distances_list.append(batch_dists.cpu())
            indices_list.append((batch_indices + start_idx).cpu())
    
    torch.cuda.synchronize()  

    all_distances = torch.cat(distances_list)
    all_indices = torch.cat(indices_list)
    sorted_indices = all_indices[torch.argsort(all_distances)[:K]].cpu().numpy()

    return sorted_indices


# CPU top k
# based on same format as GPU version
# batching not necessary but can be added if needed
# using numpy

def distance_cosine_cpu(X, Y):
    dot_product = np.dot(X, Y)
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    return 1 - (dot_product / (norm_X * norm_Y))

def distance_l2_cpu(X, Y):
    return np.linalg.norm(X - Y)

def distance_dot_cpu(X, Y):
    return np.dot(X, Y)

def distance_manhattan_cpu(X, Y):
    return np.sum(np.abs(X - Y))

def our_knn_cpu(N, D, A, X, K):
    if A.shape != (N, D) or X.shape != (D,):
        raise ValueError("Input dimensions do not match.")

    distance_func = {
        "cosine": distance_cosine_cpu,
        "l2": distance_l2_cpu,
        "dot": distance_dot_cpu,
        "manhattan": distance_manhattan_cpu
    }.get(dist_metric)

    if distance_func is None:
        raise ValueError("Invalid distance metric. Choose from 'cosine', 'l2', 'dot', 'manhattan'.")

    # Compute distances and find top K efficiently
    distances = np.array([distance_func(X, Y) for Y in A])

    # Ensure the distances are sorted
    sorted_indices = np.argpartition(distances, K)[:K] # sort and get top K
    return sorted_indices

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)

def test_knn_cpu():
    N, D, A, X, K = testdata_knn("")
    knn_result = our_knn_cpu(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

# incorporate timing into testing
def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure that GPU computation has finished
    end = time.perf_counter()
    return result, end - start

def test_knn_2D():
    # Load test data from JSON (using testdata_knn)
    N, D, A, X, K = testdata_knn("2d_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    # Measure CPU time
    _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    print(f"CPU Time (2D): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"GPU Time (2D): {gpu_time:.6f} seconds")

    # Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup (CPU to GPU): {speedup:.2f}x\n")


def test_knn_215():
    # Load test data from JSON
    N, D, A, X, K = testdata_knn("215_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    print(f"CPU Time (2^15): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"GPU Time (2^15): {gpu_time:.6f} seconds")

    # Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 2*15 (CPU to GPU): {speedup:.2f}x\n")

def test_knn_4k():
    # Load test data from JSON
    N, D, A, X, K = testdata_knn("4k_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    print(f"CPU Time (4k): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"GPU Time (4k): {gpu_time:.6f} seconds")

    # Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 4k (CPU to GPU): {speedup:.2f}x\n")
    
def test_knn_40k():
    # Load test data from JSON
    N, D, A, X, K = testdata_knn("40k_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    print(f"CPU Time (40k): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"GPU Time (40k): {gpu_time:.6f} seconds")

    # Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 40k (CPU to GPU): {speedup:.2f}x\n")

def test_knn_4m():
    # Load test data from JSON
    N, D, A, X, K = testdata_knn("4m_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    print(f"CPU Time (4m): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"GPU Time (4m): {gpu_time:.6f} seconds")

    # Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 4m (CPU to GPU): {speedup:.2f}x\n")

if __name__ == "__main__":
    # Set the distance metric from command line argument
    set_distance_metric(args.distance)
    
    # Conditional test execution based on the --test argument
    if args.test == "knn":
        test_knn()
        test_knn_cpu()
        test_knn_2D()
        test_knn_215()
        test_knn_4k()
        test_knn_4m()
    elif args.test == "kmeans":
        test_kmeans()
    elif args.test == "ann":
        test_ann()