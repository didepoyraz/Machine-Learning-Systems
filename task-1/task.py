import torch
import cupy as cp
#import triton
from sklearn.cluster import KMeans, MiniBatchKMeans
from cuml.cluster import KMeans as cuKMeans
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann, read_data

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

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K, dist_metric='dot'):
    # Move X to GPU (A is moved in batches below)
    X_tensor = torch.from_numpy(X).cuda()

    # Automatically determine batch size based on N
    if N <= 500:
        batch_size = 64
    elif N <= 1000:
        batch_size = 128
    elif N <= 10_000:
        batch_size = 1024
    elif N <= 100_000:
        batch_size = 4096
    else:
        batch_size = 8192  # Large batch for very large datasets

    num_batches = (N + batch_size - 1) // batch_size  # Compute number of batches
    distances_list = []
    indices_list = []

    # Create a small pool of CUDA streams (e.g., 4 streams)
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # Process the batches in a loop
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)

        # Move only the current batch of A to GPU
        A_batch = torch.from_numpy(A[start_idx:end_idx]).cuda()

        # Use a stream from the pool to overlap computation
        stream = streams[i % num_streams]  # Reuse streams in a round-robin manner

        with torch.cuda.stream(stream):  # Use the selected stream
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

            # Get top-K indices in this batch
            batch_dists, batch_indices = torch.topk(dists, k=K, largest=False)

            # Store results (move to CPU asynchronously)
            distances_list.append(batch_dists.cpu())
            indices_list.append((batch_indices + start_idx).cpu())  # Adjust indices

    # Synchronize all CUDA streams
    torch.cuda.synchronize()

    # Concatenate all batches
    all_distances = torch.cat(distances_list)
    all_indices = torch.cat(indices_list)

    # Sort overall K-nearest from all batches
    sorted_indices = torch.argsort(all_distances)[:K]

    sorted_indices = all_indices[sorted_indices].cpu().numpy()
    print("KNN - Top K nearest neighbors indices:", sorted_indices)
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

def our_knn_cpu(N, D, A, X, K, dist_metric="dot"):
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


def euclidean_distance(vec1, vec2):
    return torch.sqrt(torch.sum((vec1 - vec2) ** 2, dim=-1))

def negative_dot_distance(vec1, vec2):
    return -torch.sum(vec1 * vec2, dim=-1)

def ann_search(A, cluster_centers, cluster_assignments, X, K1, K2):
    """
    Perform Approximate Nearest Neighbor (ANN) search using K-Means clustering.

    Parameters:
        A (torch.Tensor): (N, D) Tensor containing dataset vectors.
        cluster_centers (torch.Tensor): (K_clusters, D) Tensor of cluster centers.
        cluster_assignments (torch.Tensor): (N,) Tensor mapping each point to its cluster.
        X (torch.Tensor): (D,) Query vector.
        K1 (int): Number of nearest clusters to consider.
        K2 (int): Number of nearest neighbors to return.

    Returns:
        torch.Tensor: (K2,) Tensor containing the indices of the top K2 nearest neighbors.
    """

    cluster_distances = negative_dot_distance(X, cluster_centers)
    nearest_clusters = torch.argsort(cluster_distances)[:K1] 

    candidate_indices = torch.cat([
        torch.where(cluster_assignments == cluster_idx)[0] for cluster_idx in nearest_clusters
    ])

    candidate_points = A[candidate_indices]

    candidate_distances = negative_dot_distance(candidate_points, X)
    nearest_neighbors = candidate_indices[torch.argsort(candidate_distances)[:K2]]

    return nearest_neighbors

def our_ann(N, D, A, X, K):
    device = "cuda"
    K_clusters = 5  # Number of clusters for K-Means
    K1 = 3  # Number of clusters and neighbors to consider
    starttime = time.time()
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float16)
        print("Time taken by if isinstance1 = " + str(time.time() - starttime))

    starttime = time.time()
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float16)
        print("Time taken by if isinstance2 = " + str(time.time() - starttime))

    starttime = time.time()
    # Move data to GPU (if available)
    A = A.to(device)
    X = X.to(device)
    print("Time taken to device = " + str(time.time() - starttime))

    # K-Means clustering
    if N < 10000:
            starttime = time.time()
            kmeans = KMeans(n_clusters=K_clusters, max_iter=10, n_init=5)
            print("Time taken by kmeans = " + str(time.time() - starttime))
    else:
        starttime = time.time()
        kmeans = MiniBatchKMeans(n_clusters=K_clusters, batch_size=10000, max_iter=10, n_init=5)
        print("Time taken by kmeans = " + str(time.time() - starttime))
    
    starttime = time.time()
    cluster_assignments = torch.tensor(kmeans.fit_predict(A.cpu()), device=device)
    print("Time taken by cluster assignemnt = " + str(time.time() - starttime))
    starttime = time.time()
    cluster_centers = torch.tensor(kmeans.cluster_centers_, device=device)
    print("Time taken by cluster center assignemnt = " + str(time.time() - starttime))

    # ANN search
    starttime = time.time()
    top_k_neighbors = ann_search(A, cluster_centers, cluster_assignments, X, K1, K)
    print("Time taken by ANN search = " + str(time.time() - starttime))
    # move from GPU to CPU
    print("ANN - Top K nearest neighbors indices:", top_k_neighbors.cpu().numpy())

    return top_k_neighbors

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
    N, D, A, X, K = testdata_ann("")
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
    print(f"Speedup (CPU to GPU): {speedup:.2f}x")


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
    print(f"Speedup (CPU to GPU): {speedup:.2f}x")

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
    print(f"Speedup (CPU to GPU): {speedup:.2f}x")

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
    print(f"Speedup (CPU to GPU): {speedup:.2f}x")

def test_ann_2D():
    print("\n\n------------------------\n\n")
    print("For 2D: ")
    N, D, A, X, K = testdata_knn("2d_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

    N, D, A, X, K = testdata_ann("2d_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    ann_result, gpu_time = measure_time(our_ann, N, D, A, X, K)
    print(f"ANN Time: {gpu_time:.6f} seconds")

    list1 = {int(x) for x in knn_result}
    list2 = {int(x.cpu()) for x in ann_result}

    rr = recall_rate(list1, list2)
    print("Recall Rate: " + str(rr))

def test_ann_215():
    print("\n\n------------------------\n\n")
    print("For 215: ")
    # Load test data from JSON
    N, D, A, X, K = testdata_knn("215_meta.json")

    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

    N, D, A, X, K = testdata_ann("215_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    ann_result, gpu_time = measure_time(our_ann, N, D, A, X, K)
    print(f"ANN: {gpu_time:.6f} seconds")

    list1 = {int(x) for x in knn_result}
    list2 = {int(x.cpu()) for x in ann_result}

    rr = recall_rate(list1, list2)
    print("Recall Rate: " + str(rr))

def test_ann_4k():
    print("\n\n------------------------\n\n")
    print("For 4k ")
    N, D, A, X, K = testdata_knn("4k_meta.json")

    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

    N, D, A, X, K = testdata_ann("4k_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    ann_result, gpu_time = measure_time(our_ann, N, D, A, X, K)
    print(f"ANN: {gpu_time:.6f} seconds")

    list1 = {int(x) for x in knn_result}
    list2 = {int(x.cpu()) for x in ann_result}

    rr = recall_rate(list1, list2)
    print("Recall Rate: " + str(rr))

def test_ann_40k():
    print("\n\n------------------------\n\n")
    print("For 40k ")
    N, D, A, X, K = testdata_knn("40k_meta.json")

    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

    N, D, A, X, K = testdata_ann("40k_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    ann_result, gpu_time = measure_time(our_ann, N, D, A, X, K)
    print(f"ANN: {gpu_time:.6f} seconds")

    list1 = {int(x) for x in knn_result}
    list2 = {int(x.cpu()) for x in ann_result}

    rr = recall_rate(list1, list2)
    print("Recall Rate: " + str(rr))

def test_ann_4m():
    print("\n\n------------------------\n\n")
    print("For 4m ")

    # Load test data from JSON
    N, D, A, X, K = testdata_knn("4m_meta.json")

    # Measure GPU time
    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

    N, D, A, X, K = testdata_ann("4m_meta.json")

    # Manually check file types for A and X
    if isinstance(A, str):  # If A is a file path (for .txt or .npy)
        A = read_data(A)
    if isinstance(X, str):  # If X is a file path (for .txt or .npy)
        X = read_data(X)

    ann_result, gpu_time = measure_time(our_ann, N, D, A, X, K)
    print(f"ANN: {gpu_time:.6f} seconds")

    list1 = {int(x) for x in knn_result}
    list2 = {int(x.cpu()) for x in ann_result}

    rr = recall_rate(list1, list2)
    print("Recall Rate: " + str(rr))


if __name__ == "__main__":
    #test_knn()
    #test_knn_cpu()
    #test_knn_2D()
    #test_knn_215()
    #test_knn_4k()
    #test_knn_4m()
    test_ann()
    test_ann_2D()
    test_ann_215()
    test_ann_4k()
    test_ann_40k()
    test_ann_4m()