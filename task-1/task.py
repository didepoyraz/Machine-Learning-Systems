import cupy as cp
import torch
import triton
import numpy as np
import random
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import matplotlib.pyplot as plt
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

def our_knn(N, D, A, X, K, dist_metric='manhattan'):
    
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

def our_knn_cpu(N, D, A, X, K, dist_metric="manhattan"):
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
    print(f"A: \n: {A}\n, K: {K}")
    
    # Pick K initial centroids
        # do this by picking K random elements from A for random centroid initialisation
    
    # now iterate through A and assign each vector to the nearest centroid
    # call distance function
    #--------------------------------------------------------------------------------------#
        # do this by first computing distances from every point to all K centroids 
        # call dist function K times
    #--------------------------------------------------------------------------------------#   
    # then assign each point to the closest centroid
        # iterate through A and check for each vector i  which distance array gave the smallest distance one of the centroids
    
    # now for each cluster, compute the mean distance of all assigned vectors to find new centroids
    # if the centroids have changed in the iteration repeat until convergence (until they don't change anymore)
        #reassign points, recompute centroids and repeat until results are stable
    
    dist_metric = "l2"
    
    max_iterations = 100 #decide
    centroid_shift_tolerance = 1e-4 # decide
    converged = False
    
    #Initialise Centroids, by selecting K random vectors from A
    A_tensor = torch.from_numpy(A).cuda(non_blocking=True)  
    indices = torch.randint(0, N, (K,), device="cuda") # select K random indices from the indices of A
    init_centroids_d = A_tensor[indices]  #filter the chosen K random vectors directly on GPU //question whether this is on the gpu
    print(f"init centroids: {init_centroids_d}")
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    
    distances =  torch.empty(K, device="cuda")
    
    # an empty matrix of shape (K, N) on the GPU, initialized with -1 (to represent empty slots)
    cluster_labels = torch.full((K,N), -1, dtype=torch.int32, device="cuda")
    cluster_distances = torch.full((K, N), float('inf'), dtype=torch.float32, device="cuda")  # stores min distances with infinity to differentiate if there is a distance written

    # to track how many vectors have been assigned per centroid
    centroid_counts = torch.zeros(K, dtype=torch.int32, device="cuda")
    
    # batch_num = None
    # batches, batch_size = divide_batches(batch_num, A, N)
    
    # Find the best batch size according to the available memory in the GPU
    # MAX_FRACTION = 0.8
    # device = torch.cuda.current_device()
    # total_memory = torch.cuda.get_device_properties(device).total_memory
    # allocated_memory = torch.cuda.memory_allocated(device)
    # available_memory = total_memory - allocated_memory
    # usable_memory = available_memory * MAX_FRACTION
    
    # bytes_per_vec_element = 8
    # bytes_per_vec = D * bytes_per_vec_element
    # batch_size = int(usable_memory // bytes_per_vec)

    # num_batches = (N + batch_size - 1) // batch_size
    # move A to the GPU in batches
    distances = torch.empty(N, device="cuda")
    # A_gpu_batches = [] #does not need to be on the GPU
    
    # for i in range(num_batches):
    #     start_idx = i * batch_size
    #     end_idx = min((i + 1) * batch_size, N)
        
    #     A_batch = torch.from_numpy(A[start_idx : end_idx]).cuda(non_blocking=True)
    #     A_gpu_batches.append(A_batch)
   
    #---------------------------------------------------------------------------------------#
    # A = (N,D), init_centroids_d = (K,D) ,distances = (N,K), 
    print(f"A shape: {A_tensor.shape}, init centroids shape: {init_centroids_d.shape}")
    iteration = 0
    while not converged and iteration < max_iterations:
        print(f"iteration {iteration}")
        iteration += 1
        if dist_metric == "l2":
            distances = torch.sum((A_tensor[:,None] - init_centroids_d)**2, dim=2)
        elif dist_metric == "cosine":
            A_norm = torch.nn.functional.normalize(A_tensor, p=2, dim=1)
            C_norm = torch.nn.functional.normalize(init_centroids_d, p=2, dim=0)
            similarities = torch.matmul(A_norm, C_norm.T) #take transpose of centroids to make it (D,K) so matmul can give (N,K)
            distances = 1 - similarities
        elif dist_metric == "dot":
            distances = -torch.matmul(A_tensor, init_centroids_d.T)  # Negate so smaller is better, take transpose of centroids to make it (D,K) so matmul can give (N,K)
        elif dist_metric == "manhattan":
            distances = torch.sum(torch.abs(A_tensor[:,None] - init_centroids_d), dim=2) # expand A to (N,1,D) so that broadcasting can be done and move dim to 2 so it collapses for dimension D still and get (N,K)
        else:
            raise ValueError("Invalid distance metric")
        
        print(f"shape of distances (should be ({N},{K})): {distances.shape}")
        cluster_labels = torch.argmin(distances, dim=1)  # (N,), finds the minimum column of each row, each row corresponds to one vector of A and the columns correspond to the distance to each centroid from that vector
        
        new_centroids = torch.zeros((K, D), device=device)
        counts = torch.bincount(cluster_labels, minlength=K).float().unsqueeze(1)
        new_centroids.scatter_add_(0, cluster_labels[:, None].expand(-1, D), A_tensor)
        counts[counts == 0] = 1  # Avoid division by zero
        new_centroids /= counts
        print(f"count: {counts}\n, new centroids shape: {new_centroids.shape}, and itself: \n {new_centroids}")
               
        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()
        
        if torch.all(centroid_shift <=centroid_shift_tolerance):
            converged = True
            
    new_centroids_cpu = new_centroids.cpu().numpy()
    plt.scatter(A[:, 0], A[:, 1], c=cluster_labels.cpu().numpy(), cmap="viridis", alpha=0.5)
    plt.scatter(new_centroids_cpu[:, 0], new_centroids_cpu[:, 1], c='red', marker='x', s=200, label="Centroids")
    plt.title("K-Means Clustering Results")
    plt.legend()
    
    save_path ="kmeans_plot.png"
    # Save plot instead of showing
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
        
    return cluster_labels.cpu().numpy(), new_centroids.cpu().numpy() # decide on the return value based on what is needed for 2.2

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

def test_kmeans_10():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("10_meta.json")

    # # Measure CPU time
    # _, cpu_time = measure_time(our_knn_cpu, N, D, A, X, K)
    # print(f"CPU Time (4m): {cpu_time:.6f} seconds")

    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K)
    print(f"GPU Time (10): {gpu_time:.6f} seconds")

    # Calculate Speedup
    # speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    # print(f"Speedup 4m (CPU to GPU): {speedup:.2f}x\n")

if __name__ == "__main__":
    # test_knn()
    # test_knn_cpu()
    # test_knn_2D()
    # test_knn_215()
    # test_knn_4k()
    # #test_knn_40k()
    # test_knn_4m()
    
    test_kmeans_10()
  