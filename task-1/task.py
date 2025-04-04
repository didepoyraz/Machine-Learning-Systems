import cupy as cp
import torch
import triton
import numpy as np
import heapq
import time
import json
from test import read_data, testdata_kmeans, testdata_knn, testdata_ann
from sklearn.cluster import KMeans, MiniBatchKMeans
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

# add arguments
parser = argparse.ArgumentParser(description="KNN implementation with GPU and CPU")
parser.add_argument("--distance", choices=["cosine", "l2", "dot", "manhattan"], default="cosine",
                    help="Choose distance metric (default: manhattan)")
parser.add_argument("--test", choices=["dist", "knn", "kmeans", "ann"], default="knn",
                    help="Choose test type (default: knn)")
args = parser.parse_args()

dist_metric = args.distance

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
            raise ValueError(f"Invalid distance metric: {args.distance}")
     
        # Get top-K indices in this batch
        # Manually sort the distances and get the indices of top-K
        sorted_indices = torch.argsort(dists)[:K]  # Get the indices of the top K smallest distances
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
    min_heap = []

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

        # Manually sort the distances and get the indices of top-K
        batch_sorted_indices = torch.argsort(dists)[:K]  # Get the indices of the top K smallest distances
        # Add the batch's top K to the heap, maintaining only the global top K
        for j in range(K):
            heapq.heappush(min_heap, (dists[batch_sorted_indices[j]].item(), start_idx + batch_sorted_indices[j].item()))
            if len(min_heap) > K:
                heapq.heappop(min_heap)  # Keep only the top-K smallest distances
    
    # Extract the top-K global results from the heap
    sorted_indices = [idx for _, idx in sorted(min_heap, key=lambda x: x[0])]
    return np.array(sorted_indices)  # Convert to numpy for the final output


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

NUM_INIT = None 

def our_kmeans(N, D, A, K):
    global dist_metric
    if dist_metric not in ["l2", "cosine"]:
        print(f"Warning: K-means only supports l2 and cosine distances. Using l2 instead of {dist_metric}.")
        dist_metric = "l2"  # Set a fallback metric
    
    max_iterations = 300 #decide
    centroid_shift_tolerance = 1e-5 # decide
    converged = False
    
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    # new_centroids = torch.zeros((K, D), device="cuda")
    
    # an empty matrix of shape (K, N) on the GPU, initialized with -1 (to represent empty slots)
    cluster_labels_batches = []
    cluster_labels = torch.full((K,N), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(K, dtype=torch.float32, device="cuda")
    distances = torch.empty(N, device="cuda")
    
    #------------------------------------------------------------------------#
    # Find the best batch size according to the available memory in the GPU and transfer A in batches
    MAX_FRACTION = 0.8
    MAX_BATCH_SIZE = 100_000
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory
    usable_memory = available_memory * MAX_FRACTION
    
    bytes_per_vec_element = 4
    bytes_per_vec = D * bytes_per_vec_element
    
    batch_size = int(usable_memory // bytes_per_vec) # num of vectors per batch
    batch_size = min(batch_size, MAX_BATCH_SIZE)  
    
    num_batches = (N + batch_size - 1) // batch_size
    # print("batch size for 1m: ", batch_size, "usable memory: ", usable_memory, "num batches: ", num_batches)
    
    A_gpu_batches = [] #does not need to be on the GPU
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        A_batch = torch.from_numpy(A[start_idx : end_idx]).cuda(non_blocking=True)
        A_gpu_batches.append(A_batch)
    #--------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------#
    # this was used for transferring A to the GPU and choosing indices, without any batching
    # A_tensor = torch.from_numpy(A).to(dtype=torch.float32, device="cuda", non_blocking=True)
    # indices = torch.randint(0, N, (K,), device="cuda") # select K random indices from the indices of A
    # init_centroids_d = A_tensor[indices]  #filter the chosen K random vectors directly on GPU //question whether this is on the gpu
    #--------------------------------------------------------------------------------------#
    
    # initial_indices = np.random.choice(N, K, replace=False)
    # init_centroids_d = torch.tensor(A[initial_indices], device="cuda", dtype=torch.float32)

    #-----------------------------------------------------------------------------#
    # Use this initialisation of random centroids if you would like to compare sklearns with controlled init conditions in print_kmeans()
    np.random.seed(2)
    initial_indices = np.random.choice(N, K, replace=False)
    init_centroids_d = torch.tensor(A[initial_indices], device="cuda", dtype=torch.float32)
    #-----------------------------------------------------------------------------#
    
    stream1 = torch.cuda.Stream()  # For distance calculation
    stream2 = torch.cuda.Stream()  # For centroid labels and counts calculation
    
    iteration = 0
    while not converged and iteration < max_iterations:
        iteration += 1
        
        # assign clusters to all vectors
        # start_idx = 0
        for i, batch in enumerate(A_gpu_batches):
            # start_idx = i * batch_size
            # end_idx = min((i + 1) * batch_size, N)
            new_centroids.zero_()
            counts.zero_()
            cluster_labels_batches.clear()
            
            #---- stream1
            with torch.cuda.stream(stream1):
                if dist_metric == "l2":
                    distances = torch.sum((batch[:,None] - init_centroids_d)**2, dim=2)
                    # distances = torch.cdist(A_tensor, init_centroids_d, p=2) ** 2
                elif dist_metric == "cosine":
                    A_norm = torch.nn.functional.normalize(batch, p=2, dim=1)
                    C_norm = torch.nn.functional.normalize(init_centroids_d, p=2, dim=0)
                    similarities = torch.matmul(A_norm, C_norm.T) #take transpose of centroids to make it (D,K) so matmul can give (N,K)
                    distances = 1 - similarities
                else:
                    raise ValueError("Invalid distance metric")
              
            #---- stream2
            with torch.cuda.stream(stream2):
                stream2.wait_stream(stream1)  # Ensure batch is available before computing
                
                batch_cluster_labels = torch.argmin(distances, dim=1)
                new_centroids.scatter_add_(0, batch_cluster_labels[:, None].expand(-1, D), batch.to(torch.float32)) # cumulatively adds all vectors belonging to the same cluster
                counts.scatter_add_(0, batch_cluster_labels, torch.ones_like(batch_cluster_labels, dtype=torch.float32)) # for each cluster index in batch_labels, it adds 1.0 to counts at that position.
                
                cluster_labels_batches.append(batch_cluster_labels) # append the batches of cluster labels to concatenate at the end 
                
        torch.cuda.synchronize()  
        
        cluster_labels = torch.cat(cluster_labels_batches, dim=0)
        counts[counts == 0] = 1  # avoid division by zero
        new_centroids /= counts.unsqueeze(1)

        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()

        # if torch.mean(centroid_shift) <= centroid_shift_tolerance:
        if torch.max(centroid_shift) <= centroid_shift_tolerance:
            converged = True
        
        # gpu_ssd = ((A - new_centroids[cluster_labels]) ** 2).sum().item()
        # print("gpu ssd: ", gpu_ssd)
        print_kmeans(A, N, K, new_centroids, cluster_labels, initial_indices)

    return cluster_labels.cpu().numpy(), new_centroids.cpu().numpy() # decide on the return value based on what is needed for 2.2


def print_kmeans(A, N, K, new_centroids, cluster_labels, initial_indices):
    plot = "1m_10_k10"
    init_centroids = A[initial_indices]  # Use same initialization
    
    # change init to "kmeans++" if you would like to see better init conditions for improved cluster alignment
    sklearn_kmeans = KMeans(n_clusters=K, init=init_centroids, n_init=1, max_iter=500, random_state=2)
    sklearn_kmeans.fit(A)
    
    colors = ListedColormap(["blue", "green", "yellow"])
    plt.clf()
    # Scatter plot for clustered points (with smaller size and transparency)
    plt.scatter(A[:, 0], A[:, 1], c=sklearn_kmeans.labels_, cmap=colors, alpha=0.2, s=2)

    plt.title("K-Means Clustering Results SKLEARN")
    plt.legend()
    timestamp = int(time.time())  # Generate a unique filename
    save_path =f"figures/kmeans_sklearns/kmeans_plot_sklearns_{plot}_{timestamp}.png"
    plt.savefig(save_path)
    print(f"sklearns Plot saved to kmeans_sklearns_plot{plot}.png")  
    
    plt.clf()
    plt.scatter(A[:, 0], A[:, 1], c=cluster_labels.cpu().numpy(), cmap=colors, alpha=0.2, s=2)
    plt.title("K-Means Clustering Results GPU")
    plt.legend()
    
    timestamp = int(time.time())  # Generate a unique filename
    save_path =f"figures/kmeans_gpu/kmeans_plot{plot}_{timestamp}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    gpu_ssd = ((A - new_centroids[cluster_labels]) ** 2).sum().item()
    print(f"GPU K-Means SSD: {gpu_ssd}")

    sklearn_ssd = sklearn_kmeans.inertia_
    print(f"Sklearn K-Means SSD: {sklearn_ssd}")

    # Compare
    ssd_diff = abs(gpu_ssd - sklearn_ssd)
    print(f"SSD Difference: {ssd_diff:.4f}")
    torch_centroids = new_centroids.cpu().numpy()
    sklearn_centroids = sklearn_kmeans.cluster_centers_

    print("K-Means Centroids:\n", torch_centroids)
    print("Sklearn K-Means Centroids:\n", sklearn_centroids)
    
    centroid_error = np.linalg.norm(torch_centroids - sklearn_centroids, axis=1).mean()
    print(f"Average centroid difference: {centroid_error:.4f}")

    if np.array_equal(cluster_labels.cpu().numpy(), sklearn_kmeans.labels_):
        print("All cluster labels match exactly!")
    else:
        print("Some cluster assignments differ, skleanrs: ", sklearn_kmeans.labels_, "gpu: ", cluster_labels.cpu().numpy())

def our_kmeans_cpu(N, D, A, K):
    """
    NumPy-only K-Means implementation (CPU only).
    A: (N, D) NumPy array of data points
    K: Number of clusters
    """
    max_iterations=300
    tol=1e-5
    # Randomly initialize K centroids from data points
    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices]
    
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iterations:
        # print(f"iteration {iteration}")
        iteration += 1

        distances = np.linalg.norm(A[:, None, :] - centroids[None, :, :], axis=2)  # Shape (N, K)
        cluster_labels = np.argmin(distances, axis=1)  # Shape (N,)

        new_centroids = np.zeros((K, D))

        for k in range(K):
            assigned_points = A[cluster_labels == k]
            if len(assigned_points) > 0:
                new_centroids[k] = assigned_points.mean(axis=0)

        centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        
        if centroid_shift < tol:
            converged =True
       
    return cluster_labels, centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

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

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float16, device="cuda")

    if dist_metric == "l2":
        cluster_distances = euclidean_distance(X, cluster_centers)
    elif dist_metric == "cosine":
        # For cosine, normalize vectors first
        X_norm = torch.nn.functional.normalize(X.unsqueeze(0), p=2, dim=1).squeeze(0)
        centers_norm = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)
        cluster_distances = negative_dot_distance(X_norm, centers_norm)

    nearest_clusters = torch.argsort(cluster_distances)[:K1] 

    candidate_indices = torch.cat([
        torch.where(cluster_assignments == cluster_idx)[0] for cluster_idx in nearest_clusters
    ]).to("cuda")

    candidate_points = A[candidate_indices]

    candidate_distances = negative_dot_distance(candidate_points, X)
    #candidate_distances = -torch.sum(candidate_points * X, dim=-1)
    nearest_neighbors = candidate_indices[torch.argsort(candidate_distances)[:K2]]

    return nearest_neighbors

def our_ann(N, D, A, X, K):
    device = "cuda"
    #K_clusters = 5  # Number of clusters for K-Means
    
    K = min(int(np.sqrt(N)), 100)  # Number of clusters and neighbors to consider
    K1 = 2*K

    global dist_metric
    if dist_metric not in ["l2", "cosine"]:
        print(f"Warning: K-means only supports l2 and cosine distances. Using l2 instead of {dist_metric}.")
        dist_metric = "l2"  # Set a fallback metric

    if isinstance(A, np.ndarray):
        starttime = time.time()
        A = torch.as_tensor(A, device="cuda", dtype=torch.float32)
        print("A to torch tensor took - " + str(time.time() - starttime))

    starttime = time.time()
    
    max_iterations = 300 #decide
    centroid_shift_tolerance = 1e-5 # decide
    converged = False
    
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    # new_centroids = torch.zeros((K, D), device="cuda")
    
    # an empty matrix of shape (K, N) on the GPU, initialized with -1 (to represent empty slots)
    cluster_labels_batches = []
    cluster_labels = torch.full((K,N), -1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(K, dtype=torch.float32, device="cuda")
    distances = torch.empty(N, device="cuda")
    
    #------------------------------------------------------------------------#
    # Find the best batch size according to the available memory in the GPU and transfer A in batches
    MAX_FRACTION = 0.8
    MAX_BATCH_SIZE = 100_000
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory
    usable_memory = available_memory * MAX_FRACTION
    
    bytes_per_vec_element = 4
    bytes_per_vec = D * bytes_per_vec_element
    
    batch_size = int(usable_memory // bytes_per_vec) # num of vectors per batch
    batch_size = min(batch_size, MAX_BATCH_SIZE)  
    
    num_batches = (N + batch_size - 1) // batch_size
    # print("batch size for 1m: ", batch_size, "usable memory: ", usable_memory, "num batches: ", num_batches)
    
    A_gpu_batches = [] #does not need to be on the GPU
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        A_batch = (A[start_idx : end_idx])
        A_gpu_batches.append(A_batch)
    #--------------------------------------------------------------------------#
    
    #--------------------------------------------------------------------------------------#
    # this was used for transferring A to the GPU and choosing indices, without any batching
    # A_tensor = torch.from_numpy(A).to(dtype=torch.float32, device="cuda", non_blocking=True)
    # indices = torch.randint(0, N, (K,), device="cuda") # select K random indices from the indices of A
    # init_centroids_d = A_tensor[indices]  #filter the chosen K random vectors directly on GPU //question whether this is on the gpu
    #--------------------------------------------------------------------------------------#
    
    initial_indices = np.random.choice(N, K, replace=False)
    init_centroids_d = torch.tensor(A[initial_indices], device="cuda", dtype=torch.float32)

    #-----------------------------------------------------------------------------#
    # Use this initialisation of random centroids if you would like to compare sklearns with controlled init conditions in print_kmeans()
    # np.random.seed(2)
    # initial_indices = np.random.choice(N, K, replace=False)
    # init_centroids_d = torch.tensor(A[initial_indices], device="cuda", dtype=torch.float32)
    #-----------------------------------------------------------------------------#
    
    stream1 = torch.cuda.Stream()  # For distance calculation
    stream2 = torch.cuda.Stream()  # For centroid labels and counts calculation
    
    iteration = 0
    while not converged and iteration < max_iterations:
        iteration += 1
        
        # assign clusters to all vectors
        # start_idx = 0
        for i, batch in enumerate(A_gpu_batches):
            # start_idx = i * batch_size
            # end_idx = min((i + 1) * batch_size, N)
            new_centroids.zero_()
            counts.zero_()
            cluster_labels_batches.clear()
            
            #---- stream1
            with torch.cuda.stream(stream1):
                if dist_metric == "l2":
                    distances = torch.cdist(batch, init_centroids_d, p=2.0)**2
                elif dist_metric == "cosine":
                    A_norm = torch.nn.functional.normalize(batch, p=2, dim=1)
                    C_norm = torch.nn.functional.normalize(init_centroids_d, p=2, dim=0)
                    similarities = torch.matmul(A_norm, C_norm.T) #take transpose of centroids to make it (D,K) so matmul can give (N,K)
                    distances = 1 - similarities
                else:
                    raise ValueError("Invalid distance metric")
              
            #---- stream2
            with torch.cuda.stream(stream2):
                stream2.wait_stream(stream1)  # Ensure batch is available before computing
                
                batch_cluster_labels = torch.argmin(distances, dim=1)
                new_centroids.scatter_add_(0, batch_cluster_labels[:, None].expand(-1, D), batch.to(torch.float32)) # cumulatively adds all vectors belonging to the same cluster
                counts.scatter_add_(0, batch_cluster_labels, torch.ones_like(batch_cluster_labels, dtype=torch.float32)) # for each cluster index in batch_labels, it adds 1.0 to counts at that position.
                
                cluster_labels_batches.append(batch_cluster_labels) # append the batches of cluster labels to concatenate at the end 
                
        torch.cuda.synchronize()  
        
        cluster_labels = torch.cat(cluster_labels_batches, dim=0)
        counts[counts == 0] = 1  # avoid division by zero
        new_centroids /= counts.unsqueeze(1)

        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()

        # if torch.mean(centroid_shift) <= centroid_shift_tolerance:
        if torch.max(centroid_shift) <= centroid_shift_tolerance:
            converged = True
    
    print("our_kmeans took - " + str(time.time() - starttime))

    starttime = time.time()
    top_k_neighbors = ann_search(A, new_centroids, cluster_labels, X, K, K1)

    print('ann_search function took - ' + str(time.time() - starttime))
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

def test_distance_functions(dim=2, num_samples=1000):
    # Create random vectors for testing
    X_cpu = np.random.rand(dim).astype(np.float32)
    Y_cpu = np.random.rand(num_samples, dim).astype(np.float32)
    
    X_gpu = torch.from_numpy(X_cpu).cuda()
    Y_gpu = torch.from_numpy(Y_cpu).cuda()
    
    # Test each distance function
    for dist_name in ["cosine", "l2", "dot", "manhattan"]:
        # Get CPU distance function
        cpu_func = {
            "cosine": distance_cosine_cpu,
            "l2": distance_l2_cpu,
            "dot": distance_dot_cpu,
            "manhattan": distance_manhattan_cpu
        }[dist_name]
        
        # Get GPU distance function
        gpu_func = {
            "cosine": distance_cosine,
            "l2": distance_l2,
            "dot": distance_dot,
            "manhattan": distance_manhattan
        }[dist_name]
        
        # Measure CPU time (compute all distances)
        start = time.perf_counter()
        for i in range(num_samples):
            cpu_func(X_cpu, Y_cpu[i])
        cpu_time = time.perf_counter() - start
        
        # Measure GPU time (compute all distances)
        start = time.perf_counter()
        for i in range(num_samples):
            gpu_func(X_gpu, Y_gpu[i])
        torch.cuda.synchronize()  # Ensure GPU computations are complete
        gpu_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{dist_name.upper()} Distance (dim={dim}):")
        print(f"  CPU time: {cpu_time:.6f} seconds")
        print(f"  GPU time: {gpu_time:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x\n")

def test_distance_functions_batch(dim=2, num_samples=1000):
    # Create random vectors for testing
    X_cpu = np.random.rand(dim).astype(np.float32)
    Y_cpu = np.random.rand(num_samples, dim).astype(np.float32)
    
    X_gpu = torch.from_numpy(X_cpu).cuda()
    Y_gpu = torch.from_numpy(Y_cpu).cuda()
    
    # Test each distance function
    for dist_name in ["cosine", "l2", "dot", "manhattan"]:
        cpu_func = {
            "cosine": distance_cosine_cpu,
            "l2": distance_l2_cpu,
            "dot": distance_dot_cpu,
            "manhattan": distance_manhattan_cpu
        }[dist_name]
        
        gpu_func = {
            "cosine": distance_cosine,
            "l2": distance_l2,
            "dot": distance_dot,
            "manhattan": distance_manhattan
        }[dist_name]

        # CPU: Fully vectorized
        start = time.perf_counter()
        cpu_dists = np.apply_along_axis(cpu_func, 1, Y_cpu, X_cpu)
        cpu_time = time.perf_counter() - start

        # GPU: Fully vectorized
        start = time.perf_counter()
        if dist_name == "dot":
            gpu_dists = torch.matmul(Y_gpu, X_gpu)  # Batch-friendly dot product
        else:
            gpu_dists = gpu_func(Y_gpu, X_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"{dist_name.upper()} Distance (dim={dim}):")
        print(f"  CPU time: {cpu_time:.6f} seconds")
        print(f"  GPU time: {gpu_time:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x\n")

def test_kmeans_10():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("10_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (10): {cpu_time:.6f} seconds")
    plot = "10"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (10): {gpu_time:.6f} seconds")

    #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 10 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1000_2():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1000_2_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1000_2): {cpu_time:.6f} seconds")
    plot = "1000_2"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1000_2): {gpu_time:.6f} seconds")

    #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1000_2 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1000_1024():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1000_1024_meta.json")

    # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1000_1024): {cpu_time:.6f} seconds")
    plot = "1000_1024"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1000_1024): {gpu_time:.6f} seconds")

    #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1000_1024 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_100k():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("100k_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (100k): {cpu_time:.6f} seconds")
    plot = "100k"
    
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (100k): {gpu_time:.6f} seconds")

    #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 100k (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_100k_K30():
    # Load test data from JSON
    print("Starting testing: ")
    N, D, A, K = testdata_kmeans("100k_K30_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (100k_K30): {cpu_time:.6f} seconds")
    plot = "100k_K30"
    
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (100k_K30): {gpu_time:.6f} seconds")

    #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 100k_K30 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1m_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m): {cpu_time:.6f} seconds")
    plot = "1m"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_K10():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1m_K10_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_K10): {cpu_time:.6f} seconds")
    plot = "1m_K10"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_K10): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_K10 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_K50():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1m_K50_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_K50): {cpu_time:.6f} seconds")
    plot = "1m_K50"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_K50): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_K50 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_10_K10():
    # Load test data from JSON
    print("statrting k30")
    N, D, A, K = testdata_kmeans("1m_10_K10_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_10_K10): {cpu_time:.6f} seconds")
    plot = "1m_10_K10"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_10_K10): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_10_K10 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_50_K10():
    # Load test data from JSON
    print("statrting k30")
    N, D, A, K = testdata_kmeans("1m_50_K10_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_50_K30): {cpu_time:.6f} seconds")
    plot = "1m_50_K10"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_50_K10): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_50_K10 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_K30():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1m_K30_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_K30): {cpu_time:.6f} seconds")
    plot = "1m_K30"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_K30): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_K30 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_50_K30():
    # Load test data from JSON
    print("statrting k30")
    N, D, A, K = testdata_kmeans("1m_50_K30_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_50_K30): {cpu_time:.6f} seconds")
    plot = "1m_50_K30"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_50_K30): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_50_K30 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_1m_K100():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("1m_K100_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (1m_K100): {cpu_time:.6f} seconds")
    plot = "1m_K100"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (1m_K100): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 1m_K100 (CPU to GPU): {speedup:.2f}x\n")

def test_kmeans_2m_K50():
    # Load test data from JSON
    N, D, A, K = testdata_kmeans("2m_K50_meta.json")

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (2m_K50): {cpu_time:.6f} seconds")
    plot = "2m_K50"
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K, plot)
    print(f"GPU Time (2m_K50): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 2m_K50 (CPU to GPU): {speedup:.2f}x\n")

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
    
    # Conditional test execution based on the --test argument
    if args.test == 'dist':
        test_distance_functions(dim=2)
        test_distance_functions(dim=2**15)
        test_distance_functions_batch(dim=2)
        test_distance_functions_batch(dim=2**15)
    elif args.test == "knn":
        test_knn()
        test_knn_cpu()
        test_knn_2D()
        test_knn_215()
        test_knn_4k()
        test_knn_4m()
    elif args.test == "kmeans":
        test_kmeans_1000_2()
        # test_kmeans_1m_10_K10()
    elif args.test == "ann":
        # test_ann()
        test_ann_2D()
        test_ann_215()
        test_ann_4k()
        test_ann_40k()
        test_ann_4m()