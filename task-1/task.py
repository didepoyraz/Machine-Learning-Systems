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
import math
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
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
    '''
    Calculate distance between two vectors using cosine similarity.

    Parameters:
        X (torch.Tensor): A 1D tensor representing the first vector.
        Y (torch.Tensor): A 1D tensor representing the second vector.

    Returns:
        torch.Tensor: A scalar tensor representing the cosine distance (1 - cosine similarity).
    '''
    cosine_similarity = torch.nn.functional.cosine_similarity(X, Y, dim=0)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    '''
    Calculate distance between two vectors using l2 (Euclidean) distance.

    Parameters:
        X (torch.Tensor): A 1D tensor representing the first vector.
        Y (torch.Tensor): A 1D tensor representing the second vector.

    Returns:
        torch.Tensor: A scalar tensor representing the l2 distance.
    '''
    return torch.norm(X - Y)

def distance_dot(X, Y):
    '''
    Calculate distance between two vectors using dot product

    Parameters:
        X (torch.Tensor): A 1D tensor representing the first vector.
        Y (torch.Tensor): A 1D tensor representing the second vector.

    Returns:
        torch.Tensor: A scalar tensor representing the dot product.
    '''
    return torch.dot(X, Y)

def distance_manhattan(X, Y):
    '''
    Calculate distance between two vectors using Manhattan distance.

    Parameters:
        X (torch.Tensor): A 1D tensor representing the first vector.
        Y (torch.Tensor): A 1D tensor representing the second vector.

    Returns:
        torch.Tensor: A scalar tensor representing the L1 (Manhattan) distance.
    '''
    return torch.sum(torch.abs(X - Y))

def get_distance_function():
    '''
    Returns:
        Distance function to be used.
    '''
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
    '''
    Perform K-Nearest Neighbors (KNN) search using a specified distance metric.

    This function computes the K nearest neighbors of a query vector `X` within a dataset `A`,
    optionally batching the computation to handle large datasets with limited GPU memory.
    It supports multiple distance metrics and automatically selects batch size based on available GPU memory.

    Parameters:
        N (int): Number of vectors in the dataset A.
        D (int): Dimensionality of each vector.
        A (Union[np.ndarray, torch.Tensor]): Dataset of shape (N, D), can be a NumPy array or PyTorch tensor.
        X (Union[np.ndarray, torch.Tensor]): Query vector of shape (D,) as NumPy array or PyTorch tensor.
        K (int): Number of nearest neighbors to retrieve.

    Returns:
        np.ndarray: Array of indices (shape: (K,)) of the top K nearest neighbors in the dataset A.
    '''
    global dist_metric

    if dist_metric is None:
        raise ValueError("Distance metric not set. Please specify one via command-line arguments.")
    
    X_tensor = X.cuda() if isinstance(X, torch.Tensor) else torch.from_numpy(X).cuda()

    if isinstance(A, np.ndarray):
        A_source = 'numpy'
    else:
        A_source = 'tensor'
    
    if N <= 100000:
        A_tensor = torch.from_numpy(A).cuda(non_blocking=True) if A_source == 'numpy' else A  
    
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
            
    if isinstance(A, np.ndarray):
        A_batch = torch.from_numpy(A[:batch_size]).cuda(non_blocking=True)
    else:
        # A is already a tensor, just get the batch
        A_batch = A[:batch_size] 
    min_heap = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        with torch.cuda.stream(stream1):
        # Transfer next batch to the GPU
            if i < num_batches - 1:
                if A_source == 'numpy':
                    A_batch = torch.from_numpy(A[end_idx : end_idx + batch_size]).cuda(non_blocking=True)
                else:  # A is already a tensor
                    A_batch = A[end_idx : end_idx + batch_size].cuda(non_blocking=True)

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
    '''
    Computes cosine distance (1 - cosine similarity) between two 1D NumPy vectors.

    Parameters:
        X (np.ndarray): A 1D NumPy array representing the first vector.
        Y (np.ndarray): A 1D NumPy array representing the second vector.
    
    Returns:
        float: The computed cosine similarity score between X and Y.

    '''
    dot_product = np.dot(X, Y)
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    return 1 - (dot_product / (norm_X * norm_Y))

def distance_l2_cpu(X, Y):
    '''
    Computes l2 distance between two 1D NumPy vectors.

    Parameters:
        X (np.ndarray): A 1D NumPy array representing the first vector.
        Y (np.ndarray): A 1D NumPy array representing the second vector.
    
    Returns:
        float: The computed l2 distance between X and Y.

    '''
    return np.linalg.norm(X - Y)

def distance_dot_cpu(X, Y):
    '''
    Computes dot product between two 1D NumPy vectors.

    Parameters:
        X (np.ndarray): A 1D NumPy array representing the first vector.
        Y (np.ndarray): A 1D NumPy array representing the second vector.
    
    Returns:
        float: The computed dot product score between X and Y.

    '''
    return np.dot(X, Y)

def distance_manhattan_cpu(X, Y):
    '''
    Computes manhattan distancebetween two 1D NumPy vectors.

    Parameters:
        X (np.ndarray): A 1D NumPy array representing the first vector.
        Y (np.ndarray): A 1D NumPy array representing the second vector.
    
    Returns:
        float: The manhattan distance between X and Y.

    '''
    return np.sum(np.abs(X - Y))

def our_knn_cpu(N, D, A, X, K):
    '''
    Performs K-Nearest Neighbors search entirely on CPU using the specified distance metric.

        Parameters:
            N (int): Number of vectors in dataset A.
            D (int): Dimensionality of each vector.
            A (np.ndarray): Dataset of shape (N, D).
            X (np.ndarray): Query vector of shape (D,).
            K (int): Number of nearest neighbors to retrieve.

        Returns:
            np.ndarray: Array of indices (shape: (K,)) of the top K nearest neighbors.

    '''
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

NUM_INIT = None 

def our_kmeans(N, D, A, K):
    '''
    Performs KMeans clustering on a large dataset using CUDA-enabled PyTorch 
    for high performance on GPU. Supports 'l2' and 'cosine' distance metrics.

    Parameters:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray): Input dataset of shape (N, D), in NumPy format.
        K (int): Number of clusters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - cluster_labels (np.ndarray): Array of shape (N,) containing the cluster assignment for each point.
            - centroids (np.ndarray): Array of shape (K, D) representing the final cluster centers.

    '''

    global dist_metric
    if dist_metric not in ["l2", "cosine"]:
        print(f"Warning: K-means only supports l2 and cosine distances. Using l2 instead of {dist_metric}.")
        dist_metric = "l2"  # Set a fallback metric
    
    max_iterations = 150 
    centroid_shift_tolerance = 1e-5 
    converged = False
    
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    cluster_labels_batches = []
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
    
    A_gpu_batches = [] #does not need to be on the GPU
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        
        A_batch = torch.from_numpy(A[start_idx : end_idx]).to(dtype=torch.float32, device="cuda", non_blocking=True)
        A_gpu_batches.append(A_batch)
    #--------------------------------------------------------------------------#
    #----Choose a random initialisation point for K centroids from A
    np.random.seed()
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
        new_centroids.zero_()
        counts.zero_()
        cluster_labels_batches = []
       
        for i, batch in enumerate(A_gpu_batches):

            #---- stream1
            with torch.cuda.stream(stream1):
                if dist_metric == "l2":
                    distances = torch.sum((batch[:,None] - init_centroids_d)**2, dim=2)
                    # distances = torch.cdist(batch, init_centroids_d, p=2) ** 2
                elif dist_metric == "cosine":
                    A_norm = torch.nn.functional.normalize(batch, p=2, dim=1)
                    C_norm = torch.nn.functional.normalize(init_centroids_d, p=2, dim=0)
                    similarities = torch.matmul(A_norm, C_norm.T) #take transpose of centroids to make it (D,K) so matmul can give (N,K)
                    distances = 1 - similarities
                else:
                    raise ValueError("Invalid distance metric")
              
            #---- stream2
            with torch.cuda.stream(stream2):
                stream2.wait_stream(stream1)  # To be able to ensure batch is available before computing
                
                batch_cluster_labels = torch.argmin(distances, dim=1)
                new_centroids.scatter_add_(0, batch_cluster_labels[:, None].expand(-1, D), batch.to(torch.float32)) # Cumulatively adds all vectors belonging to the same cluster
                counts.scatter_add_(0, batch_cluster_labels, torch.ones_like(batch_cluster_labels, dtype=torch.float32)) # For each cluster index in batch_labels, it adds 1.0 to counts at that position.
                
                cluster_labels_batches.append(batch_cluster_labels) # Append the batches of cluster labels to concatenate at the end 
                
        torch.cuda.synchronize()  
        
        
        counts[counts == 0] = 1  # This is done in order to avoid division by zero
        new_centroids /= counts.unsqueeze(1)

        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()
        
        if torch.max(centroid_shift) <= centroid_shift_tolerance:
            converged = True
      
        
    cluster_labels = torch.cat(cluster_labels_batches, dim=0)
    #---Enable this line if a comparison between the library function and our implementation for KMeans is needed.
    #print_kmeans(A, K, new_centroids, cluster_labels, initial_indices)
    
    return cluster_labels.cpu().numpy(), new_centroids.cpu().numpy() # decide on the return value based on what is needed for 2.2


def print_kmeans(A, N, K, new_centroids, cluster_labels, initial_indices):
    '''
    Call this function at the end of kmeans if a detailed analysis between the sklearns KMeans library function and our gpu KMeans is desired. 
    It will output the Squared Sum of Differences (SSD), and the average centroid difference to see the accuracy of our implementation.

    Parameters:
        A (np.ndarray): The original dataset of shape (N, D).
        N (int): Number of data points.
        K (int): Number of clusters.
        new_centroids (torch.Tensor): Final centroids from the GPU-based KMeans.
        cluster_labels (torch.Tensor): Cluster assignments from the GPU-based KMeans.
        initial_indices (np.ndarray): Indices of the data points used to initialize centroids.

    Outputs:
        Prints the following metrics:
            - Sum of Squared Differences (SSD) for both implementations.
            - Difference between SSDs.
            - Average L2 distance between corresponding centroids.
            - Whether the cluster assignments match exactly.
    '''

    init_centroids = A[initial_indices]  
    cluster_labels = cluster_labels.cpu().numpy()
    new_centroids = new_centroids.cpu().numpy()
    # change init to "kmeans++" if you would like to see better init conditions for improved cluster alignment
    sklearn_kmeans = KMeans(n_clusters=K, init=init_centroids, n_init=1, max_iter=150)
    sklearn_kmeans.fit(A)

    gpu_ssd = ((A - new_centroids[cluster_labels]) ** 2).sum().item()
    print(f"GPU K-Means SSD: {gpu_ssd}")

    sklearn_ssd = sklearn_kmeans.inertia_
    print(f"Sklearn K-Means SSD: {sklearn_ssd}")

    ssd_diff = abs(gpu_ssd - sklearn_ssd)
    print(f"SSD Difference: {ssd_diff:.4f}")
    torch_centroids = new_centroids
    sklearn_centroids = sklearn_kmeans.cluster_centers_

    centroid_error = np.linalg.norm(torch_centroids - sklearn_centroids, axis=1).mean()
    print(f"Average centroid difference: {centroid_error:.4f}")

    if np.array_equal(cluster_labels, sklearn_kmeans.labels_):
        print("All cluster labels match exactly!")
    else:
        print("Some cluster assignments differ, skleanrs")

def our_kmeans_cpu(N, D, A, K):
    '''
    KMeans CPU implementation imitating our exact logic of the GPU implementation for performance comparison

    Parameters:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray): Input dataset of shape (N, D), in NumPy format.
        K (int): Number of clusters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - cluster_labels (np.ndarray): Array of shape (N,) containing the cluster assignment for each point.
            - centroids (np.ndarray): Array of shape (K, D) representing the final cluster centers.

    '''
    
    max_iterations=150
    tol=1e-5
    #---Randomly initialize K centroids from data points
    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices]
    
    converged = False
    iteration = 0
    
    while not converged and iteration < max_iterations:
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

def euclidean_distance_batched(vecs, query, batch_size=100_000):
    '''
    Computes Euclidean (L2) distances between a query vector and a large set of vectors in batches.

    Parameters:
        vecs (torch.Tensor): Tensor of shape (N, D) containing N vectors.
        query (torch.Tensor): A single vector of shape (D,) to compare against.
        batch_size (int): Number of vectors to process per batch (default: 100,000).

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the L2 distances.
    '''
    distances = []
    for i in range(0, vecs.shape[0], batch_size):
        chunk = vecs[i:i+batch_size]
        dist = torch.norm(chunk - query, dim=1)
        distances.append(dist)
    return torch.cat(distances)

def negative_dot_distance_batched(vecs, query, batch_size=100_000):
    '''
    Computes negative dot product distances between a query vector and a large set of vectors in batches. 
    Negative to maintain the less is closer logic.

    Parameters:
        vecs (torch.Tensor): Tensor of shape (N, D) containing N vectors.
        query (torch.Tensor): A single vector of shape (D,) to compare against.
        batch_size (int): Number of vectors to process per batch (default: 100,000).

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the negative dot product values.
    '''
    distances = []
    for i in range(0, vecs.shape[0], batch_size):
        chunk = vecs[i:i+batch_size]
        dist = -torch.sum(chunk * query, dim=1)
        distances.append(dist)
    return torch.cat(distances)

def manhattan_distance_batched(vecs, query, batch_size=100_000):
    '''
    Computes Manhattan (L1) distances between a query vector and a large set of vectors in batches.

    Parameters:
        vecs (torch.Tensor): Tensor of shape (N, D) containing N vectors.
        query (torch.Tensor): A single vector of shape (D,) to compare against.
        batch_size (int): Number of vectors to process per batch (default: 100,000).

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the L1 distances.
    '''

    distances = []
    for i in range(0, vecs.shape[0], batch_size):
        chunk = vecs[i:i+batch_size]
        dist = torch.sum(torch.abs(chunk - query), dim=1)
        distances.append(dist)
    return torch.cat(distances)

def cosine_distance_batched(vecs, query, batch_size=100_000):
    '''
    Computes cosine distances between a query vector and a large set of vectors in batches.
    Distance is defined as (1 - cosine similarity).

    Parameters:
        vecs (torch.Tensor): Tensor of shape (N, D) containing N vectors.
        query (torch.Tensor): A single vector of shape (D,) to compare against.
        batch_size (int): Number of vectors to process per batch (default: 100,000).

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the cosine distances.
    '''
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    distances = []
    for i in range(0, vecs.shape[0], batch_size):
        chunk = vecs[i:i+batch_size]
        chunk = torch.nn.functional.normalize(chunk, p=2, dim=1)
        dist = 1 - torch.sum(chunk * query, dim=1)
        distances.append(dist)
    return torch.cat(distances)

def numpy_to_cuda(arr, dtype=torch.float32):
    '''
    Transfers a NumPy array to the CUDA device as a PyTorch tensor with optional dtype.

    Parameters:
        arr (np.ndarray): NumPy array to be transferred to the GPU.
        dtype (torch.dtype): Desired data type of the tensor (default: torch.float32).

    Returns:
        torch.Tensor: The input array converted to a pinned-memory CUDA tensor.
    '''
    return torch.from_numpy(arr).pin_memory().to(dtype=dtype, device="cuda", non_blocking=True)

def our_kmeans_for_ann(N, D, A, K):

    '''
    KMeans for ANN, difference from the task 2.1 KMeans implementation: returns the output in the GPU instead of the CPU 
    to use memory and time efficiently and prevent redundant transfers

    Parameters:
        N (int): Number of data points.
        D (int): Dimension of each data point.
        A (np.ndarray or torch.Tensor): Input dataset of shape (N, D), in NumPy format.
        K (int): Number of clusters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - cluster_labels (torch.Tensor): Tensor of shape (N,) containing the cluster assignment for each point.
            - centroids (torch.Tensor): Tensor of shape (K, D) representing the final cluster centers.

    '''
    global dist_metric
    if dist_metric not in ["l2", "cosine"]:
        print(f"Warning: K-means only supports l2 and cosine distances. Using l2 instead of {dist_metric}.")
        dist_metric = "l2"  # Set a fallback metric
    
    max_iterations = 150 #decide
    centroid_shift_tolerance = 1e-5 # decide
    converged = False
    
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    

    cluster_labels_batches = []
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
    A_gpu_batches = [] #does not need to be on the GPU
    if isinstance(A, torch.Tensor) and A.device.type == "cuda":
    # A is already a CUDA tensor, just slice it into batches
        for i in range(0, N, batch_size):
            A_gpu_batches.append(A[i:i+batch_size])
    else:
        # A is still a NumPy array, move it to GPU in batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            A_batch = torch.from_numpy(A[start_idx:end_idx]).to(
                dtype=torch.float32, device="cuda", non_blocking=True
            )
            A_gpu_batches.append(A_batch)
    #--------------------------------------------------------------------------#
    
    np.random.seed()
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
        new_centroids.zero_()
        counts.zero_()
        cluster_labels_batches = []
       
        for i, batch in enumerate(A_gpu_batches):

            #---- stream1
            with torch.cuda.stream(stream1):
                if dist_metric == "l2":
                    distances = torch.sum((batch[:,None] - init_centroids_d)**2, dim=2)
                    # distances = torch.cdist(batch, init_centroids_d, p=2) ** 2
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
        
        
        counts[counts == 0] = 1  # avoid division by zero
        new_centroids /= counts.unsqueeze(1)

        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()
        
        if torch.max(centroid_shift) <= centroid_shift_tolerance:
            converged = True

    cluster_labels = torch.cat(cluster_labels_batches, dim=0)
    return cluster_labels, new_centroids # decide on the return value based on what is needed for 2.2

def ann_search(A, cluster_centers, cluster_assignments, X, K1, K2, batch_size = 100_000):
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
        np.ndarray: NumPy array of shape (K2, D) containing the indices (from A) of the K2 nearest neighbors to the query vector X.
    """

    if dist_metric == "l2":
        cluster_distances = euclidean_distance_batched(cluster_centers, X)
    elif dist_metric == "cosine":
        cluster_distances = cosine_distance_batched(cluster_centers, X)
    elif dist_metric == "manhattan":
        cluster_distances = manhattan_distance_batched(cluster_centers, X)
    elif dist_metric == "dot":
        cluster_distances = negative_dot_distance_batched(cluster_centers, X) 
    else:
        raise ValueError(f"Unsupported distance metric: {dist_metric}")
    
    nearest_clusters = torch.argsort(cluster_distances)[:K1]

    # Safely gather candidate indices and points
    
    candidate_indices = []
    for cluster_idx in nearest_clusters:
        idx = torch.where(cluster_assignments == cluster_idx)[0]
        candidate_indices.append(idx)

    candidate_indices = torch.cat(candidate_indices)
    # Batch the candidate point gathering (optional for large A)

    candidate_points = []
    for i in range(0, candidate_indices.shape[0], batch_size):
        idx_chunk = candidate_indices[i:i+batch_size]
        candidate_points.append(A[idx_chunk])
    candidate_points = torch.cat(candidate_points, dim=0)

    # pass CUDA tensors directly to our_knn
    top_k_local_indices = our_knn(
        N=candidate_points.shape[0],
    D=X.shape[0],
        A=candidate_points,  # Already a tensor
        X=X,                 # Already a tensor
        K=K2
    )
    
    # Map local indices back to original dataset indices
    nearest_neighbors = candidate_indices[top_k_local_indices]
    return nearest_neighbors

def our_ann(N, D, A, X, K):
    '''
    Perform Approximate Nearest Neighbor (ANN) search with adaptive K-Means clustering.

    Parameters:
        N (int): Number of data points in the dataset.
        D (int): Dimensionality of each data point.
        A (np.ndarray): Array of shape (N, D) containing the dataset vectors.
        X (np.ndarray or torch.Tensor): Array/Tensor of shape (D,) representing the query vector.
        K (int): Number of nearest neighbors to retrieve.
    
    Returns:
        np.ndarray: NumPy array of shape (K2, D) containing the indices (from A) of the K2 nearest neighbors to the query vector X.
    '''
    device = "cuda"
    #K_clusters = 5
    print(str(K) + " nearest neighbors to find")
    K2 = K
    i = 5
    #K = 3
    if N < 5000:
        K = 3
        K1 = math.ceil(K * 0.6)
    elif N < 10000:
        K = 5
        K1 = math.ceil(K * 0.6)
    elif N < 100000:
        K = 10
        K1 = math.ceil(K * 0.6)
    elif N < 1000000:
        K = 100
        K1 = math.ceil(K * 0.4)
    else:
        K = 250
        K1 = math.ceil(K * 0.4)

    best_ssd = float("inf")
    best_centroids = None
    best_cluster_labels = None
    A = numpy_to_cuda(A)
    
    for i in range(i):
        cluster_labels, new_centroids = our_kmeans_for_ann(N,D,A,K)
        ssd = torch.sum((A - new_centroids[cluster_labels]) ** 2).item()
        
        if ssd < best_ssd:
            best_ssd = ssd
            best_centroids = new_centroids
            best_cluster_labels = cluster_labels

    if isinstance(best_centroids, np.ndarray):
        best_centroids = numpy_to_cuda(best_centroids)
    if isinstance(best_cluster_labels, np.ndarray):
        best_cluster_labels = torch.from_numpy(best_cluster_labels).to(
            dtype=torch.long, device="cuda", non_blocking=True
        )

    if isinstance(X, np.ndarray):
        X = numpy_to_cuda(X)

    starttime = time.time()
    top_k_neighbors = ann_search(A, best_centroids, best_cluster_labels, X, K1, K2)

    print('ann_search function took - ' + str(time.time() - starttime))
    print("ANN - Top K nearest neighbors indices:", top_k_neighbors)
    return top_k_neighbors

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------    
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

# incorporate timing into testing
def measure_time(func, *args, **kwargs):
    '''
    Measures time taken to execute the functions.
    '''
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure that GPU computation has finished
    end = time.perf_counter()
    return result, end - start

def test_knn(filename=""):
    '''
    Reads given file and performs KNN tests.

    Parameters:
        filename (str): Name of the JSON that contains meta data for the test.

    '''
    print("\n\nTesting for " + filename)
    # Load test data from JSON (using testdata_knn)
    N, D, A, X, K = testdata_knn(filename)

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

def test_distance_functions(dim=2, num_samples=1000):
    '''
    Reads given file and performs distance function tests.

    Parameters:
        filename (str): Name of the JSON that contains meta data for the test.
        
    '''
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
    '''
    Reads given file and performs batched distance function tests.

    Parameters:
        filename (str): Name of the JSON that contains meta data for the test.
        
    '''
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

def test_kmeans(filename=""):
    '''
    Reads given file and performs KMeans tests.

    Parameters:
        filename (str): Name of the JSON that contains meta data for the test.
        
    '''
    print("\n\nTesting for " + filename)
    N, D, A, K = testdata_kmeans(filename)

    # # Measure CPU time
    _, cpu_time = measure_time(our_kmeans_cpu, N, D, A, K)
    print(f"CPU Time (2m_K50): {cpu_time:.6f} seconds")
    # Measure GPU time
    _, gpu_time = measure_time(our_kmeans, N, D, A, K)
    print(f"GPU Time (2m_K50): {gpu_time:.6f} seconds")

    # #Calculate Speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')  # Avoid division by zero
    print(f"Speedup 2m_K50 (CPU to GPU): {speedup:.2f}x\n")

def test_ann(filename=""):
    '''
    Reads given file and performs ANN tests.

    Parameters:
        filename (str): Name of the JSON that contains meta data for the test.
        
    '''
    print("\n\nTesting for " + filename)

    # Load test data from JSON
    N, D, A, X, K = testdata_knn(filename)

    print(f"{N} vectors with dimension D = {D}")
    print(f"Finding {K} nearest vectors to vector X")

    # Measure GPU time
    knn_result, gpu_time = measure_time(our_knn, N, D, A, X, K)
    print(f"KNN: {gpu_time:.6f} seconds")

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
        files = ["2d_meta.json", "215_meta.json", "4k_meta.json", "40k_meta.json" + "4m_meta.json"]
        '''
        for file in files:
            try:
                test_knn(file)
            except Exception as e:
                print("Error: ) + str(e))
        '''
        test_knn("4m_meta.json")
    elif args.test == "kmeans":
        #files = ["2m_K50_meta.json", "1m_K100_meta.json", "1m_50_K30_meta.json", "1000_215_meta.json"]
        test_kmeans("1000_215_meta.json")
        '''
        for file in files:
            try:
                test_kmeans(file)
            except Exception as e:
                print("Error: " + str(e))
        '''
    elif args.test == "ann":
        files = ["ann_test1", "ann_test2", "ann_test3", "ann_test4", "ann_test5", "ann_test6", "ann_test7", "ann_test8", "ann_test9"]
        for file in files:
            try:
                test_ann(file + "_meta.json")
            except Exception as e:
                print("Error: " + str(e))
        #test_ann("ann_test8_meta.json")
