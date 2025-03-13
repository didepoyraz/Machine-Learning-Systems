import torch
import cupy as cp
import numpy as np
import time
import json
from cuml.cluster import KMeans
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
def distance_kernel(X, Y, D):
    # matches to which distance function is desired to be used so none of the functions are in need of seperate calls
    """match D:
        case "cos":
            distance_cosine(X,Y)
        case "l2":
            distance_l2(X,Y)
        case "dot":
            distance_dot(X,Y)
        case "man":
            distance_manhattan(X,Y)
        case _:
            raise ValueError("Please provide a valid distance function.")
            """
            

def distance_cosine(X, Y):
    cosine_similarity = torch.nn.functional.cosine_similarity(X, Y, dim=0)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    return torch.norm(X - Y)

def distance_dot(X, Y):
    return torch.dot(X, Y)

def distance_manhattan(X, Y):
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
def distance_kernel(X, Y, D):
#     pass
    # A.size(0) = N
    # compute the distances between X and each vector in A
    distances = torch.zeros(A.size(0), device=A.device)
    
    distances = torch.norm(A - X, dim=1)  # vectorized L2 norm computation

    # sort distances and get the top-K nearest neighbors
    _, indices = torch.topk(distances, k=K, largest=False)

    # get the top-K nearest vectors using the indices
    result = A[indices]  # The top K vectors from A

    return result

def our_knn(N, D, A, X, K):
    
    # first divide the vector into batches for copying and processing the distances
    
    # 1. Copy the first batch to the GPU
    # 2. Compute the distances of all vectors in the batch with X
    # 3. Repeat for all batches
    
    # At the end find the top k
    
    # 2 different streams are needed, 1st for copying data, 2nd for computing the distances
    # at the end synchronize both streams and compute top k
    
    # potentially do this for larger than N vectors in a collection, or test it out to see
    # if it is still fast even with smaller vectors
    
    pass

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

def euclidean_distance(vec1, vec2):
    return cp.sqrt(cp.sum((vec1 - vec2) ** 2))
    
def ann_search(A, cluster_centers, cluster_assignments, X, K1, K2):
    """
    Perform Approximate Nearest Neighbor (ANN) search using K-Means clustering.
    
    Parameters:
        A: (N, D) CuPy array containing dataset vectors.
        cluster_centers: (K_clusters, D) CuPy array of cluster centers.
        cluster_assignments: (N,) CuPy array mapping each point to its cluster.
        X: (D,) CuPy array representing the query vector.
        K1: Number of nearest clusters to consider.
        K2: Number of nearest neighbors to return.
        
    Returns:
        result: (K2,) CuPy array containing the indices of the top K2 nearest neighbors.
    """
    # Step 1: Find the K1 nearest cluster centers
    cluster_distances = cp.linalg.norm(X - cluster_centers, axis=1)
    nearest_clusters = cp.argsort(cluster_distances)[:K1]  # Get top K1 cluster indices

    # Step 2: Gather points from the selected clusters
    candidate_indices = cp.concatenate([
        cp.where(cluster_assignments == cluster_idx)[0] for cluster_idx in nearest_clusters
    ])

    candidate_points = A[candidate_indices]  # Retrieve points from the dataset

    # Step 3: Find the K2 nearest neighbors within selected clusters
    candidate_distances = cp.linalg.norm(candidate_points - X, axis=1)
    nearest_neighbors = candidate_indices[cp.argsort(candidate_distances)[:K2]]

    return nearest_neighbors

def our_ann(N, D, A, X, K):
    """
    Performs Approximate Nearest Neighbors (ANN) search using GPU-accelerated K-Means.
    
    Parameters:
        N: Number of vectors.
        D: Dimension of vectors.
        A: (N, D) CuPy array containing dataset vectors.
        X: (D,) CuPy array representing the query vector.
        K: Top K nearest neighbors to return.
    
    Returns:
        Top K nearest neighbors indices.
    """
    K_clusters = 10 
    K1, K2 = 2, 15  

    kmeans = KMeans(n_clusters=K_clusters, max_iter=10)
    cluster_assignments = kmeans.fit_predict(A) 
    cluster_centers = kmeans.cluster_centers_  

    cluster_assignments = cp.array(cluster_assignments)
    cluster_centers = cp.array(cluster_centers)

    top_k_neighbors = ann_search(A, cluster_centers, cluster_assignments, X, K1, K2)

    print("Top K nearest neighbors indices:", top_k_neighbors.get())

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
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
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

if __name__ == "__main__":
    test_kmeans()
