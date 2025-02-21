import torch
import cupy as cp
import triton
import numpy as np
import random
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------
# You can create any kernel here
def distance_kernel(X, Y, D):

    # making sure X and Y are both CPU or both GPU
    if X.device != Y.device:
        raise ValueError("X and Y must be on the same device")
    # matches to which distance function is desired to be used so none of the functions are in need of seperate calls

    distance_func = {
         "cosine": distance_cosine,
        "l2": distance_l2,
        "dot": distance_dot,
        "manhattan": distance_manhattan
    }.get(D)   
    
    if distance_func is None:
        raise ValueError("Invalid distance metric. Choose from 'cosine', 'l2', 'dot', 'manhattan'.")

    distance_func(X,Y)

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

def divide_batches(batch_num, A, N):
    
    # find the batch_size with N number of vectors divided by our desired batch number
    batch_size = N // batch_num if N >= batch_num else N

    #divide to batches
    return torch.split(A, batch_size), batch_size

def find_distance_to_X(A, N, batch_num, X_d, dist_metric):
    # make all input parameters on the GPU already
    batches, batch_size = divide_batches(batch_num, A, N)
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    distances = torch.empty(N, device="cuda")
    
    for batch_id, batch in batches:
    
        # batch_id, the current branch number, kacinci branch
        # batch_size: how many vectors in a batch
        start_pc = batch_id * batch_size
        
        # stream 1 operations
        with torch.cuda.stream(stream1):
            A_d = batch.to("cuda", non_blocking=True)
            
        # stream 2 operations
        with torch.cuda.stream(stream2):
            stream2.wait_stream(stream1)

            for i, Y in enumerate (A_d):
                distances[start_pc + i] = distance_kernel(X_d, Y, dist_metric)

    # wait for all streams to to finish before proceeding with finding top-k
    torch.cuda.synchronize()
    
    return distances

    
def our_knn(N, D, A, X, K):
    #---------------------------------------------------------------------------------------------#
    # first divide the vector into batches for copying and processing the distances
    
    # 1. Copy the first batch to the GPU
    # 2. Compute the distances of all vectors in the batch with X
    # 3. Repeat for all batches
    
    # At the end find the top k
    
    # 2 different streams are needed, 1st for copying data, 2nd for computing the distances
    # at the end synchronize both streams and compute top k
    
    # potentially do this for larger than N vectors in a collection, or test it out to see
    # if it is still fast even with smaller vectors
    #--------------------------------------------------------------------------------------------#
    dist_metric = "cosine"

    # TODO define appropriate number of batches
    batch_num = None

    X_d = X.to("cuda") 
   
    distances = find_distance_to_X(A, N, batch_num, X_d, dist_metric)
    
    # find the top k
    _, indices = torch.topk(distances, k=K, largest=False)
    indices_cpu = indices.cpu()
    result = A[indices_cpu]
    
    return result

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

def our_knn_cpu(N, D, A, X, K, dist_metric="l2"):
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
    indices = np.argpartition(distances, K)[:K]

    return A[indices]

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    
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
    dist_metric = "cosine"
    batch_num = None
    
    max_iterations = 100 #decide
    centroid_shift_tolerance = None # decide
    converged = False
   
    #Initialise Centroids, by selecting K random vectors from A
    init_centroids = random.sample(A, K)
    init_centroids_d = [centroid.to("cuda") for centroid in init_centroids]  # Move to GPU
    
    new_centroids = torch.empty((K,D), dtype=torch.float32, device="cuda")
    
    distances =  torch.empty(K, device="cuda")
    
    # an empty matrix of shape (K, N) on the GPU, initialized with -1 (to represent empty slots)
    cluster_labels = torch.full((K,N), -1, dtype=torch.int32, device="cuda")
    cluster_distances = torch.full((K, N), float('inf'), dtype=torch.float32, device="cuda")  # stores min distances with infinity to differentiate if there is a distance written

    # to track how many vectors have been assigned per centroid
    centroid_counts = torch.zeros(K, dtype=torch.int32, device="cuda")
    
    #---------------------------------------------------------------------------------------#
    # todo set up loop structure
    # maybe put these for loops into streams because they depend on each other?
    iteration = 0
    while not converged and iteration < max_iterations:
        iteration += 1
        
        for i, centroid in enumerate(init_centroids_d):
            distances[i] = find_distance_to_X(A, N, batch_num, centroid, dist_metric)  # (K, N)
            # distances will have K rows and N column (for the number of vectors in A) and each row will have the distances to Ki from every other vector
    
        # TODO QUESTION: i copy A batch by batch in find_Distance_to_x, but then I need to do further operations with it and copy it again?
        
        # TODO go through the columns of distances, for each column find the index of minimum and assign that vector to the corresponding centroid
        
        for i, vec in enumerate(A_d):
            
            # find the closest centroid to vec
            distance_to_all_centroids = distances[:, i]
            min_distance, min_centroid_index = distance_to_all_centroids.min(dim=0)
            closest_centroid = min_centroid_index.item() # gives the row number (centroid number) of the closest centroid
            
            # Assign vector index i to the next free position in cluster_labels[closest_centroid, :] for vector index and corresponding minimum distance
            cluster_labels[closest_centroid, centroid_counts[closest_centroid]] = i
            cluster_distances[closest_centroid, centroid_counts[closest_centroid]] = min_distance
            
            # increment the counter for this centroid
            centroid_counts[closest_centroid] += 1
        
        for k in range(K):
            assigned_points = A[cluster_labels[k, cluster_labels[k] != -1]]
            if len(assigned_points) > 0:
                new_centroids[k] = assigned_points.mean(dim=0)  # Compute new mean
            # TODO: print the new centroids and confirm that they are in fact dimension D, vectors as new points
            
        centroid_shift = torch.norm(new_centroids - init_centroids_d, dim=1)
        init_centroids_d = new_centroids.clone()
        
        if torch.all(centroid_shift <=centroid_shift_tolerance):
            converged = True
            
    # now check if the centroids have changed after this iteration, either go on iterating until they do not change or until the max iteration count
    # # find the mean of the minimum distances per row
    # valid_entries = cluster_distances != float('inf') # make a boolean matrix that has 1s for each valid entry and 0s for infinities
    # sum_distances = torch.sum(cluster_distances * valid_entries, dim=1) # sum the entries that are valid per row
    # num_valid = torch.sum(valid_entries, dim=1) # the number of valid distances per centroid
    
    # average_distances_per_centroid = sum_distances / num_valid # new centroid mean points
   
    
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
