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
    match D:
        case "cos":
            return distance_cosine(X,Y)
        case "l2":
            return distance_l2(X,Y)
        case "dot":
            return distance_dot(X,Y)
        case "man":
            return distance_manhattan(X,Y)
        case _:
            raise ValueError("Please provide a valid distance function.")
            

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

# # You can create any kernel here
# def distance_kernel(X, Y, D):
# #     pass
#     # A.size(0) = N
#     # compute the distances between X and each vector in A
#     distances = torch.zeros(A.size(0), device=A.device)
    
#     distances = torch.norm(A - X, dim=1)  # vectorized L2 norm computation

#     # sort distances and get the top-K nearest neighbors
#     _, indices = torch.topk(distances, k=K, largest=False)

#     # get the top-K nearest vectors using the indices
#     result = A[indices]  # The top K vectors from A

#     return result

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
    
    dist_metric = "cosine"
     # Create cuda streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # create an event for synchronization
    event = torch.cuda.Event()
    
    # define appropriate number of batches
    batch_num = None
    
    # find the batch_size with N number of vectors divided by our desired batch number
    batch_size = N // batch_num if N >= batch_num else N
    
    #divide the batches
    batches = torch.split(A, batch_size)
    
    # distances = torch.cuda.FloatTensor()
    distances = torch.empty(N, device="cuda")
    
    X_d = X.to("cuda") 
    
    # run through batches
    for batch_id, batch in batches:
        
        # batch_id, the current branch number, kacinci branch
        # batch_size: how many vectors in a batch
        start_pc = batch_id * batch_size
        
        # stream 1 operations
        with torch.cuda.stream(stream1):
            A_d = batch.to("cuda", non_blocking=True)
            event.record()
            
        # stream 2 operations
        with torch.cuda.stream(stream2):
            event.wait()

            for i, Y in enumerate (A_d):
                distances[start_pc + i] = distance_kernel(X_d, Y, dist_metric)
    
    # wait for all streams to to finish before proceeding with finding top-k
    torch.cuda.synchronize()
    
    # find the top k
    _, indices = torch.topk(distances, k=K, largest=False)
    indices_cpu = indices.cpu()
    result = A[indices_cpu]
    
    return result

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    
    # Pick K initial centroids
        # do this by picking K random elements from A for random centroid initialisation
    # Now iterate through A and assign each vector to the nearest centroid
        # do this by first computing distances from every point to all K centroids
        # then assign each point to the closest centroid
    # now for each cluster, compute the mean of all assigned vectors to find new centroids
    # if the centroids have changed in the iteration repeat until convergence (until they don't change anymore)
        #reassign points, recompute centroids and repeat until results are stable
    
    #Initialise Centroids, by selecting K random vectors from A
    init_centroid = random.choice(A)
    
    #do we need to put these parts in batches?
    # yes, we do. Again make A into batches. Make the batching into a function
    distances = torch.empty(N, device="cuda")
    for i, Y in enumerate (A):
            distances[start_pc + i] = distance_kernel(X_d, Y, dist_metric)

    
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
