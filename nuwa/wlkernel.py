import os
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from scipy.sparse import lil_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import numpy as np


def make_graph(arg):
    
    i, X_data, radius, subsample_factor = arg
    # Construct the KD-tree
    tree = KDTree(X_data)

    # Find all points within 'radius' of each point
    ind, dist = tree.query_radius(X_data, r=radius, return_distance=True)

    # Create the adjacency matrix using sparse matrix
    A = lil_matrix((len(X_data), len(X_data)), dtype=int)
    for idx, neighbors in enumerate(ind):
        A[idx, neighbors] = 1

    # Delete self-loops
    A = A - lil_matrix(np.eye(len(X_data)))

    # Create a degree dictionary ensuring the indices are in the closest integer 
    degree_dict = {i: int(A[i, :].sum()) for i in range(X_data.shape[0])}

    # Subsample the graph
    A = A[::subsample_factor, ::subsample_factor]
    degree_dict = {k: v for k, v in degree_dict.items() if k % subsample_factor == 0}

    # Reindex the keys in the degree dictionary
    degree_dict = {k // subsample_factor: v for k, v in degree_dict.items()}

    return A, degree_dict



# Grakel Graph Generation
def calculate_grakel_graph(args):
    i, adjacency_matrix, node_attributes = args
    return Graph(initialization_object=adjacency_matrix, node_labels=node_attributes)



class GaussianProcess:
    
    def __init__(self, K, K_star, K_star_star=None):
        # self.kernel = kernel
        self.K = K
        self.K_star = K_star
        # self.K_star_star = K_star_star
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self):
        self.K_inv = np.linalg.inv(self.K)
        mu = self.K_star @ self.K_inv @ self.y_train
        # cov = self.K_star_star - self.K_star @ self.K_inv @ self.K_star
        return mu