import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nuwa.wlkernel import make_graph, calculate_grakel_graph
import multiprocessing
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from scipy.sparse import lil_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
np.set_printoptions(suppress=True)
import pickle
from sklearn.base import BaseEstimator, RegressorMixin



class wl_GaussianProcess(BaseEstimator, RegressorMixin):
    
    def __init__(self, radius=0.1, subsample_factor=1, n_iter=3, num_cpu=24):
        self.radius = radius
        self.subsample_factor = subsample_factor
        self.n_iter = n_iter
        self.num_cpu = num_cpu
        self.wl_kernel = None
    
    def fit(self, X_train, y_train):

        self.X_train, self.y_train = X_train, y_train
        # Make the graph objects for all realizations in parallel
        with multiprocessing.Pool(processes=self.num_cpu) as pool:
            graph_list, degree_list = zip(*pool.map(make_graph, tqdm([(i, X_train[i], self.radius, self.subsample_factor) for i in range(X_train.shape[0])])))
        
        # Initialize the grakel graph kernel
        self.wl_kernel = WeisfeilerLehman(n_iter=self.n_iter, base_graph_kernel=VertexHistogram)
        args_list = [(i, graph_list[i], degree_list[i]) for i in range(len(graph_list))]
        
        # Calculate grakel graphs in parallel
        with multiprocessing.Pool(processes=self.num_cpu) as pool:
            grakel_list = list(tqdm(pool.imap(calculate_grakel_graph, args_list), total=len(graph_list)))
        
        # Normalize kernel values
        kernel_values = self.wl_kernel.fit_transform(grakel_list)
        self.K = kernel_values / np.max(kernel_values)
        
        return kernel_values
    
    def predict(self, X_test):
        # Make the graph objects for all realizations in parallel
        with multiprocessing.Pool(processes=self.num_cpu) as pool:
            graph_list, degree_list = zip(*pool.map(make_graph, tqdm([(i, X_test[i], self.radius, self.subsample_factor) for i in range(X_test.shape[0])])))

        args_list = [(i, graph_list[i], degree_list[i]) for i in range(len(graph_list))]

        # Calculate grakel graphs in parallel
        with multiprocessing.Pool(processes=self.num_cpu) as pool:
            grakel_list_test = list(tqdm(pool.imap(calculate_grakel_graph, args_list), total=len(graph_list)))

        # Normalize kernel values
        K_star = self.wl_kernel.transform(grakel_list_test)
        self.K_star = K_star / np.max(K_star)

        K_inv = np.linalg.inv(self.K)
        mu = self.K_star @ K_inv @ self.y_train
        return mu
    