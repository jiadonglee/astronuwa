import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
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

class GaussianProcess:
    
    def __init__(self, K, K_star):
        # self.kernel = kernel
        self.K = K
        self.K_star = K_star
        # self.K_star_star = K_star_star
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self):
        self.K_inv = np.linalg.inv(self.K)
        # print(np.min(self.K_inv), np.max(self.K_inv))
        mu = self.K_star @ self.K_inv @ self.y_train
        # cov = self.K_star_star - self.K_star @ self.K_inv @ self.K_star
        return mu
    

def train(radius=0.1, subsample_factor=1, n_iter=3, num_cpu=24, ):
    # Make the graph objects for all realisations in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        graph_list, degree_list = zip(*pool.map(make_graph, tqdm([(i, X_train[i], radius, subsample_factor) for i in range(X_train.shape[0])])))

        
    # Initialize the grakel graph kernel
    wl_kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram)
    args_list = [(i, graph_list[i], degree_list[i]) for i in range(len(graph_list))]

    # Calculate grakel graphs in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        grakel_list = list(tqdm(pool.imap(calculate_grakel_graph, args_list), total=len(graph_list)))

    # Normalize kernel values
    kernel_values = wl_kernel.fit_transform(grakel_list)
    kernel_values = kernel_values / np.max(kernel_values)
    
    return kernel_values, wl_kernel


def test(wl_kernel, K, radius=0.1, subsample_factor=1, n_iter=3, num_cpu=24):

    """test"""
    # Make the graph objects for all realisations in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        graph_list, degree_list = zip(*pool.map(make_graph, tqdm([(i, X_test[i], radius, subsample_factor) for i in range(X_test.shape[0])])))

        
    # wl_kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram)
    args_list = [(i, graph_list[i], degree_list[i]) for i in range(len(graph_list))]

    # Calculate grakel graphs in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        grakel_list_test = list(tqdm(pool.imap(calculate_grakel_graph, args_list), total=len(graph_list)))

    # Normalize kernel values
    K_star = wl_kernel.transform(grakel_list_test)
    K_star = K_star / np.max(K)
    return K_star


class wl_GaussianProcess:
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
    
    # def fit(self, X, y):
    #     self.X_train = X
    #     self.y_train = y
        
    # def predict(self, x):
    #     K_inv = np.linalg.inv(self.K)
    #     mu = self.K_star @ K_inv @ self.y_train
    #     return mu

if __name__ == "__main__":

    """hyper parameters"""
    num_cpu = 48
    radius  = 0.05
    subsample_factor = 1
    n_iter = 5

    # n_train = 500
    # n_star  = 2000
    """load data"""
    data_dir = "/nfsdata/users/jdli_ny/wlkernel/mock/"
    # data = np.load(data_dir+f'binary_train_flatZ_abg_{n_train}tr_{n_star}cmd.npz')
    
    n_train = 200
    n_star  = 5000
    # data = np.load(data_dir + f'binary_train_flatZ_abg_{n_train}tr_{n_star}cmd.npz')
    # data = np.load(data_dir+f'binary_train_flatZ_abg_mesh_{n_star}cmd.npz')
    dname = data_dir + 'binary_train_flatZ_abg_baseline.npz'
    # dname = data_dir + f'binary_train_moh_m0p5_0_abg_{n_train}tr_{n_star}cmd.npz'
    data  = np.load(dname)

    # Split X and Y into training and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(
        data['X'], data['Y'], test_size=0.1, random_state=42
    )
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


    """train and test"""

    # K, wl_kernel = train(
    #     radius=radius, subsample_factor=subsample_factor, 
    #     n_iter=n_iter, num_cpu=num_cpu
    #     )

    # K_star = test(
    #     wl_kernel, K,
    #     radius=radius, subsample_factor=subsample_factor, 
    #     n_iter=n_iter, num_cpu=num_cpu
    # )

    # gp = GaussianProcess(K, K_star)
    # gp.fit(X_train, Y_train)
    # y_pred = gp.predict()

    # model_dir = "/nfsdata/users/jdli_ny/wlkernel/model/"
    # # np.save(model_dir+f"K_flatZ_{n_train}tr_{n_star}cmd.npy",     kernel_values)
    # # np.save(model_dir+f"K_star_flatZ_{n_train}tr_{n_star}cmd.npy", K_star)

    # K_name = model_dir + f"K_flatZ_moh_m0p5_0.npy"
    # K_star_name = model_dir + f"K_star_moh_m0p5_0.npy"

    # print(f"save npy file {K_name}, {K_star_name}")
    # np.save(K_name,      K)
    # np.save(K_star_name, K_star)
    wl_gp = wl_GaussianProcess(
        radius=radius, subsample_factor=subsample_factor, 
        n_iter=n_iter, num_cpu=num_cpu
        )

    wl_gp.fit(X_train, Y_train)
    y_pred = wl_gp.predict(X_test)

    print(y_pred.shape)

    test_dir = "/nfsdata/users/jdli_ny/wlkernel/test/"
    np.save(test_dir+'ypred_baseline.npy', y_pred)
