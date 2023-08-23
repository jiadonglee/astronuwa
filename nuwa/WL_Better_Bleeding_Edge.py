import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import os
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import multiprocessing
import flow_torch 
import argparse
from scipy.sparse import lil_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# Define global variables
n_realization = 600
n_sample = int(10**5)
num_cpu = 56

# ----------------------------------------------------------------------------------------------------------------------
# Simulation Generator Class
# ----------------------------------------------------------------------------------------------------------------------

class SimulationGenerator:
    def __init__(self, power_law_slope=-1.5, high_mass_cutoff=100, n_sample=10**5):
        # Initialize power law slope, high mass cutoff and the number of samples
        self.power_law_slope = power_law_slope
        self.high_mass_cutoff = high_mass_cutoff
        self.n_sample = n_sample

        self.X_data = None
        self.X_center = None
        self.flow = flow_torch.NormalizingFlow(22, 8)  # Initialize NormalizingFlow

    def read_cluster_center(self):
        # Load the trained normalizing flow
        state_dict = torch.load('flow_results_joint_all.pth', map_location=torch.device('cpu'))
        self.flow.load_state_dict(state_dict)

        # Generate samples from the trained flow
        x_sample = self.flow.dist.sample(torch.Size([self.n_sample, ]))
        X_flow, _ = self.flow.backward(x_sample)
        X_flow = X_flow.detach().numpy()

        # Exclude the first few dummy variables and select relevant dimensions
        X_flow = X_flow[:, 5:]
        X_flow = np.delete(X_flow, [5, 9], axis=1)

        self.X_center = X_flow

    def generate_clusters(self):
        # Generate clusters based on the power law distribution
        num_target = self.n_sample
        star_count = 0
        cluster_count = 0

        X_data = np.zeros((num_target, 15))  # Initial empty array for clusters
        num_count = np.zeros(num_target)

        # Define the normalization constant for the power law distribution
        Cnorm = (1 + self.power_law_slope) / (
                self.high_mass_cutoff ** (1 + self.power_law_slope) - 1 ** (1 + self.power_law_slope))

        # Define photon noise
        photon_noise = np.array([9.58e-03, 3.60e-02, 0.011, 1.80e-02, 9.72e-03,
                                     2.95e-02, 3.10e-02, 1.14e-02, 3.73e-02, 2.98e-02,
                                     1.43e-02, 0.008, 3.08e-02, 1.26e-02, 3.01e-02])

        # Generate clusters until we reach the desired number of stars
        while star_count < num_target:
            # Determine the number of stars in the current cluster
            num_sample = (np.random.uniform() * (1 + self.power_law_slope) / Cnorm +
                          1 ** (1 + self.power_law_slope)) ** (1. / (1. + self.power_law_slope))
            num_sample = int(num_sample) + (np.random.uniform() < (num_sample - int(num_sample)))

            # If the total star count exceeds the target, adjust the number of samples
            if star_count + num_sample > num_target:
                num_sample = num_target - star_count

            # Generate the data for the current cluster and increment counters
            X_data[star_count:star_count + num_sample, :] = self.X_center[cluster_count, :] + np.random.normal(
                size=(num_sample, 15))*photon_noise

            num_count[star_count:star_count + num_sample] = num_sample

            star_count += num_sample
            cluster_count += 1

        self.X_data = X_data

        # Cull the list of centers to only include those that are used
        self.X_center = self.X_center[:cluster_count, :]

    def generate_simulations(self):
        # Generate the simulations by reading the cluster center and generating the clusters
        self.read_cluster_center()
        self.generate_clusters()

# ----------------------------------------------------------------------------------------------------------------------
# Graph Generation and Processing
# ----------------------------------------------------------------------------------------------------------------------

def generate_simulations_with_alpha(alpha):
    sim_gen = SimulationGenerator(power_law_slope=alpha, high_mass_cutoff=100, n_sample=n_sample)
    sim_gen.generate_simulations()
    return sim_gen.X_data

def generate_simulations_with_alpha_and_high_mass_cutoff(args):
    alpha, high_mass_cutoff = args
    sim_gen = SimulationGenerator(power_law_slope=alpha, high_mass_cutoff=high_mass_cutoff, n_sample=n_sample)
    sim_gen.generate_simulations()
    return sim_gen.X_data

#------------------------------------------------------------
def make_graph(args):
    i, X_data, radius, subsample_factor = args

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

# ----------------------------------------------------------------------------------------------------------------------
# Grakel Graph Generation
def calculate_grakel_graph(args):
    i, adjacency_matrix, node_attributes = args
    return Graph(initialization_object=adjacency_matrix, node_labels=node_attributes)

# ----------------------------------------------------------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------------------------------------------------------

def main(radius=0.1, n_iter=2, subsample_factor=1, num_index=1):
    alpha_array = np.random.uniform(-1.5, -2.5, n_realization)
    high_mass_cutoff_array = (10.**np.random.uniform(0.5, 2.0, n_realization)).astype("int")

#------------------------------------------------------------
    # Generate all realizations in parallel with only alpha as input, high mass cutoff is fixed at 100
    # with multiprocessing.Pool(processes=num_cpu) as pool:
    #     X_data_total = np.array(list(tqdm(pool.imap(generate_simulations_with_alpha, alpha_array), total=len(alpha_array))))

    # Generate all realizations in parallel with alpha and high mass cutoff as input
    with multiprocessing.Pool(processes=16) as pool:
        X_data_total = np.array(list(tqdm(pool.imap(generate_simulations_with_alpha_and_high_mass_cutoff,
                                                    zip(alpha_array, high_mass_cutoff_array)), total=len(alpha_array))))
    print(X_data_total.shape)

#------------------------------------------------------------
    # add the observed data to X_data_total
    # load_data = np.load("X_APOGEE_resampled.npy")[:n_sample,:]
    # load_data = np.expand_dims(load_data, axis=0)
    # X_data_total = np.concatenate((X_data_total, load_data), axis=0)
    # print(X_data_total.shape)

#------------------------------------------------------------
    # Make the graph objects for all realisations in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        graph_list, degree_list = zip(*pool.map(make_graph, tqdm([(i, X_data_total[i], radius, subsample_factor) for i in range(X_data_total.shape[0])])))

    # Initialize the grakel graph kernel
    wl_kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram)
    args_list = [(i, graph_list[i], degree_list[i]) for i in range(len(graph_list))]

    # Calculate grakel graphs in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
       grakel_list = list(tqdm(pool.imap(calculate_grakel_graph, args_list), total=len(graph_list)))
    
    # Normalize kernel values
    kernel_values = wl_kernel.fit_transform(grakel_list)
    kernel_values = kernel_values / np.max(kernel_values)

    # Save with the radius value and number of iterations in the filename
    np.save(f"kernel_values_{radius}_{n_iter}_{subsample_factor}_num_index{num_index}_alpha_and_cutoff.npy", kernel_values)
    np.save(f"alpha_array_{radius}_{n_iter}_{subsample_factor}_num_index{num_index}_alpha_and_cutoff.npy", alpha_array)
    np.save(f"high_mass_cutoff_array_{radius}_{n_iter}_{subsample_factor}_num_index{num_index}_alpha_and_cutoff.npy", high_mass_cutoff_array)

# ----------------------------------------------------------------------------------------------------------------------
# Run the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the script with custom radius, number of iterations, and subsample factor.')
    parser.add_argument('--radius', type=float, default=0.1,
                        help='The radius to use for graph generation.')
    parser.add_argument('--n_iter', type=int, default=2,
                        help='The number of iterations for the WeisfeilerLehman algorithm.')
    parser.add_argument('--subsample_factor', type=int, default=1,
                        help='The subsample factor for graph generation.')
    parser.add_argument('--num_index', type=int, default=1,
                        help='The index of runs.')
    args = parser.parse_args()

    main(radius=args.radius, n_iter=args.n_iter, subsample_factor=args.subsample_factor, num_index=args.num_index)
