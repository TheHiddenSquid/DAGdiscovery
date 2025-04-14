import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(4)
    np.random.seed(4)
    num_nodes = 6
    num_edge_colors = 5
    num_node_colors = 5
    edge_prob = 0.4
    sample_size = 1000
    MCMC_iterations = 10_000

    real_edge_partition, real_node_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_edge_colors, num_node_colors, edge_prob)
    edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)

   

    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_node_color_map(real_node_partition), edge_color=utils.generate_edge_color_map(G, real_edge_partition), with_labels=True)
    plt.title("Real DAG")


    
    plt.axes(ax2)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    

    t = time.perf_counter()
    A, PE, PN, bic = CausalMCMC(samples, MCMC_iterations)
    
    G = nx.DiGraph(A)
    nx.draw_circular(G, node_color=utils.generate_node_color_map(PN), edge_color=utils.generate_edge_color_map(G, PE), with_labels=True)


    plt.title("Found DAG")

    plt.show()

    
if __name__ == "__main__":
    main()
