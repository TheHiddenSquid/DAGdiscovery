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
    num_nodes = 4
    num_edge_colors = 3
    num_node_colors = 5
    edge_prob = 0.4
    sample_size = 1000
    MCMC_iterations = 100_000   # 100_000 iters 3.45 sec on 4 nodes

    real_edge_partition, real_node_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_edge_colors, num_node_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)


    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_node_color_map(real_node_partition), edge_color=utils.generate_edge_color_map(G, real_edge_partition), with_labels=True)
    plt.title("Real DAG")


    
    plt.axes(ax2)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    

    t = time.perf_counter()
    A, PE, PN, bic, found_iter, num_fails = CausalMCMC(samples, MCMC_iterations, debug=True)

    print("Result of running causalMCMC")
    print(f"It took {time.perf_counter()-t} seconds")
    print(f"Ran for {MCMC_iterations} iterations")
    print(f"Failed to jump {num_fails} times")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", found_iter)
    print("MCMC: SHD to real DAG:", utils.calc_SHD(A, real_edge_array))
    print("MCMC: Node CHD to real DAG:", utils.calc_CHD(real_node_partition, PN))
    print("MCMC: Edge CSHD to real DAG:", utils.calc_CSHD(A, real_edge_array, PE, real_edge_partition))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_edge_partition, real_node_partition))

    
    G = nx.DiGraph(A)
    nx.draw_circular(G, node_color=utils.generate_node_color_map(PN), edge_color=utils.generate_edge_color_map(G, PE), with_labels=True)
    plt.title("Found DAG")
    plt.show()

    
if __name__ == "__main__":
    main()
