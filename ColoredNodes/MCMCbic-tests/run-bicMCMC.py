import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 4
    no_colors = 3
    edge_prob = 0.9
    sample_size = 1000
    MCMC_iterations = 100_000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)

    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    t = time.perf_counter()
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    print("sample-gen+GES", time.perf_counter()-t)

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    

    t = time.perf_counter()
    edge_array, partition, bic, found_iter, _ = CausalMCMC(samples, MCMC_iterations, debug=True)


    print("Result of running causalMCMC")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", found_iter)
    print("MCMC: SHD to real DAG:", utils.calc_SHD(edge_array, real_edge_array))
    print("GES: SHD to real DAG:", utils.calc_SHD(GES_edge_array, real_edge_array))
    print("The found DAG with correct coloring gives BIC:", utils.score_DAG(samples, edge_array, real_partition))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))



    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
