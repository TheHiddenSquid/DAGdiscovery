import copy
import random
import sys

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from MCMCfuncs import CausalMCMC, MCMC_iteration, score_DAG_full


def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 6
    no_colors = 3
    edge_prob = 0.6
    sample_size = 1000
    MCMC_iterations = 100_000
    no_chains = 3

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    


    # Fully random colored DAG
    initial_partition, initial_edge_array, _ = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
    initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
    

    

    # RUN MCMC

    chain_bics = []
    chain_cumsum = []


    best_edge_array = None
    best_partition = None
    best_bic = -np.infty
    best_iter = None


    for j in range(no_chains):
        initial_partition, initial_edge_array, _ = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
        CausalMCMC(samples, 0)
        initial_bic, initial_ML_data = score_DAG_full(initial_edge_array, initial_partition)

        current_edge_array = initial_edge_array.copy()
        current_partition = initial_partition.copy()
        current_bic = initial_bic.copy()
        current_ML_data = initial_ML_data.copy()

        bics = [initial_bic]
        cumsum = [initial_bic]

        for i in range(MCMC_iterations):
            
            move = random.choices([0, 1], k=1, weights=[0.4,0.6])[0]
            current_edge_array, current_partition, current_bic, current_ML_data, _ = MCMC_iteration(move, current_edge_array, current_partition, current_bic, current_ML_data)

            if current_bic > best_bic:
                best_edge_array = current_edge_array.copy()
                best_partition = copy.deepcopy(current_partition)
                best_bic = current_bic
                best_iter = j*MCMC_iterations+i
            
            bics.append(current_bic)
            cumsum.append(cumsum[-1]+current_bic)

        chain_bics.append(bics)
        chain_cumsum.append(cumsum)


    print(f"Ran MCMC for {no_chains * MCMC_iterations} iterations")
    print("Found DAG with BIC:", best_bic)
    print("Found on iteration:", best_iter)
    print("SHD to real DAG was:", utils.calc_SHD(best_edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))



    plt.axes(ax3)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(best_partition), with_labels=True)
    plt.title("MCMC")
    plt.show()


    plt.subplot(1,2,1)
    for i in range(no_chains):
        plt.plot(range(MCMC_iterations+1), chain_bics[i])

    plt.subplot(1,2,2)
    for i in range(no_chains):
        plt.plot(range(len(chain_cumsum[i])),[chain_cumsum[i][j]/(j+1) for j in range(len(chain_cumsum[i]))])

    plt.show()

    
if __name__ == "__main__":
    main()
