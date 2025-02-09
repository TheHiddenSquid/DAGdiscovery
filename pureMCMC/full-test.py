import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
import ges
import time
from MCMCfuncs import MCMC
from MCMCfuncs import score_DAG
from generateDAGs import generate_colored_DAG
from generateDAGs import generate_sample



def generate_color_map(partition):
    if len(partition) > 10:
        raise ValueError("Too many colors needed for color map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink"]
    length = sum([len(x) for x in partition])
    color_map = [None] * length

    for i, part in enumerate(partition):
        for node in part:
            color_map[node] = colors[i]

    return color_map


def calc_SHD(edge_array1, edge_array2):
    return np.sum(np.abs(edge_array1-edge_array2))

def calc_partition_distance(partition1, partition2):
    pa1 = partition1.copy()
    pa2 = partition2.copy()

    n = sum(len(x) for x in pa1)
    parts = max(len(pa1), len(pa2))

    pa1 += [[]]*(parts - len(pa1))
    pa2 += [[]]*(parts - len(pa2))

    cost_matrix = np.zeros((parts, parts), dtype="int")
    for i in range(parts):
        for j in range(parts):
            cost_matrix[i,j] = len(set(pa1[i]).intersection(set(pa2[j])))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return n - cost_matrix[row_ind, col_ind].sum()


def main():
    no_nodes = 5
    no_colors = 3
    edge_probability = 0.4
    sample_size = 1000
    MCMC_iterations = 10000

    real_partition, real_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    

    t = time.perf_counter()
    edge_array, partition, bic, iter = MCMC(samples, MCMC_iterations, move_list = None, start_from_GES = False)



    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", iter)
    print("SHD to real DAG was:", calc_SHD(edge_array, real_edge_array))
    print("The found DAG with correct coloring gives BIC:", score_DAG(samples, edge_array, real_partition))
    print("Correct DAG and correct coloring gives BIC:", score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
