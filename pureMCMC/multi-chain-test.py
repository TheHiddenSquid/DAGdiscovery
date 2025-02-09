import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
import ges
from MCMCfuncs import MCMC_iteration
from MCMCfuncs import score_DAG
from MCMCfuncs import get_sorted_edges
from generateDAGs import generate_colored_DAG
from generateDAGs import generate_sample
from generateDAGs import generate_partition


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
    random.seed(1)
    np.random.seed(1)
    no_nodes = 6
    no_colors = 3
    edge_probability = 0.3
    sample_size = 1000
    MCMC_iterations = 100_000
    start_with_GES_DAG = True
    no_chains = 3

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
    

    # MCMC setup
    if start_with_GES_DAG:

        # Take an initial DAG from the given GES CPDAG
        initial_edge_array =  GES_edge_array.copy()
        double = []
        for i in range(no_nodes):
            for j in range(i+1, no_nodes):
                if initial_edge_array[i,j] == 1 and initial_edge_array[j,i] == 1:
                    double.append((i,j))
                    initial_edge_array[i,j] = 0
                    initial_edge_array[j,i] = 0

        for edge in double:
            new_edges = initial_edge_array.copy()
            new_edges[edge[0], edge[1]] = 1
            G = nx.DiGraph(new_edges)
            if nx.is_directed_acyclic_graph(G):
                initial_edge_array = new_edges
                continue

            new_edges = initial_edge_array.copy()
            new_edges[edge[1], edge[0]] = 1
            G = nx.DiGraph(new_edges)
            if nx.is_directed_acyclic_graph(G):
                initial_edge_array = new_edges
                continue

            raise ValueError("Could not create DAG")

        # Random initial coloring guess
        initial_partition = generate_partition(no_nodes, no_nodes)

    else:
        # Fully random colored DAG
        initial_partition, initial_edge_array, _ = generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
    

    

    # RUN MCMC



    chain_bics = []
    chain_cumsum = []


    best_edge_array = None
    best_partition = None
    best_bic = -np.infty
    best_iter = None


    for j in range(no_chains):
        initial_partition, initial_edge_array, _ = generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
        initial_bic = score_DAG(samples, initial_edge_array, initial_partition)

        current_edge_array = initial_edge_array.copy()
        current_partition = initial_partition.copy()
        current_bic = initial_bic.copy()
        current_sorted_edges = get_sorted_edges(current_edge_array)

        bics = [initial_bic[0]]
        cumsum = [initial_bic[0]]

        for i in range(MCMC_iterations):

            current_edge_array, current_partition, current_bic, current_sorted_edges = MCMC_iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges)

            if current_bic[0] > best_bic:
                best_edge_array = current_edge_array.copy()
                best_partition = current_partition.copy()
                best_bic = current_bic[0]
                best_iter = j*MCMC_iterations+i
            
            bics.append(current_bic[0])
            cumsum.append(cumsum[-1]+current_bic[0])

        chain_bics.append(bics)
        chain_cumsum.append(cumsum)


    print(f"Ran MCMC for {no_chains * MCMC_iterations} iterations")
    print("Found DAG with BIC:", best_bic)
    print("Found on iteration:", best_iter)
    print("SHD to real DAG was:", calc_SHD(best_edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", score_DAG(samples, real_edge_array, real_partition)[0])



    plt.axes(ax3)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(best_partition), with_labels=True)
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
