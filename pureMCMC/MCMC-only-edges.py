import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
from scipy import stats
import ges
import time
from MCMCfuncs import MCMC_iteration
from MCMCfuncs import score_DAG
from MCMCfuncs import get_sorted_edges

def generate_colored_DAG(no_nodes, no_colors, edge_probability):
    
    # Add edges and make sure it is a DAG
    G = nx.DiGraph()
    nodes = [*range(no_nodes)]
    random.shuffle(nodes)

    for node in nodes:
        G.add_node(node)
        others = [*range(no_nodes)]
        others.remove(node)
        random.shuffle(others)

        for other in others:
            if random.random() < edge_probability:
                if random.random() < 0.5:
                    G.add_edge(node, other)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(node, other)
                else:
                    G.add_edge(other, node)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(other, node)


    # Create partition for colors
    partition = generate_partition(no_nodes, no_colors)

    # Generate lambda matrix
    lambda_matrix = nx.adjacency_matrix(G).todense().astype("float64")
    for i in range(no_nodes):
        for j in range(no_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(-1,1)

    # Generate omega matrix
    choices = [random.random() for _ in range(no_colors)]
    omega_matrix = [None] * no_nodes
    for i, part in enumerate(partition):
        for node in part:
            omega_matrix[node] = choices[i]


    return partition, lambda_matrix, omega_matrix

def generate_sample(size, lambda_matrix, omega_matrix):
    no_nodes = len(omega_matrix)

    errors = np.zeros((no_nodes,size), dtype="float64")
    for i, omega in enumerate(omega_matrix):
        rv = stats.norm(scale = omega)
        errors[i,:] = rv.rvs(size=size)

    X = np.transpose(np.linalg.inv(np.identity(no_nodes) - lambda_matrix))
    
    sample = np.zeros((no_nodes,size), dtype="float64")
    for i in range(size):
        sample[:,i] = np.matmul(X, errors[:,i])
    
    return np.transpose(sample)


def generate_partition(no_nodes, no_colors):
    partition = [[] for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].append(node)
    return partition

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
    no_nodes = 6
    no_colors = 3
    edge_probability = 0.3
    sample_size = 1000
    MCMC_iterations = 10000
    start_with_GES_DAG = False

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

    else:
        # Fully random colored DAG
        _, initial_edge_array, _ = generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
    

    

    # RUN MCMC

    current_edge_array = initial_edge_array.copy()
    current_partition = real_partition.copy()
    current_sorted_edges = get_sorted_edges(current_edge_array)

    best_edge_array = current_edge_array.copy()
    best_partition = current_partition.copy()
    best_bic = score_DAG(samples, current_edge_array, current_partition)
    best_iter = 0


    t = time.perf_counter()
    for i in range(MCMC_iterations):

        current_edge_array, current_partition, current_bic, current_sorted_edges = MCMC_iteration(current_edge_array, current_partition, samples, current_sorted_edges, ["add_edge", "remove_edge"])

        if current_bic > best_bic:
            best_edge_array = current_edge_array.copy()
            best_partition = current_partition.copy()
            best_bic = current_bic
            best_iter = i


    print("MCMC given correct colors")
    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", best_bic)
    print("Found on iteration:", best_iter)
    print("SHD to real DAG was:", calc_SHD(best_edge_array, real_edge_array))
    
    if start_with_GES_DAG:
        print("A random GES DAG with correct coloring gives BIC:", score_DAG(samples, initial_edge_array, real_partition))
        print("Its SHD to real DAG was:", calc_SHD(initial_edge_array, real_edge_array))

    print("Correct DAG and correct coloring gives BIC:", score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax3)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(best_partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
