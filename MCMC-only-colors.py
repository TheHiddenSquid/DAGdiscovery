import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
from scipy import stats
import time

# Conclusion: MCMC cannot find the correct coloring even with correct edges given. This tells is that BIC is not a good heuristic

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


def update_DAG(edge_array, partition):
    tmp_edge_array = edge_array.copy()
    tmp_partition = partition.copy()

    no_nodes = np.shape(edge_array)[0]  # Number of current noded
    no_colors = len(tmp_partition)      # Number of possible colors


    move = "change_color"


    if move == "change_color":
        node = random.randrange(no_nodes)
        for i, part in enumerate(tmp_partition):
            if node in part:
                current_color = i
                break
        tmp_partition[current_color].remove(node)

        new_color = current_color
        while new_color == current_color:
            new_color = random.randrange(no_colors)
        tmp_partition[new_color].append(node)

        # Relative probability of jumping back
        q_quotient = 1


    return tmp_edge_array, tmp_partition, q_quotient



def score_DAG(samples, edge_array, partition):
    samples = np.transpose(samples)

    n = sum(len(x) for x in partition)


    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((n,n), dtype="float")
    for i in range(n):
        parents = get_parents(i, edge_array)
        ans = np.linalg.solve(np.matmul(samples[parents,:],np.transpose(samples[parents,:])), np.matmul(samples[parents,:],np.transpose(samples[i,:])))
        edges_ML[parents, i] = ans


    # Calculate ML-eval of the different color omegas
    omegas_for_color = [None] * len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue  
        tot = 0
        for node in part:
            parents = get_parents(node, edge_array)
            tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
        omegas_for_color[i] = tot / (n * len(part))

    omegas_ML = [None] * n
    for i, part in enumerate(partition):
        for node in part:
            omegas_ML[node] = omegas_for_color[i]


    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - np.log(n) * (1/n) * sum(len(get_parents(x, edge_array)) for x in part)
    bic = tot / 2


    return bic

def get_parents(node, edge_array):
    parents = []
    n = np.shape(edge_array)[0]
    for i in range(n):
        if edge_array[i, node] == 1:
            parents.append(i)
    return parents


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

    real_partition, real_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")
    samples = generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # Create plots
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # RUN MCMC
    initial_edge_array = real_edge_array
    initial_partition = generate_partition(no_nodes, no_nodes)

    current_edge_array = initial_edge_array.copy()
    current_partition = initial_partition.copy()


    best_edge_array = initial_edge_array.copy()
    best_partition = initial_partition.copy()
    best_bic = score_DAG(samples, initial_edge_array, initial_partition)
    best_iter = 0


    t = time.perf_counter()
    for i in range(MCMC_iterations):

        old_edge_array = current_edge_array.copy()
        old_partition = current_partition.copy()

        new_edge_array, new_partition, q_quotient = update_DAG(old_edge_array, old_partition)

        new_bic = score_DAG(samples, new_edge_array, new_partition)
        old_bic = score_DAG(samples, old_edge_array, old_partition)

        if random.random() <= (new_bic / old_bic) * q_quotient: 
            current_edge_array = new_edge_array
            current_partition = new_partition
        else:
            current_edge_array = old_edge_array
            current_partition = old_partition

        if new_bic > best_bic:
            best_edge_array = new_edge_array.copy()
            best_partition = new_partition.copy()
            best_bic = new_bic
            best_iter = i


    print("MCMC given the correct edges")
    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It tool {time.perf_counter()-t} seconds")
    print("Found coloring with BIC:", best_bic)
    print("Found on iteration:", best_iter)
    print("PHD to real coloring was:", calc_partition_distance(real_partition, best_partition))
    
    print("Correct DAG and correct coloring gives BIC:", score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax2)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(best_partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
