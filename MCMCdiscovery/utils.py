import numpy as np
import random
from scipy.optimize import linear_sum_assignment

import networkx as nx
from scipy import stats


# DAG generation

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
    partition = generate_random_partition(no_nodes, no_colors)

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



# Graph functions

def is_DAG(A):
    numnodes = A.shape[0]
    P = np.linalg.matrix_power(A, numnodes)
    if np.argmax(P) != 0:
        return False
    return not P[0,0]

def get_parents(node, A):
    parents = []
    n = np.shape(A)[0]
    for i in range(n):
        if A[i, node] == 1:
            parents.append(i)
    return parents



# Partition generation and manipulation

def generate_random_partition(no_nodes, no_colors):
    partition = [[] for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].append(node)
    return partition

def sorted_partition(partition):
    num_nodes = sum(len(x) for x in partition)
    dones = [False] * num_nodes
    sorted_partition = []
    for i in range(num_nodes):
        if dones[i] == True:
            continue
        for j, part in enumerate(partition):
            part.sort()
            if i in part:
                sorted_partition.append(part)
                for k in part:
                    dones[k] = True
    return sorted_partition



# Metric calculation functions

def calc_SHD(A, B):
    return np.sum(np.abs(A-B))

def calc_CHD(P1, P2):
    tmp_p1 = P1.copy()
    tmp_p2 = P2.copy()

    num_nodes = sum(len(x) for x in tmp_p1)
    num_colors = max(len(tmp_p1), len(tmp_p2))

    tmp_p1 += [[]]*(num_colors - len(tmp_p1))
    tmp_p2 += [[]]*(num_colors - len(tmp_p2))

    cost_matrix = np.zeros((num_colors, num_colors), dtype="int")
    for i in range(num_colors):
        for j in range(num_colors):
            cost_matrix[i,j] = len(set(tmp_p1[i]).intersection(set(tmp_p2[j])))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return num_nodes - cost_matrix[row_ind, col_ind].sum()



# Other functions

def generate_color_map(P):
    if len(P) > 10:
        raise ValueError("Too many colors needed for color-map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink"]
    length = sum([len(x) for x in P])
    color_map = [None] * length

    for i, part in enumerate(P):
        for node in part:
            color_map[node] = colors[i]

    return color_map



