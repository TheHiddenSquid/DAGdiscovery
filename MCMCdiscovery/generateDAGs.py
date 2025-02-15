import numpy as np
import networkx as nx
import random
from scipy import stats
import utils


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
    partition = utils.generate_random_partition(no_nodes, no_colors)

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

