import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from scipy import stats
import ges

def partition_nodes_by_color(no_nodes, no_colors):
    partition = [[] for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].append(node)
    return partition

def generate_color_map(partition):
    if len(partition) > 6:
        raise ValueError("Too many colors needed for color map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown"]
    length = sum([len(x) for x in partition])
    color_map = [None] * length

    for i, part in enumerate(partition):
        for node in part:
            color_map[node] = colors[i]

    return color_map

def generate_lambdas(G):
    lambda_matrix = nx.adjacency_matrix(G).todense()
    lambda_matrix = lambda_matrix.astype("float64") 

    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = round(random.uniform(-1,1),3)
    
    return lambda_matrix

def generate_omegas(partition):
    length = sum([len(x) for x in partition])
    choices = [random.random() for _ in range(len(partition))]

    omegas = [None] * length
    for i, part in enumerate(partition):
        for node in part:
            omegas[node] = choices[i]

    return omegas

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

def main():
    no_nodes = 6
    no_colors = 3
    edge_probability = 0.8

    while True:
        edges = []
        for i in range(no_nodes):
            for j in range(i+1, no_nodes):
                if random.random() < edge_probability:
                    if random.random() < 0.5:
                        edges.append((i,j))
                    else:
                        edges.append((j,i))

        G = nx.DiGraph()

        for node in range(no_nodes):
            G.add_node(node)

        for i,j in edges:
            G.add_edge(i,j)

        if nx.is_directed_acyclic_graph(G):
            break

    


    
    partition = partition_nodes_by_color(no_nodes=no_nodes, no_colors=no_colors)
    

    lambda_matrix = generate_lambdas(G)
    omega_matrix = generate_omegas(partition)


    samples = generate_sample(100000, lambda_matrix, omega_matrix)


    # Data generating graph
    plt.subplot(1,2,1)
    
    color_map = generate_color_map(partition)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    plt.title("Data generating DAG")

    # GES estimate of graph
    plt.subplot(1,2,2)
    res = ges.fit_bic(data=samples)

    G = nx.DiGraph(res[0])
    nx.draw_circular(G, with_labels=True)
    plt.title("GES resulting CPDAG")

    print("score:", res[1])
    plt.show()

    
    

if __name__ == "__main__":
    main()
