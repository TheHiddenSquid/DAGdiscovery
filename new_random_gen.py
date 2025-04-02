import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def is_DAG(A):
    numnodes = A.shape[0]
    if numnodes < 8:
        P = A
    else:
        P = A.astype(np.float32)
    power = 1
    while power < numnodes:
        P = P @ P
        power *= 2
    if np.argmax(P) != 0:
        return False
    return not P[0,0]

def get_random_DAG1(num_nodes, sparse = False):
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for _ in range(100 * num_nodes**2):
        edge = (random.randrange(num_nodes), random.randrange(num_nodes))
        if edge_array[edge] == 1:
            edge_array[edge] = 0
        else:
            if sparse and np.sum(edge_array) >= 1.5*num_nodes:
                continue
            tmp = edge_array.copy()
            tmp[edge] = 1
            if is_DAG(tmp):
                edge_array = tmp
    
    return edge_array


def get_random_DAG2(num_nodes, prob):
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i+j+1 < num_nodes:
                if prob > random.random():
                    edge_array[i,j] = 1
    edge_array = np.flipud(edge_array)            
    return edge_array

def main():

    A = get_random_DAG1(5, sparse=True)
    plt.subplot(1,2,1)
    G = nx.DiGraph(A)
    nx.draw_circular(G, with_labels=True)

    A = get_random_DAG2(5, 0.2)
    print(A)
    plt.subplot(1,2,2)
    G = nx.DiGraph(A)
    nx.draw_circular(G, with_labels=True)
    plt.show()

if __name__ == "__main__":
    main()