import numpy as np
import random
from scipy.optimize import linear_sum_assignment

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


def calc_CHD(partition1, partition2):
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


def is_DAG(A):
    numnodes = A.shape[0]
    P = np.linalg.matrix_power(A, numnodes-1)
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


