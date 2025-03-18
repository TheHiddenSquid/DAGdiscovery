import random

import networkx as nx
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


# DAG generation
def get_random_DAG(num_nodes, sparse = False):
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

def generate_colored_DAG(num_nodes, num_colors, sparse = False):
    
    # Create partition for colors
    partition = generate_random_partition(num_nodes, num_colors)

    # Generate lambda matrix
    lambda_matrix = get_random_DAG(num_nodes, sparse)
    lambda_matrix = lambda_matrix.astype(np.float64)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(-1,1)

    # Generate omega matrix
    choices = [random.random() for _ in range(num_colors)]
    omega_matrix = [None] * num_nodes
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

def get_parents(node, A):
    parents = []
    n = np.shape(A)[0]
    for i in range(n):
        if A[i, node] == 1:
            parents.append(i)
    return parents


def score_DAG(samples, edge_array, partition):
    samples = samples.T
    
    num_nodes = samples.shape[0]
    num_samples = samples.shape[1]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = get_parents(i, edge_array)
        ans = np.linalg.lstsq(samples[parents,:].T, samples[i,:].T, rcond=None)[0]
        edges_ML[parents, i] = ans

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * len(partition)
    bic = 0

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = get_parents(node, edge_array)
            tot += np.dot(x:=(samples[node,:] - edges_ML[parents,node].T @ samples[parents,:]), x)
        omegas_ML[i] = tot / (num_samples * len(part))


        # Calculate BIC
        bic  += -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = bic/2 - BIC_constant * (sum(1 for part in partition if len(part)>0) + np.sum(edge_array))
 

    return bic


# Partition generation and manipulation

def generate_random_partition(no_nodes, no_colors):
    partition = [set() for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].add(node)
    partition = [set(x) for x in sorted_partition(partition)]
    for _ in range(no_colors-len(partition)):
        partition.append(set())
    return partition

def sorted_partition(partition):
    num_nodes = sum(len(x) for x in partition)
    dones = [False] * num_nodes
    sorted_partition = []
    for i in range(num_nodes):
        if dones[i] == True:
            continue
        for part in partition:
            if i in part:
                sorted_partition.append(sorted(part))
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



