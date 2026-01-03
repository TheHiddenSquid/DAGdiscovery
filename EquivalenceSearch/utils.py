import random

import networkx as nx
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


# DAG generation
def get_random_DAG(num_nodes, edge_prob = 0.5):
    # Generate erdős-rényi under-triangular matrix
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i > j:
                if edge_prob > random.random():
                    edge_array[i,j] = 1      
    
    # Shuffle nodes
    G = nx.DiGraph(edge_array)
    node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
    G_new = nx.relabel_nodes(G, node_mapping)
    edge_array = nx.adjacency_matrix(G_new, node_mapping).todense()

    return edge_array

def generate_colored_DAG(num_nodes, num_colors, edge_prob = 0.5):
    
    # Create partition for colors
    partition = generate_random_partition(num_nodes, num_colors)

    # Generate lambda matrix
    lambda_matrix = get_random_DAG(num_nodes, edge_prob)
    lambda_matrix = lambda_matrix.astype(np.float64)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(0.25,1) * random.randrange(-1,2,2)

    # Generate omega matrix
    choices = [random.uniform(0.2,2) for _ in range(num_colors)]
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


def score_DAG(data, A, PE, PN_flat):
    data = data.T
    num_samples = data.shape[1]
    num_nodes = data.shape[0]
    data_S = data @ data.T / num_samples

    # Calculate ML-eval of the different lambdas
    edges_ML_ungrouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = get_parents(i, A)
        edges_ML_ungrouped[parents, i] = np.linalg.lstsq(data[parents,:].T, data[i,:].T, rcond=None)[0]

    # Block the lambdas as averages
    edges_ML_grouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_ungrouped[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_grouped[edge] = block_lambda

    # Calculate ML-eval of the different color omegas
    omegas_ML_ungrouped = [None] * num_nodes
    for node in range(num_nodes):
        parents = get_parents(node, A)
        omegas_ML_ungrouped[node] = np.dot(x:=(data[node,:] - edges_ML_ungrouped[parents,node].T @ data[parents,:]), x) / num_samples

    # Block the omegas as averages
    omegas_ML_grouped = [None] * num_nodes
    for part in PN_flat:
        tot = 0
        for node in part:
            tot += omegas_ML_ungrouped[node]
        block_omega = tot/len(part)
        for node in part:
            omegas_ML_grouped[node] = block_omega
       
    # Calculate BIC 
    log_likelihood = (num_samples/2) * (-np.log(np.prod(omegas_ML_grouped)) + np.log(np.linalg.det(x:=(np.eye(num_nodes)-edges_ML_grouped))**2) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S))

    bic = (1/num_samples) * (log_likelihood - (np.log(num_samples)/2) * (len(PN_flat) + len(PE)))

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



# Get CPDAG given DAG and partition
def R1(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node2, node3] == 1 and A[node3, node2] == 1:
                        if A[node1, node3] == 0 and A[node3, node1] == 0:
                            A[node3, node2] = 0
                            return A, True
    return A, False
def R2(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node2, node3] == 1 and A[node3, node2] == 0:
                        if A[node1, node3] == 1 and A[node3, node1] == 1:
                            A[node3, node1] = 0
                            return A, True
    return A, False
def R3(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node3, node2] == 1 and A[node2, node3] == 0:
                        if A[node1, node3] == 0 and A[node3, node1] == 0:
                            for node4 in range(num_nodes):
                                if node4 == node1 or node4 == node2 or node4 == node3:
                                    continue
                                if A[node1, node4] == 1 and A[node4, node1] == 1:
                                    if A[node2, node4] == 1 and A[node4, node2] == 1:
                                        if A[node3, node4] == 1 and A[node4, node3] == 1:
                                            A[node2, node4] = 0
                                            return A, True
    return A, False
def R4(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 1:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node3, node2] == 1 and A[node2, node3] == 0:
                        if A[node1, node3] == 1 and A[node3, node1] == 1:
                            for node4 in range(num_nodes):
                                if node4 == node1 or node4 == node2 or node4 == node3:
                                    continue
                                if A[node1, node4] == 1 and A[node4, node1] == 1:
                                    if A[node2, node4] == 0 and A[node4, node2] == 0:
                                        if A[node3, node4] == 0 and A[node4, node3] == 1:
                                            A[node2, node1] = 0
                                            return A, True
    return A, False


def getCPDAG(A, P):
    # Skeleton
    num_nodes = A.shape[0]
    newA = np.array(A + A.T != 0, dtype=np.int64)
 
    # V-structures
    v_triplets = list(nx.dag.v_structures(nx.DiGraph(A)))
    for pa1, col, pa2 in v_triplets:
        newA[(col, pa1)] = 0
        newA[(col, pa2)] = 0

    # Repeat R1, R2, R3
    while True:
        newA, did_change1 = R1(newA)
        newA, did_change2 = R2(newA)
        newA, did_change3 = R3(newA)
        if not (did_change1 or did_change2 or did_change3):
            break
    
    # Add color info
    nodes_to_look_at = []
    for part in P:
        if len(part) >= 2:
            nodes_to_look_at += part
    
    for node in nodes_to_look_at:
        for other in range(num_nodes):
            if A[node, other] == 1:
                newA[other, node] = 0
            if A[other, node] == 1:
                newA[node, other] = 0  
    
    # Repeat R1, R2, R3
    while True:
        newA, did_change1 = R1(newA)
        newA, did_change2 = R2(newA)
        if not (did_change1 or did_change2):
            break

    return newA

def enum_MEC(A):
    cpdag = getCPDAG(A, [[i] for i in range(A.shape[0])])
    return class_enum(cpdag)

def class_enum(A):
    # Maximal orientation
    k = np.shape(A)[0]
    H = A.copy()
    
    while True:
        H, did_change1 = R1(H)
        H, did_change2 = R2(H)
        H, did_change3 = R3(H)
        H, did_change4 = R4(H)
        if not (did_change1 or did_change2 or did_change3 or did_change4):
            break


    undir = []
    for i in range(k):
        for j in range(i):
            if H[i,j] == 1 and H[j,i] == 1:
                undir.append((i,j))
    
    if len(undir) == 0:
        return [H]
    else:
        edge = undir[0]

        H[edge] = 0
        sublist1 = class_enum(H)
        H[edge] = 1

        H[tuple(reversed(edge))] = 0
        sublist2 = class_enum(H)
        H[tuple(reversed(edge))] = 1

        return sublist1 + sublist2
            


# Other functions

def generate_color_map(P):
    if len(P) > 12:
        raise ValueError("Too many colors needed for color-map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink", "cyan", "gray"]
    length = sum([len(x) for x in P])
    color_map = [None] * length

    for i, part in enumerate(P):
        for node in part:
            color_map[node] = colors[i]

    return color_map

def hash_DAG(edge_array, partition):
    return (edge_array.tobytes(), tuple(tuple(x) for x in sorted_partition(partition)))

