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

def get_sorted_edges(edge_array):
  
    tmp_edge_array = edge_array.copy()
    n = np.shape(tmp_edge_array)[0]

    edges_in_DAG = []
    edges_giving_DAGs = []
    edges_not_giving_DAGs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if tmp_edge_array[i, j] == 1:
                edges_in_DAG.append((i,j))
                continue

            tmp_edge_array[i, j] = 1
            if is_DAG(tmp_edge_array):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def score_DAG(data, A, P):
    my_data = data.T
    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval via least sqares
    omegas_ML = [0] * num_nodes
    for node in range(num_nodes):
        parents = get_parents(node, A)
        a = my_data[parents,:]
        b = my_data[node,:]
        beta = np.linalg.solve(a @ a.T, a @ b)
        x = b - a.T @ beta
        ss_res = np.dot(x,x)
        omegas_ML[node] = ss_res / num_samples


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (np.log(block_omega) + 1)
    
    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))
    
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

