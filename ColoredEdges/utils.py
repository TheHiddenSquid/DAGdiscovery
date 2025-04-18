import copy
import random

import networkx as nx
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


# DAG generation
def generate_colored_DAG(num_nodes, num_edge_colors, num_node_colors, p = 0.5):
    # Generate erdős-rényi under-triangular matrix
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i > j:
                if p > random.random():
                    edge_array[i,j] = 1

    # Shuffle nodes
    G = nx.DiGraph(edge_array)
    node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
    G_new = nx.relabel_nodes(G, node_mapping)
    edge_array = nx.adjacency_matrix(G_new, node_mapping).todense()
    edges = list(G_new.edges())


    # Partition edges into color
    random.shuffle(edges)
    edge_partition = [[] for _ in range(num_edge_colors)]
    for edge in edges:
        color = random.randrange(num_edge_colors)
        edge_partition[color].append(edge)
    edge_partition = [x for x in edge_partition if len(x)>0]


    # Group nodes that need to be together
    super_nodes = []
    for part in edge_partition:
        super_node = set()
        for edge in part:
            super_node.add(edge[1])
        super_nodes.append(super_node)

    real_supnodes = []
    while True:
        if len(super_nodes) == 0:
            break
        edits = 0
        last = super_nodes.pop()

        for part in super_nodes:
            if len(last.intersection(part)) > 0:
                super_nodes.remove(part)
                last = last.union(part)
                edits += 1
        if edits > 0:
            super_nodes.append(last)
        else:
            real_supnodes.append(last)
    super_nodes = [list(x) for x in real_supnodes]


    # Add potental "solo" supnodes that are not forced by edge colors
    used = [False]*num_nodes
    for supnode in super_nodes:
        for node in supnode:
            used[node] = True
    needed = [i for i, x in enumerate(used) if x == False]
    for node in needed:
        super_nodes.append([node])


    # Partition nodes
    random.shuffle(super_nodes)
    node_partition = [[] for _ in range(num_node_colors)]
    for node in super_nodes:
        color = random.randrange(num_node_colors)
        node_partition[color].append(node)
    node_partition = [x for x in node_partition if len(x)>0]


    # Generate lambda matrix
    choices = [random.uniform(0.25,1) * random.randrange(-1,2,2) for _ in range(num_edge_colors)]
    lambda_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, part in enumerate(edge_partition):
        for edge in part:
            lambda_matrix[edge] = choices[i]


    # Generate omega matrix
    choices = [random.uniform(0.2,2) for _ in range(num_node_colors)]
    omega_matrix = [None] * num_nodes
    for i, part in enumerate(node_partition):
        for super_node in part:
            for node in super_node:
                omega_matrix[node] = choices[i]


    #optional: flatten node partition
    node_partition = [sum(x,[]) for x in node_partition]
    
    return edge_partition, node_partition, lambda_matrix, omega_matrix

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

def get_supnodes(PE, num_nodes):
    super_nodes = []

    for part in PE:
        super_node = set()
        for edge in part:
            super_node.add(edge[1])
        super_nodes.append(super_node)

    real_supnodes = []
    while True:
        if len(super_nodes) == 0:
            break
        edits = 0
        last = super_nodes.pop()

        for part in super_nodes:
            if len(last.intersection(part)) > 0:
                super_nodes.remove(part)
                last = last.union(part)
                edits += 1
        if edits > 0:
            super_nodes.append(last)
        else:
            real_supnodes.append(last)
    super_nodes = [list(x) for x in real_supnodes]

    # Add potental "solo" supnodes that are not forced by edge colors
    used = [False]*num_nodes
    for supnode in super_nodes:
        for node in supnode:
            used[node] = True
    needed = [i for i, x in enumerate(used) if x == False]
    for node in needed:
        super_nodes.append([node])

    return super_nodes


# Partition generation and manipulation

def generate_random_node_partition(no_nodes, no_colors):
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

def generate_node_color_map(PN):
    if len(PN) > 12:
        raise ValueError("Too many colors needed for color-map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink", "cyan", "gray"]

    PN = sorted_partition(PN)
    length = sum(len(x) for x in PN)
    color_map = [None] * length
    

    for i, part in enumerate(PN):
        for node in part:
            color_map[node] = colors[i]

    return color_map

def generate_edge_color_map(G, PE):
    if len(PE) > 12:
        raise ValueError("Too many colors needed for color-map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink", "cyan", "gray"]
    num_edges = sum([len(x) for x in PE])
    color_map = [None] * num_edges

    for i, edge in enumerate(G.edges()):
        for j, part in enumerate(PE):
            if edge in part:
                color_map[i] = colors[j]

    return color_map
