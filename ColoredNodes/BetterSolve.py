import copy
import functools
import math
import random

import numpy as np
import utils
from numba import njit

# Main functions

def CausalGreedySearch(samples, num_waves = 5):

    #Clear cache for new run of algorithm
    calc_lstsq_S.cache_clear()
    
    # Setup global variables
    global S
    global num_edges
    global num_nodes
    global BIC_constant

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    S = (1/num_samples) * samples.T @ samples
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes))
    num_edges = np.count_nonzero(best_A)
    best_bic, _ = score_DAG_full(best_A, best_P)

    edge_probs = list(np.linspace(0,1,num_waves))
    num_colors = [int(x) for x in np.linspace(1,num_nodes,num_waves)]

    
    # Perform iterations
    for i in range(num_waves):
        _, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_colors[i], edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)

        num_edges = np.count_nonzero(A)
        bic, ML_data = score_DAG_full(A, P)
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            A, bic, ML_data, sorted_edges, done = Greedyiteration(A, bic, ML_data, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_bic = bic

    # Ectract optimal partition
    best_P = None
    CPDAG_A = utils.getCPDAG(best_A, best_P)
    return CPDAG_A, best_P, best_bic
    
def Greedyiteration(A, bic, ML_data, sorted_edges):
    global num_edges

    best_move = None
    best_A = None
    best_P = None
    best_bic = bic
    best_ML_data = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges



    # Check all potential edge adds
    for edge in edges_giving_DAGs:
        A[edge] = 1
        num_edges += 1
        potential_bic, potential_ML_data = score_DAG_edge_edit(A, ML_data, edge)

        if potential_bic > best_bic:
            best_A = A.copy()
            best_bic = potential_bic
            best_ML_data = potential_ML_data
            best_move = "add_edge"
            best_saved_edge = edge
    
        A[edge] = 0
        num_edges -= 1


    # Check all potential edge removals
    for edge in edges_in_DAG:
        A[edge] = 0
        num_edges -= 1
        
        potential_bic, potential_ML_data = score_DAG_edge_edit(A, ML_data, edge)

        if potential_bic > best_bic:
            best_A = A.copy()
            best_bic = potential_bic
            best_ML_data = potential_ML_data
            best_move = "remove_edge"
            best_saved_edge = edge
        
        A[edge] = 1
        num_edges += 1



    # Do the best possible jump

    if best_ML_data is None:
        return A, bic, ML_data, sorted_edges, True
    
    else:
        new_bic = best_bic 
        new_ML_data = best_ML_data
        

        if best_move == "add_edge":
            new_A = best_A
            num_edges += 1
            new_sorted_edges = update_sorted_edges_ADD(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "remove_edge":
            new_A = best_A
            num_edges -= 1
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)

        
    return new_A, new_bic, new_ML_data, new_sorted_edges, False



# For edge lookups

def get_sorted_edges(A):
  
    tmp_edge_array = A.copy()
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
            if utils.is_DAG(tmp_edge_array):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_sorted_edges_REMOVE(A, edges_in, addable_edges, not_addable_edges, removed_edge):
    
    tmp_edge_array = A.copy()

    edges_in_DAG = edges_in.copy()
    edges_in_DAG.remove(removed_edge)

    edges_giving_DAGs = addable_edges.copy() + [removed_edge]
    edges_not_giving_DAGs = []

    for edge in not_addable_edges:
        tmp_edge_array[edge] = 1
        if utils.is_DAG(tmp_edge_array):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_sorted_edges_ADD(A, edges_in, addable_edges, not_addable_edges, added_edge):

    tmp_edge_array = A.copy()

    edges_in_DAG = edges_in.copy() + [added_edge]
    edges_giving_DAGs = []
    edges_not_giving_DAGs = not_addable_edges.copy()

    for edge in addable_edges:
        if edge == added_edge:
            continue

        tmp_edge_array[edge] = 1
        if utils.is_DAG(tmp_edge_array):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]



# For DAG heuristic
def score_DAG_full(A):

    # Calculate ML-eval via least sqares
    ss_res = [0] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        ss_res[node] = calc_lstsq_S(node, tuple(parents))


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    block_sums = [0] * num_nodes

    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += ss_res[node]
        block_sums[i] = tot
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (math.log(block_omega) + 1)
    
    # Calculate full BIC
    bic_decomp_sum = sum(bic_decomp)
    nc = sum(1 for x in P if len(x)!=0)
    bic = bic_decomp_sum/2 - BIC_constant * (num_edges + nc)
    
    return bic, [ss_res, block_sums, bic_decomp, bic_decomp_sum]


def score_DAG_edge_edit(A, ML_data, changed_edge):

    # Get old ML-eval
    ss_res, block_sums, bic_decomp, bic_decomp_sum = ML_data
    ss_res = ss_res.copy()
    block_sums = block_sums.copy()
    bic_decomp = bic_decomp.copy()

    
    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    old_node_ss_res = ss_res[active_node]
    new_node_ss_res = calc_lstsq_S(active_node, tuple(parents))
    ss_res[active_node] = new_node_ss_res


    # Update decomposed BIC
    for i, block in enumerate(P):
        if active_node in block:
            active_block_id = i

    bic_decomp_sum -= bic_decomp[active_block_id]
    active_block = P[active_block_id]
    block_sums[active_block_id] -= old_node_ss_res
    block_sums[active_block_id] += new_node_ss_res
    block_omega = block_sums[active_block_id] / len(active_block)
    bic_decomp[active_block_id] = -len(active_block) * (math.log(block_omega) + 1)
    bic_decomp_sum += bic_decomp[active_block_id]
  

    # Calculate full BIC
    nc = sum(1 for x in P if len(x)!=0)
    bic = bic_decomp_sum/2 - BIC_constant * (num_edges + nc)

    return bic, [ss_res, block_sums, bic_decomp, bic_decomp_sum]


@functools.cache
def calc_lstsq_S(node, parents):
    return calc_lstsq_S_numba(node, parents, S)

@njit(cache=True)
def calc_lstsq_S_numba(node, parents, G):

    k = len(parents)
    if k == 0:
        return G[node, node]
    
    A = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            A[i, j] = G[parents[i], parents[j]]
            
    b = np.zeros(k)
    for i in range(k):
        b[i] = G[parents[i], node]
        
    beta = np.linalg.solve(A, b)
    
    explained = 0.0
    for i in range(k):
        explained += beta[i] * b[i]
        
    return G[node, node] - explained










def get_optimal_partition(A, X):
    num_nodes = X.shape[1]
    num_samples = X.shape[0]
    S = (1/num_samples) * X.T @ X
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ss_res via least sqares
    ss_res = [[i, 0] for i in range(num_nodes)]
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        ss_res[node][1] = calc_lstsq_S_old(S, node, tuple(parents))

    ss_res.sort(key=lambda x: x[1])
    r = [x[1] for x in ss_res]


    # Solve optimal partition by BIC
    memoization = [[] for _ in range(len(r)+1)] 

    for i in range(len(r)):
        active_residuals = r[:i+1]
        options = []
        for j in range(len(active_residuals)):
            last_pivot = j
            earlier_pivots = memoization[j]
            pivots = [*earlier_pivots, last_pivot]
            val = obj(active_residuals, pivots, BIC_constant)
            options.append([pivots, val])

        best = min(options, key=lambda x: x[1])
        memoization[i+1] = best[0]


    partition = []
    best_pivots = best[0]
    for i in range(len(best_pivots)-1):
        partition.append(set(ss_res[i][0] for i in range(best_pivots[i], best_pivots[i+1])))
    partition.append(set(ss_res[i][0] for i in range(best_pivots[-1], num_nodes)))

    partition = utils.sorted_partition(partition)
    return partition
   

def calc_lstsq_S_old(S, node, parents):
    g_nn = S[node, node]
    
    if len(parents) == 0:
        return g_nn
    
    g_pa = S[parents, node]
    G_pa = S[parents, :][:, parents]
    beta = np.linalg.solve(G_pa, g_pa)
    ss_res = g_nn - np.dot(beta, g_pa)
    
    return ss_res


def obj(r, pivot_indices, BIC_constant):
    tot = 0
    for i in range(len(pivot_indices)-1):
        start = pivot_indices[i]
        end = pivot_indices[i+1]
        r_mean = np.mean(r[start:end])
        size = end - start
        tot += size * (math.log(r_mean) + 1)

    start = pivot_indices[-1]
    end = len(r)
    r_mean = np.mean(r[start:end])
    size = end - start
    tot += size * (math.log(r_mean) + 1)
    
    return tot + BIC_constant*len(pivot_indices)


def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 10
    no_colors = 2
    edge_prob = 0.6
    sample_size = 100

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)
    print(real_partition)

    sample = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

    found_P = get_optimal_partition(real_edge_array, sample)
    print(found_P)

        
    



if __name__ == "__main__":
    main()

