import functools
import math
import random
import time
from collections.abc import Iterable

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utils
from numba import njit

"""
Greedy solver that uses the optimal partition for every graph and thus only needs to search through the space of 
DAGs during the optimization. The drawback is O(p^2) extra computation each iteration.
"""

# Main functions

def CausalGreedyDP(samples, num_starts = 5, moves = None):

    #Clear cache for new run of algorithm
    calc_lstsq_S.cache_clear()
    
    # Setup global variables
    global S
    global num_edges
    global num_nodes
    global BIC_constant
    global used_moves

    if moves == None:
        used_moves = ["add_edge", "remove_edge", "flip_edge"]
    else:
        used_moves = moves

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    S = (1/num_samples) * samples.T @ samples
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes))
    num_edges = np.count_nonzero(best_A)
    best_bic, best_ss_res, _ = score_DAG_full(best_A)

    edge_probs = list(np.linspace(0,1,num_starts))
    num_colors = [int(x) for x in np.linspace(1,num_nodes,num_starts)]

    
    # Perform iterations
    for i in range(num_starts):
        _, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_colors[i], edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)

        num_edges = np.count_nonzero(A)
        bic, ss_res, _ = score_DAG_full(A)
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            A, bic, ss_res, sorted_edges, done = iteration(A, bic, ss_res, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_ss_res = ss_res.copy()
                best_bic = bic

    # Ectract optimal partition
    best_P, _ = get_optimal_partition(best_ss_res)
    CPDAG_A = utils.getCPDAG(best_A, best_P)
    return CPDAG_A, best_P, best_bic
    
def iteration(A, bic, ss_res, sorted_edges):
    global num_edges

    best_move = None
    best_A = None
    best_bic = bic
    best_ss_res = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges



    # Check all potential edge adds
    if "add_edge" in used_moves:
        for edge in edges_giving_DAGs:
            A[edge] = 1
            num_edges += 1
            potential_bic, potential_ss_res = score_DAG_edge_edit(A, ss_res, [edge])

            if potential_bic > best_bic:
                best_A = A.copy()
                best_bic = potential_bic
                best_ss_res = potential_ss_res
                best_move = "add_edge"
                best_saved_edge = edge
        
            A[edge] = 0
            num_edges -= 1


    # Check all potential edge removals
    if "remove_edge" in used_moves:
        for edge in edges_in_DAG:
            A[edge] = 0
            num_edges -= 1
            
            potential_bic, potential_ss_res = score_DAG_edge_edit(A, ss_res, [edge])

            if potential_bic > best_bic:
                best_A = A.copy()
                best_bic = potential_bic
                best_ss_res = potential_ss_res
                best_move = "remove_edge"
                best_saved_edge = edge
            
            A[edge] = 1
            num_edges += 1


    # Check all potential edge flips
    if "flip_edge" in used_moves:
        for edge in edges_in_DAG:
            rev = (edge[1], edge[0])
            A[edge] = 0
            A[rev] = 1
        
            if utils.is_DAG(A):
                potential_bic, potential_ss_res = score_DAG_edge_edit(A, ss_res, [edge, rev])

                if potential_bic > best_bic:
                    best_A = A.copy()
                    best_bic = potential_bic
                    best_ss_res = potential_ss_res
                    best_move = "flip_edge"
                    best_saved_edge = [edge, rev]
                        
            A[edge] = 1
            A[rev] = 0


    # Do the best possible jump

    if best_ss_res is None:
        return A, bic, ss_res, sorted_edges, True
    
    else:
        new_bic = best_bic 
        new_ss_res = best_ss_res
        new_A = best_A

        if best_move == "add_edge":
            num_edges += 1
            new_sorted_edges = update_sorted_edges_ADD(new_A, *sorted_edges, best_saved_edge)
        elif best_move == "remove_edge":
            num_edges -= 1
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, *sorted_edges, best_saved_edge)
        elif best_move == "flip_edge":
            removed_edge, added_edge = best_saved_edge
            intermediate_A = new_A.copy()
            intermediate_A[added_edge] = 0
            intermediate_edges = update_sorted_edges_REMOVE(intermediate_A, *sorted_edges, removed_edge)
            new_sorted_edges = update_sorted_edges_ADD(new_A, *intermediate_edges, added_edge)
            
        
    return new_A, new_bic, new_ss_res, new_sorted_edges, False



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
    ss_res = get_ss_res(A, range(num_nodes))
    P, score = get_optimal_partition(ss_res)
    bic = - score - BIC_constant * num_edges
    
    return bic, ss_res, P

def score_DAG_edge_edit(A, ss_res, changed_edges):
    ss_res = ss_res.copy()
    for edge in changed_edges:
        _, active_node = edge
        ss_res[active_node] = get_ss_res(A, active_node)

    _, score = get_optimal_partition(ss_res)
    bic = - score - BIC_constant * num_edges
    return bic, ss_res


def get_ss_res(A, nodes):
    if not isinstance(nodes, Iterable):
        node = nodes
        parents = utils.get_parents(node, A)
        return calc_lstsq_S(node, tuple(parents))   
    else:
        ss_res = [0] * len(nodes)
        for i, node in enumerate(nodes):
            parents = utils.get_parents(node, A)
            ss_res[i] = calc_lstsq_S(node, tuple(parents))
        return ss_res

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



# Solve dynamic programming to find optimal partition
def get_optimal_partition(ss_res):
    residuals = np.asarray(ss_res, dtype=np.float64)

    if np.any(residuals <= 0):
        raise ValueError("Residual variances must be positive")

    order = np.argsort(residuals, kind="stable")
    sorted_residuals = residuals[order]

    best_score, previous = optimal_partition_core(sorted_residuals, BIC_constant)

    partition = []
    end = len(residuals)

    while end > 0:
        start = int(previous[end])
        partition.append(set(int(x) for x in order[start:end]))
        end = start

    partition.reverse()
    return utils.sorted_partition(partition), best_score
    
@njit(cache=True)
def optimal_partition_core(r, penalty):
    p = len(r)

    prefix = np.empty(p + 1)
    prefix[0] = 0.0
    for i in range(p):
        prefix[i + 1] = prefix[i] + r[i]

    dp = np.empty(p + 1)
    previous = np.empty(p + 1, dtype=np.int64)

    dp[0] = 0.0
    previous[0] = -1

    for end in range(1, p + 1):
        best_score = np.inf
        best_start = -1

        for start in range(end):
            size = end - start
            block_mean = (prefix[end] - prefix[start]) / size
            block_cost = (0.5 * size * (np.log(block_mean) + 1.0) + penalty)
            candidate = dp[start] + block_cost

            if candidate < best_score:
                best_score = candidate
                best_start = start

        dp[end] = best_score
        previous[end] = best_start

    return dp[p], previous





def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 10
    no_colors = 3
    edge_prob = 0.6
    sample_size = 1000
    num_starts = 8

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    

    t = time.perf_counter()
    edge_array, partition, bic = Causal_Greedy_Color(samples, num_starts)


    print(f"Ran Hybrid with {num_starts} starts")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Greedy: SHD to real DAG was:", utils.calc_SHD(edge_array, real_edge_array))
    print("GES: SHD to real DAG was:", utils.calc_SHD(GES_edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("Greedy")


    plt.show()

        
    



if __name__ == "__main__":
    main()

