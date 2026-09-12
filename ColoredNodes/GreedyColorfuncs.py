import copy
import functools
import math

import numpy as np
import utils
from numba import njit

"""
Greedy solver that searches through the space of all clored DAGs (G, P) - pairs during the optimization. 
This is fast per iteratation, but the search space grows extremely quick with p.
"""

# Main functions

def CausalGreedyColor(samples, num_waves = 5, moves = None):
    
    #Clear cache for new run of algorithm
    calc_lstsq_S.cache_clear()
    
    # Setup global variables
    global S
    global num_edges
    global num_nodes
    global BIC_constant
    global used_moves

    if moves == None:
        used_moves = ["change_color", "add_edge", "remove_edge", "flip_edge"]
    else:
        used_moves = moves

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    S = (1/num_samples) * samples.T @ samples
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes))
    num_edges = np.count_nonzero(best_A)
    best_P = [{i} for i in range(num_nodes)]
    best_bic, _ = score_DAG_full(best_A, best_P)

    edge_probs = list(np.linspace(0,1,num_waves))
    num_colors = [int(x) for x in np.linspace(1,num_nodes,num_waves)]

    
    # Perform iterations
    for i in range(num_waves):
        P, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_colors[i], edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)
        for _ in range(num_nodes-len(P)):
            P.append(set())

        num_edges = np.count_nonzero(A)
        bic, ML_data = score_DAG_full(A, P)
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            A, P, bic, ML_data, sorted_edges, done = iteration(A, P, bic, ML_data, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = bic

    CPDAG_A = utils.getCPDAG(best_A, best_P)
    return CPDAG_A, best_P, best_bic
    
def iteration(A, P, bic, ML_data, sorted_edges):
    global num_edges

    best_move = None
    best_A = None
    best_P = None
    best_bic = bic
    best_ML_data = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges


    # Check all neighboring colorings
    if "change_color" in used_moves:
        for node in range(num_nodes):
            old_color = None
            other_colors = []

            for i, part in enumerate(P):
                if node in part:
                    old_color = i
                elif len(part) != 0:
                    other_colors.append(i)
                else:
                    empty_color = i

            if len(P[old_color]) != 1:
                other_colors.append(empty_color)

            P[old_color].remove(node)
            for new_color in other_colors:
                P[new_color].add(node)

                potential_bic, potential_ML_data = score_DAG_color_edit(P, ML_data, node, old_color, new_color)

                if potential_bic > best_bic:
                    best_P = copy.deepcopy(P)
                    best_bic = potential_bic
                    best_ML_data = potential_ML_data
                    best_move = "change_color"


                P[new_color].remove(node)
            P[old_color].add(node)


    # Check all potential edge adds
    if "add_edge" in used_moves:
        for edge in edges_giving_DAGs:
            A[edge] = 1
            num_edges += 1
            potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)

            if potential_bic > best_bic:
                best_A = A.copy()
                best_bic = potential_bic
                best_ML_data = potential_ML_data
                best_move = "add_edge"
                best_saved_edge = edge
        
            A[edge] = 0
            num_edges -= 1


    # Check all potential edge removals
    if "remove_edge" in used_moves:
        for edge in edges_in_DAG:
            A[edge] = 0
            num_edges -= 1
            
            potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)

            if potential_bic > best_bic:
                best_A = A.copy()
                best_bic = potential_bic
                best_ML_data = potential_ML_data
                best_move = "remove_edge"
                best_saved_edge = edge
            
            A[edge] = 1
            num_edges += 1


    if "flip_edge" in used_moves:
        for edge in edges_in_DAG:
            rev = (edge[1], edge[0])
            A[edge] = 0
            A[rev] = 1
        
            if utils.is_DAG(A):
                _, tmp_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)
                potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, tmp_ML_data, rev)
    
                if potential_bic > best_bic:
                    best_A = A.copy()
                    best_bic = potential_bic
                    best_ML_data = potential_ML_data
                    best_move = "flip_edge"
                    best_saved_edge = [edge, rev]
                        
            A[edge] = 1
            A[rev] = 0



    # Do the best possible jump

    if best_ML_data is None:
        return A, P, bic, ML_data, sorted_edges, True
    
    else:
        new_bic = best_bic 
        new_ML_data = best_ML_data
        
        if best_move == "change_color":
            new_A = A
            new_P = best_P
            new_sorted_edges = sorted_edges
        elif best_move == "add_edge":
            new_A = best_A
            new_P = P
            num_edges += 1
            new_sorted_edges = update_sorted_edges_ADD(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "remove_edge":
            new_A = best_A
            new_P = P
            num_edges -= 1
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "flip_edge":
            new_A = best_A
            new_P = P
            removed_edge, added_edge = best_saved_edge
            tmp = new_A.copy()
            tmp[added_edge] = 0
            tmp_edges = update_sorted_edges_REMOVE(tmp, *sorted_edges, removed_edge)
            new_sorted_edges = update_sorted_edges_ADD(new_A, *tmp_edges, added_edge)

        
    return new_A, new_P, new_bic, new_ML_data, new_sorted_edges, False



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
def score_DAG_full(A, P):

    # Calculate ML-eval via least sqares
    omegas_ML = [0] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        ss_res = calc_lstsq_S(node, tuple(parents))
        omegas_ML[node] = ss_res


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    block_sums = [0] * num_nodes

    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_sums[i] = tot
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (math.log(block_omega) + 1)
    
    # Calculate full BIC
    bic_decomp_sum = sum(bic_decomp)
    nc = sum(1 for x in P if len(x)!=0)
    bic = bic_decomp_sum/2 - BIC_constant * (num_edges + nc)
    
    return bic, [omegas_ML, block_sums, bic_decomp, bic_decomp_sum]

def score_DAG_color_edit(P, ML_data, node, old_color, new_color):
    
    # ML data is the same
    omegas_ML, block_sums, bic_decomp, bic_decomp_sum = ML_data
    bic_decomp = bic_decomp.copy()
    block_sums = block_sums.copy()

    
    # Update decomposed BIC
    bic_decomp_sum -= (bic_decomp[old_color] + bic_decomp[new_color])
    old_block = P[old_color]
    
    if len(old_block) != 0:
        block_sums[old_color] -= omegas_ML[node]
        old_block_omega = block_sums[old_color] / len(old_block)
        bic_decomp[old_color] = -len(old_block) * (math.log(old_block_omega) + 1)
    else:
        block_sums[old_color] = 0
        bic_decomp[old_color] = 0

    new_block = P[new_color]
    block_sums[new_color] += omegas_ML[node]
    new_block_omega = block_sums[new_color] / len(new_block)
    bic_decomp[new_color] = -len(new_block) * (math.log(new_block_omega) + 1)
    bic_decomp_sum += (bic_decomp[old_color] + bic_decomp[new_color])


    # Calculate full BIC
    nc = sum(1 for x in P if len(x)!=0)
    bic = bic_decomp_sum/2 - BIC_constant * (num_edges + nc)
    
    return bic, [omegas_ML, block_sums, bic_decomp, bic_decomp_sum]

def score_DAG_edge_edit(A, P, ML_data, changed_edge):

    # Get old ML-eval
    omegas_ML, block_sums, bic_decomp, bic_decomp_sum = ML_data
    omegas_ML = omegas_ML.copy()
    block_sums = block_sums.copy()
    bic_decomp = bic_decomp.copy()

    
    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    ss_res = calc_lstsq_S(active_node, tuple(parents))
    old_omega_ML = omegas_ML[active_node]
    new_omega_ML = ss_res
    omegas_ML[active_node] = new_omega_ML


    # Update decomposed BIC
    for i, block in enumerate(P):
        if active_node in block:
            active_block_id = i

    bic_decomp_sum -= bic_decomp[active_block_id]
    active_block = P[active_block_id]
    block_sums[active_block_id] -= old_omega_ML
    block_sums[active_block_id] += new_omega_ML
    block_omega = block_sums[active_block_id] / len(active_block)
    bic_decomp[active_block_id] = -len(active_block) * (math.log(block_omega) + 1)
    bic_decomp_sum += bic_decomp[active_block_id]
  

    # Calculate full BIC
    nc = sum(1 for x in P if len(x)!=0)
    bic = bic_decomp_sum/2 - BIC_constant * (num_edges + nc)

    return bic, [omegas_ML, block_sums, bic_decomp, bic_decomp_sum]


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


def main():
    pass



if __name__ == "__main__":
    main()

