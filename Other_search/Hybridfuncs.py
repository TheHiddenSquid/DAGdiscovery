import copy
import random
import time

import numpy as np
import utils

# Main MCMC function

def CausalHybridSearch(samples, num_waves):
    
    # Setup global variables
    global num_nodes
    global num_samples
    global BIC_constant

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Begin with empty DAG
    A = np.zeros((num_nodes,num_nodes), dtype=np.int64)
    P = [{i} for i in range(num_nodes)]

    
  
    # Setup for iters
    
    score_info = score_DAG(samples, A, P)


    best_A = A.copy()
    best_P = copy.deepcopy(P)
    best_bic = score_info[0]
    best_wave = 0
    best_iter = 0
    wave_duration = 1_000
  
    wave = 0
    for _ in range(num_waves):
        iter = 0
        wave += 1
        for _ in range(wave_duration):
            iter += 1
            A, P, score_info, _ = MCMC_iteration(samples, A, P, score_info)
            if score_info[0] > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = score_info[0]
                best_wave = wave
                best_iter = iter


        sorted_edges = get_sorted_edges(A)
        done = False
        while not done:
            iter += 1
            A, P, score_info, sorted_edges, done = Greedyiteration(samples, A, P, score_info, sorted_edges)
            if score_info[0] > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = score_info[0]
                best_wave = wave
                best_iter = iter
        
        print(best_bic)

    CPDAG_A = utils.getCPDAG(best_A, best_P)
    return CPDAG_A, best_P, best_bic, (best_wave, best_iter)
    


def Greedyiteration(samples, A, P, score_info, sorted_edges, moves = None):
    # Idea: generate a number and just save that as the potential random move

    best_move = None
    best_A = None
    best_P = None
    best_score_info = None
    best_BIC = score_info[0]
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges

    if moves is not None:
        do_colors = moves[0]
        do_edges = moves[1]
    else:
        do_colors = do_edges = 1


    # Check all neighboring colorings
    if do_colors:
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


                potential_score_info = score_DAG_color_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])

                if potential_score_info[0] > best_BIC:
                    best_P = copy.deepcopy(P)
                    best_score_info = potential_score_info
                    best_move = "change_color"
                    best_BIC = potential_score_info[0]


                P[new_color].remove(node)
            P[old_color].add(node)


    # Check all potential edge adds
    if do_edges:
        for edge in edges_giving_DAGs:
            A[edge] = 1

            potential_score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], edge])

            if potential_score_info[0] > best_BIC:
                best_A = A.copy()
                best_score_info = potential_score_info
                best_move = "add_edge"
                best_saved_edge = edge
                best_BIC = potential_score_info[0]
        
            A[edge] = 0


    # Check all potential edge removals
        for edge in edges_in_DAG:
            A[edge] = 0
            
            potential_score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], edge])
  
            if potential_score_info[0] > best_BIC:
                best_A = A.copy()
                best_score_info = potential_score_info
                best_move = "remove_edge"
                best_saved_edge = edge
                best_BIC = potential_score_info[0]
            
            A[edge] = 1



    # Do the best possible jump

    if best_score_info is None:
        return A, P, score_info, sorted_edges, True
    
    else:
        new_score_info = best_score_info
        
        if best_move == "change_color":
            new_A = A
            new_P = best_P
            new_sorted_edges = sorted_edges
        elif best_move == "add_edge":
            new_A = best_A
            new_P = P
            new_sorted_edges = update_sorted_edges_ADD(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "remove_edge":
            new_A = best_A
            new_P = P
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)

        
    return new_A, new_P, new_score_info, new_sorted_edges, False

def MCMC_iteration(data, A, P, score_info, move_weights = [1,1]):
    # Check what moves are possible and pick one at random

    moves = ["change_color", "change_edge"]
    move = random.choices(moves, weights=move_weights)[0]

    # Create new colored DAG based on move
    match move:
        case "change_color":
            P, node, old_color, new_color = change_partiton(P)

            potential_score_info = score_DAG_color_edit(data, A, P, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])

        case "change_edge":
            A, edge = change_edge(A)
        
            potential_score_info = score_DAG_edge_edit(data, A, P, [score_info[1], score_info[2], score_info[3], edge])


    # Metropolis Hastings to accept or reject new colored DAG
    if random.random() <= np.exp(potential_score_info[0] - score_info[0]):
        new_score_info = potential_score_info
        failed = 0
    else:
        if move == "change_color":
            P[new_color].remove(node)
            P[old_color].add(node)
        elif move == "change_edge":
            A[edge] = 1 - A[edge]

        new_score_info = score_info
        failed = 1

    
    return A, P, new_score_info, failed



# For moves
def change_partiton(P):
    node_to_change = random.randrange(num_nodes)
    old_color = None
    other_colors = []

    for i, part in enumerate(P):
        if node_to_change in part:
            old_color = i
        elif len(part) != 0:
            other_colors.append(i)
        else:
            empty_color = i

    if len(P[old_color]) != 1:
        other_colors.append(empty_color)

    P[old_color].remove(node_to_change)
    new_color = random.choice(other_colors)
    P[new_color].add(node_to_change)

    return P, node_to_change, old_color, new_color

def change_edge(A):
    while True:
        edge = (random.randrange(num_nodes), random.randrange(num_nodes))

        if A[edge] == 1:
            A[edge] = 0
            break
        else:
            tmp = A.copy()
            tmp[edge] = 1
            if utils.is_DAG(tmp):
                A = tmp
                break
    
    return A, edge



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
def score_DAG(samples, A, P):
    samples = samples.T
    
    global num_nodes
    global num_samples
    global BIC_constant
    num_nodes = samples.shape[0]
    num_samples = samples.shape[1]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = utils.get_parents(i, A)
        ans = np.linalg.lstsq(samples[parents,:].T, samples[i,:].T, rcond=None)[0]
        edges_ML[parents, i] = ans

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * num_nodes
    bic_decomp = [0] * num_nodes

    for i, part in enumerate(P):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, A)
            tot += np.dot(x:=(samples[node,:] - edges_ML[parents,node].T @ samples[parents,:]), x)
        omegas_ML[i] = tot / (num_samples * len(part))


        # Calculate BIC
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_color_edit(samples, A, P, last_change_data):
    samples = samples.T
    
    # Edge ML is the same
    edges_ML = last_change_data[0]


    # Node ML needs local update
    omegas_ML = last_change_data[1].copy()
    
    node, old_color, new_color = last_change_data[3]
    parents = utils.get_parents(node, A)
    node_ml_contribution = np.dot(x:=(samples[node,:] - edges_ML[parents,node].T @ samples[parents,:]), x)

    if len(P[old_color]) == 0:
        omegas_ML[old_color] = None
    else:
        tot = omegas_ML[old_color] * num_samples * (len(P[old_color]) + 1)
        tot -= node_ml_contribution
        omegas_ML[old_color] = tot / (num_samples * len(P[old_color]))
    
    if len(P[new_color]) == 1:
        tot = 0
    else:
        tot = omegas_ML[new_color] * num_samples * (len(P[new_color]) - 1)
    tot += node_ml_contribution
    omegas_ML[new_color] = tot / (num_samples * len(P[new_color]))


    # Calculate BIC
    bic_decomp = last_change_data[2].copy()

    for i in [old_color, new_color]:
        part = P[i]
        if len(part) == 0:
            bic_decomp[i] = 0
            continue
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_edge_edit(samples, A, P, last_change_data):
    samples = samples.T

    # Calculate ML-eval of the different lambdas
    edges_ML = last_change_data[0].copy()
    
    new_parent, new_child = last_change_data[3]
    new_parents = utils.get_parents(new_child, A)
    old_parents = new_parents.copy()
    try:
        old_parents.remove(new_parent)
    except ValueError:
        old_parents.append(new_parent)

    old_ml = edges_ML[old_parents, new_child]
    new_ml = np.linalg.lstsq(samples[new_parents,:].T, samples[new_child,:].T, rcond=None)[0]
    edges_ML[new_parents, new_child] = new_ml


    # Calculate ML-eval of the different color omegas
    omegas_ML = last_change_data[1].copy()

    for i, part in enumerate(P):
        if new_child in part:
            current_color = i
            break

    part = P[current_color]
    tot = omegas_ML[current_color] * num_samples * len(part)
    tot -= np.dot(x:=(samples[new_child,:] - old_ml.T @ samples[old_parents,:]), x)
    tot += np.dot(x:=(samples[new_child,:] - new_ml.T @ samples[new_parents,:]), x)
    omegas_ML[current_color] = tot / (num_samples * len(part))


    # Calculate BIC
    bic_decomp = last_change_data[2].copy()
    bic_decomp[current_color] = -len(part) * (np.log(omegas_ML[current_color]) + 1)
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))


    return [bic, edges_ML, omegas_ML, bic_decomp]



def main():
    pass



if __name__ == "__main__":
    main()

