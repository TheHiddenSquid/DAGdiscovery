import copy
import random

import numpy as np
import utils

# Main MCMC function

def CausalTabuSearch(samples, num_iters):
    
    # Setup global variables
    global num_nodes
    global num_samples
    global BIC_constant

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Fully random colored DAG
    A = np.zeros((num_nodes,num_nodes), dtype=np.int64)
    partition = [{i} for i in range(num_nodes)]
    
  
    # Setup for iters
    sorted_edges = get_sorted_edges(A)
    score_info = score_DAG(samples, A, partition)


    best_A = A.copy()
    best_partition = copy.deepcopy(partition)
    best_bic = score_info[0]
    best_iter = 0


    # Run MCMC iters    
    for i in range(num_iters):
        A, partition, score_info, sorted_edges = iteration(samples, A, partition, score_info, sorted_edges)    
        if score_info[0] > best_bic:
            best_A = A.copy()
            best_partition = utils.sorted_partition(partition)
            best_bic = score_info[0]
            best_iter = i

    return best_A, best_partition, best_bic, best_iter
    


def iteration(samples, edge_array, partition, score_info, sorted_edges):
    
    edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs = sorted_edges
    num_edges = len(edges_in_DAG)


    # Check what moves are possible and pick one at random

    
    moves = [ "change_color", "add_edge", "remove_edge"]
    weights = [1/3]*3

    if num_edges == 0:
        weights[2] = 0  
    elif len(edges_giving_DAGs) == 0 :
        weights[1] = 0


    move = random.choices(moves, weights = weights, k = 1)[0]


    # Create new colored DAG based on move
    match move:
        case "change_color":
            partition, node, old_color, new_color = change_partiton(partition)

            potential_score_info = score_DAG_color_edit(samples, edge_array, partition, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])

        case "add_edge":
            edge = random.choice(edges_giving_DAGs)
            edge_array[edge] = 1

            potential_score_info = score_DAG_edge_edit(samples, edge_array, partition, [score_info[1], score_info[2], score_info[3], edge])

        case "remove_edge":
            edge = random.choice(edges_in_DAG)
            edge_array[edge] = 0

            potential_sorted_edges = update_sorted_edges_REMOVE(edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
            potential_score_info = score_DAG_edge_edit(samples, edge_array, partition, [score_info[1], score_info[2], score_info[3], edge])


    # Metropolis Hastings to accept or reject new colored DAG
    if potential_score_info[0] > score_info[0]:
        new_score_info = potential_score_info

        if move == "change_color":
            new_sorted_edges = sorted_edges
        elif move == "add_edge":
            new_sorted_edges = update_sorted_edges_ADD(edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        elif move == "remove_edge":
            new_sorted_edges = potential_sorted_edges

    else:
        if move == "change_color":
            partition[new_color].remove(node)
            partition[old_color].add(node)
        elif move == "add_edge":
            edge_array[edge] = 0
        elif move == "remove_edge":
            edge_array[edge] = 1

        new_score_info = score_info
        new_sorted_edges = sorted_edges


        
    return edge_array, partition, new_score_info, new_sorted_edges



# For moves
def change_partiton(partition):
    node_to_change = random.randrange(num_nodes)
    old_color = None
    other_colors = []

    for i, part in enumerate(partition):
        if node_to_change in part:
            old_color = i
        elif len(part) != 0:
            other_colors.append(i)
        else:
            empty_color = i

    if len(partition[old_color]) != 1:
        other_colors.append(empty_color)

    partition[old_color].remove(node_to_change)
    new_color = random.choice(other_colors)
    partition[new_color].add(node_to_change)

    return partition, node_to_change, old_color, new_color

    

# For edge lookups

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
            if utils.is_DAG(tmp_edge_array):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_sorted_edges_REMOVE(edge_array, edges_in, addable_edges, not_addable_edges, removed_edge):
    
    tmp_edge_array = edge_array.copy()

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

def update_sorted_edges_ADD(edge_array, edges_in, addable_edges, not_addable_edges, added_edge):

    tmp_edge_array = edge_array.copy()

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
def score_DAG(samples, edge_array, partition):
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
        parents = utils.get_parents(i, edge_array)
        ans = np.linalg.lstsq(samples[parents,:].T, samples[i,:].T, rcond=None)[0]
        edges_ML[parents, i] = ans

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * num_nodes
    bic_decomp = [0] * num_nodes

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, edge_array)
            tot += np.dot(x:=(samples[node,:] - edges_ML[parents,node].T @ samples[parents,:]), x)
        omegas_ML[i] = tot / (num_samples * len(part))


        # Calculate BIC
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in partition if len(part)>0) + np.sum(edge_array))


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_color_edit(samples, edge_array, partition, last_change_data):
    samples = samples.T
    
    
    # Edge ML is the same
    edges_ML = last_change_data[0]


    # Node ML needs local update
    omegas_ML = last_change_data[1].copy()
    
    node, old_color, new_color = last_change_data[3]
    parents = utils.get_parents(node, edge_array)
    node_ml_contribution = np.dot(x:=(samples[node,:] - edges_ML[parents,node].T @ samples[parents,:]), x)

    if len(partition[old_color]) == 0:
        omegas_ML[old_color] = None
    else:
        tot = omegas_ML[old_color] * num_samples * (len(partition[old_color]) + 1)
        tot -= node_ml_contribution
        omegas_ML[old_color] = tot / (num_samples * len(partition[old_color]))
    
    if len(partition[new_color]) == 1:
        tot = 0
    else:
        tot = omegas_ML[new_color] * num_samples * (len(partition[new_color]) - 1)
    tot += node_ml_contribution
    omegas_ML[new_color] = tot / (num_samples * len(partition[new_color]))


    # Calculate BIC
    bic_decomp = last_change_data[2].copy()

    for i in [old_color, new_color]:
        part = partition[i]
        if len(part) == 0:
            bic_decomp[i] = 0
            continue
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in partition if len(part)>0) + np.sum(edge_array))


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_edge_edit(samples, edge_array, partition, last_change_data):
    samples = samples.T
    

    # Calculate ML-eval of the different lambdas
    edges_ML = last_change_data[0].copy()
    
    new_parent, new_child = last_change_data[3]
    new_parents = utils.get_parents(new_child, edge_array)
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

    for i, part in enumerate(partition):
        if new_child in part:
            current_color = i
            break

    part = partition[current_color]
    tot = omegas_ML[current_color] * num_samples * len(part)
    tot -= np.dot(x:=(samples[new_child,:] - old_ml.T @ samples[old_parents,:]), x)
    tot += np.dot(x:=(samples[new_child,:] - new_ml.T @ samples[new_parents,:]), x)
    omegas_ML[current_color] = tot / (num_samples * len(part))


    # Calculate BIC
    bic_decomp = last_change_data[2].copy()
    bic_decomp[current_color] = -len(part) * (np.log(omegas_ML[current_color]) + 1)
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in partition if len(part)>0) + np.sum(edge_array))


    return [bic, edges_ML, omegas_ML, bic_decomp]



def main():
    pass



if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()
