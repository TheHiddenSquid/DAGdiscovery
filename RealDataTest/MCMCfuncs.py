import copy
import random
from collections import defaultdict
from itertools import repeat

import ges
import numpy as np
import utils

# Main MCMC function

def CausalMCMC(samples, num_iters, mode = "bic", start_from_GES = False, move_weights = None, start_edge_array = None, start_partition = None, debug = False):

    # Check that mode is legal
    if mode not in ["bic", "map"]:
        raise ValueError("Mode not supported")
    
    # Check that wieghts are legal
    if move_weights is not None:
        if len(move_weights) != 2:
            raise ValueError("Lenght of weights must be 3")
        p_change_color, p_change_edge = move_weights
        if p_change_color<0 or p_change_edge<0 or not np.isclose(sum(move_weights),1):
            raise ValueError("Invalid move probabilities")
    else:
        move_weights = [1/2]*2

    
    # Setup global variables
    global num_nodes
    global num_samples
    global BIC_constant

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Perform optional setup
    if start_from_GES:
        GES_edge_array = ges.fit_bic(data=samples)[0]

        # Take an initial DAG from the given GES CPDAG
        A =  GES_edge_array.copy()
        double = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if A[i,j] == 1 and A[j,i] == 1:
                    double.append((i,j))
                    A[i,j] = 0
                    A[j,i] = 0

        for edge in double:
            new_edges = A.copy()
            new_edges[edge[0], edge[1]] = 1
            if utils.is_DAG(new_edges):
                A = new_edges
                continue

            new_edges = A.copy()
            new_edges[edge[1], edge[0]] = 1
            if utils.is_DAG(new_edges):
                A = new_edges
                continue

            raise ValueError("Could not create DAG")

        # Every node has its own color
        partition = [{i} for i in range(num_nodes)]

    else:
        # Fully random colored DAG
        A = utils.get_random_DAG(num_nodes, sparse=True)
        partition = [{i} for i in range(num_nodes)]
    

    if start_partition is not None:
        partition = copy.deepcopy(start_partition)

    if start_edge_array is not None:
        A = start_edge_array


    # Setup for iters
    score_info = score_DAG(samples, A, partition)


    if mode == "bic":
        best_A = A.copy()
        best_partition = copy.deepcopy(partition)
        best_bic = score_info[0]
        best_iter = 0
        num_fails = 0

        # Run MCMC iters    
        for i in range(num_iters):
            A, partition, score_info, fail = MCMC_iteration(samples, A, partition, score_info, move_weights)    
            if score_info[0] > best_bic:
                best_A = A.copy()
                best_partition = utils.sorted_partition(partition)
                best_bic = score_info[0]
                best_iter = i
            num_fails += fail

        if debug: 
            return best_A, best_partition, best_bic, best_iter, num_fails
        else:
            return best_A, best_partition, best_bic
    

    if mode == "map":
        cashe = defaultdict(lambda: 0)
        num_fails = 0

        # Run MCMC iters    
        for _ in repeat(None, num_iters):

            A, partition, score_info, fail = MCMC_iteration(samples, A, partition, score_info, move_weights)

            h1 = A.tobytes()
            h2 = tuple(tuple(x) for x in utils.sorted_partition(partition))

            cashe[(h1,h2)] += 1
            num_fails += fail

        most_visited = max(cashe, key=cashe.get)
        num_visits = cashe[most_visited]

        best_A = most_visited[0]
        best_A = np.frombuffer(best_A, dtype=np.int64)
        best_A = np.reshape(best_A, (num_nodes,num_nodes))

        best_partition = [set(x) for x in most_visited[1]]

        if debug: 
            return best_A, best_partition, num_visits, num_fails
        else:
            return best_A, best_partition, num_visits
    


def MCMC_iteration(samples, edge_array, partition, score_info, move_weights):
    # Check what moves are possible and pick one at random

    moves = [ "change_color", "change_edge"]
    move = random.choices(moves, weights=move_weights)[0]

    # Create new colored DAG based on move
    match move:
        case "change_color":
            partition, node, old_color, new_color = change_partiton(partition)

            potential_score_info = score_DAG_color_edit(samples, edge_array, partition, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])

        case "change_edge":
            while True:
                edge = (random.randrange(num_nodes), random.randrange(num_nodes))

                if edge_array[edge] == 1:
                    edge_array[edge] = 0
                    break
                else:
                    tmp = edge_array.copy()
                    tmp[edge] = 1
                    if utils.is_DAG(tmp):
                        edge_array = tmp
                        break
        
            potential_score_info = score_DAG_edge_edit(samples, edge_array, partition, [score_info[1], score_info[2], score_info[3], edge])


    # Metropolis Hastings to accept or reject new colored DAG
    if random.random() <= np.exp(potential_score_info[0] - score_info[0]):
        new_score_info = potential_score_info
        failed = 0
    else:
        if move == "change_color":
            partition[new_color].remove(node)
            partition[old_color].add(node)
        elif move == "change_edge":
            if edge_array[edge] == 1:
                edge_array[edge] = 0
            else:
                tmp = edge_array.copy()
                tmp[edge] = 1
                if utils.is_DAG(tmp):
                    edge_array = tmp


        new_score_info = score_info
        failed = 1

    
    return edge_array, partition, new_score_info, failed



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
