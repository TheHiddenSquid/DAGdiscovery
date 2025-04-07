import copy
import random
from collections import defaultdict
from itertools import repeat

import ges
import numpy as np
import utils

# Main MCMC functions

def CausalMCMC(data, num_iters = None, mode = "bic", move_weights = None, A0 = None, P0 = None, debug = False):
    """
    Run MCMC to find the colored DAG that best fits the data. The data is assumed to be
    centered.

    Parameters
    ----------
    data : numpy.ndarray
        The n x p array containing the observations,
        where columns correspond to variables and rows to observations.
    num_iters : int, optional
        The number of MCMC iterations performed. Default is 50*4^p, where p is the number of variables.
        If p>8 default will take a very long time.
    mode : [{'bic', 'map'}*], optional
        Option on what DAG the algorithm will return. If 'bic' it returns the DAG with the highest bic-score.
        If 'map' it returns the most visited DAG. Default is 'bic'.
    move_weights : [int, int], optional
        List of probability of changing a color versuis changing an edge. First value is color.
        Default is [0.4, 0.6]
    start_from_GES : bool, optional
        Option to start the search at a random GES DAG.
    A0 : numpy.ndarray, optional
        The initial DAG on which the algorithm will run, where where `A0[i,j]
        != 0` implies the edge `i -> j`.
    P0 : [{int}*] optional
        The initial coloring partition on which the algorithm will run.
    debug : bool, optional
        If true, returns extra values, such as number exact iteration best DAG was found and the number of failed jumps.

    Returns
    -------
    estimate_edges : numpy.ndarray
        The adjacency matrix of the estimated DAG.
    estimate_partition : [[int]*]
        The estimated node-partition.
    total_score : float
        The score of the estimate.

    Raises
    ------
    TypeError:
        If the type of some of the parameters was not expected,
        e.g. if data is not a numpy array.
    ValueError:
        If the value of some of the parameters is not appropriat.
    """

    # Setup global variables
    global num_nodes
    global num_samples
    global BIC_constant

    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Calculate number of iterations
    if num_iters is None:
        num_iters = 50 * 4**num_nodes
        if num_nodes > 8:
            print("Warning! Default number of iterations on more than 8 variables will take time")
    elif not isinstance(num_iters, int):
        raise TypeError("num_iters needs to be an int")
    elif num_iters < 0:
        raise ValueError("num_iters needs to be positive")

    
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
        move_weights = [0.4, 0.6]

    
    # Perform optional setup
    A = np.zeros((num_nodes, num_nodes))
    P = [{i} for i in range(num_nodes)]

    if P0 is not None:
        P = copy.deepcopy(P0)

    if A0 is not None:
        A = A0


    # Setup for iters
    score_info = score_DAG(data, A, P)


    if mode == "bic":
        best_A = A.copy()
        best_P = utils.sorted_partition(P)
        best_bic = score_info[0]
        best_iter = 0
        num_fails = 0

        # Run MCMC iters    
        for i in range(num_iters):
            A, P, score_info, fail = MCMC_iteration(data, A, P, score_info, move_weights)    
            if score_info[0] > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = score_info[0]
                best_iter = i
            num_fails += fail

        CPDAG_A = utils.getCPDAG(best_A, best_P)
        if debug: 
            return CPDAG_A, best_P, best_bic, best_iter, num_fails
        else:
            return CPDAG_A, best_P, best_bic
    

    if mode == "map":
        cashe = defaultdict(lambda: 0)
        num_fails = 0

        # Run MCMC iters    
        for _ in repeat(None, num_iters):

            A, P, score_info, fail = MCMC_iteration(data, A, P, score_info, move_weights)

            h1 = A.tobytes()
            h2 = tuple(tuple(x) for x in utils.sorted_partition(P))

            cashe[(h1,h2)] += 1
            num_fails += fail

        most_visited = max(cashe, key=cashe.get)
        num_visits = cashe[most_visited]

        best_A = most_visited[0]
        best_A = np.frombuffer(best_A, dtype=np.int64)
        best_A = np.reshape(best_A, (num_nodes,num_nodes))

        best_P = [set(x) for x in most_visited[1]]

        CPDAG_A = utils.getCPDAG(best_A, best_P)
        if debug: 
            return CPDAG_A, best_P, num_visits, num_fails
        else:
            return CPDAG_A, best_P, num_visits
    
def MCMC_iteration(samples, A, P, score_info, move_weights):
    # Check what moves are possible and pick one at random

    moves = ["change_color", "change_edge"]
    move = random.choices(moves, weights=move_weights)[0]

    # Create new colored DAG based on move
    match move:
        case "change_color":
            P, node, old_color, new_color = change_partiton(P)

            potential_score_info = score_DAG_color_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])

        case "change_edge":
            A, edge = change_edge(A)
        
            potential_score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], edge])


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



# For DAG heuristic
def score_DAG(data, A, P):
    data = data.T
    
    global num_nodes
    global num_samples
    global BIC_constant
    num_nodes = data.shape[0]
    num_samples = data.shape[1]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = utils.get_parents(i, A)
        ans = np.linalg.lstsq(data[parents,:].T, data[i,:].T, rcond=None)[0]
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
            tot += np.dot(x:=(data[node,:] - edges_ML[parents,node].T @ data[parents,:]), x)
        omegas_ML[i] = tot / (num_samples * len(part))


        # Calculate BIC
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_color_edit(data, A, P, last_change_data):
    data = data.T
    
    
    # Edge ML is the same
    edges_ML = last_change_data[0]


    # Node ML needs local update
    omegas_ML = last_change_data[1].copy()
    
    node, old_color, new_color = last_change_data[3]
    parents = utils.get_parents(node, A)
    node_ml_contribution = np.dot(x:=(data[node,:] - edges_ML[parents,node].T @ data[parents,:]), x)

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

def score_DAG_edge_edit(data, A, P, last_change_data):
    data = data.T
    

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
    new_ml = np.linalg.lstsq(data[new_parents,:].T, data[new_child,:].T, rcond=None)[0]
    edges_ML[new_parents, new_child] = new_ml


    # Calculate ML-eval of the different color omegas
    omegas_ML = last_change_data[1].copy()

    for i, part in enumerate(P):
        if new_child in part:
            current_color = i
            break

    part = P[current_color]
    tot = omegas_ML[current_color] * num_samples * len(part)
    tot -= np.dot(x:=(data[new_child,:] - old_ml.T @ data[old_parents,:]), x)
    tot += np.dot(x:=(data[new_child,:] - new_ml.T @ data[new_parents,:]), x)
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
