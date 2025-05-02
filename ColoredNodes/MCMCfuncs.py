import copy
import random
from collections import defaultdict
from itertools import repeat

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
    global my_data
    global num_nodes
    global num_samples
    global BIC_constant

    my_data = data.T
    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Calculate number of iterations
    if num_iters is None:
        num_iters = 25 * 5**num_nodes
        if num_nodes > 8:
            print("Warning! Default number of iterations on more than 8 variables is too big. Set to 50*4^8")
            num_iters = 50 * 4**8
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
    bic, ML_data = score_DAG_full(data, A, P)


    if mode == "bic":
        best_A = A.copy()
        best_P = utils.sorted_partition(P)
        best_bic = bic
        best_iter = 0
        num_fails = 0

        # Run MCMC iters    
        for i in range(num_iters):
            A, P, bic, ML_data, fail = MCMC_iteration(A, P, bic, ML_data, move_weights)    
            if bic > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = bic
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
            A, P, bic, ML_data, fail = MCMC_iteration(A, P, bic, ML_data, move_weights)
            cashe[utils.hash_DAG(A, P)] += 1
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
    
def MCMC_iteration(A, P, bic, ML_data, move_weights):
    # Check what moves are possible and pick one at random

    moves = ["change_color", "change_edge"]
    move = random.choices(moves, weights=move_weights)[0]

    # Create new colored DAG based on move
    match move:
        case "change_color":
            P, node, old_color, new_color = change_partiton(P)
            potential_bic, potential_ML_data = score_DAG_color_edit(A, P, ML_data)
 
        case "change_edge":
            A, edge, did_change = change_edge(A)
            if did_change:
                potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)
            else:
                potential_bic, potential_ML_data = bic, ML_data


    # Metropolis algorithm to accept or reject new colored DAG
    if random.random() <= np.exp(potential_bic - bic):
        new_bic = potential_bic
        new_ML_data = potential_ML_data
        failed = 0
    else:
        if move == "change_color":
            P[new_color].remove(node)
            P[old_color].add(node)
        elif move == "change_edge":
            A[edge] = 1 - A[edge]

        new_bic = bic
        new_ML_data = ML_data
        failed = 1

    
    return A, P, new_bic, new_ML_data, failed



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
    did_change = False
    n1 = random.randrange(num_nodes)
    n2 = random.randrange(num_nodes)
    while n2 == n1:
        n2 = random.randrange(num_nodes)
    edge = (n1, n2)

    if A[edge] == 1:
        A[edge] = 0
        did_change = True
    else:
        A[edge] = 1
        if utils.is_DAG(A):
            did_change = True
        else:
            A[edge] = 0
    
    return A, edge, did_change



# For DAG heuristic
def score_DAG_full(data, A, P):
    data = data.T
    
    global num_nodes
    global num_samples
    global BIC_constant
    num_nodes = data.shape[0]
    num_samples = data.shape[1]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval
    omegas_ML = [None] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        beta, ss_res = np.linalg.lstsq(data[parents,:].T, data[node,:].T, rcond=None)[:2]
        omegas_ML[node] = ss_res[0] / num_samples

    # Block the omegas as averages
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        # Calculate BIC
        bic_decomp[i] = -len(block) * (np.log(block_omega) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))
    
    return bic, omegas_ML

def score_DAG_color_edit(A, P, ML_data):
    
    # ML data is the same
    omegas_ML = ML_data

    # Calculate BIC
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        # Calculate BIC
        bic_decomp[i] = -len(block) * (np.log(block_omega) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))

    return bic, omegas_ML

def score_DAG_edge_edit(A, P, ML_data, changed_edge):

    # Get old ML-eval
    omegas_ML = ML_data.copy()

    # Update ML-eval
    _, node = changed_edge
    parents = utils.get_parents(node, A)
    beta, ss_res = np.linalg.lstsq(my_data[parents,:].T, my_data[node,:].T, rcond=None)[:2]
    omegas_ML[node] = ss_res[0] / num_samples

    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        omega = tot / len(block)

        # Calculate BIC
        bic_decomp[i] = -len(block) * (np.log(omega) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))


    return bic, omegas_ML



def main():
    pass



if __name__ == "__main__":
    main()
