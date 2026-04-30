import copy
import functools
import math
import random
import time
from collections import defaultdict

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

    #Clear cache for new run of algorithm
    calc_lstsq.cache_clear()


    # Setup global constants
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
            print("Warning! Default number of iterations on more than 8 variables is too big. Set to 25*5^8")
            num_iters = 25 * 5**8
    elif not isinstance(num_iters, int):
        raise TypeError("num_iters needs to be an int")
    elif num_iters < 0:
        raise ValueError("num_iters needs to be positive")

    
    # Check that mode is legal
    if mode not in ["bic", "map"]:
        raise ValueError("Mode not supported")
    
    
    # Check that wieghts are legal and prepare moves
    if move_weights is not None:
        if len(move_weights) != 2:
            raise ValueError("Lenght of weights must be 3")
        p_change_color, p_change_edge = move_weights
        if p_change_color<0 or p_change_edge<0 or not np.isclose(sum(move_weights),1):
            raise ValueError("Invalid move probabilities")
    else:
        move_weights = [0.4, 0.6]
    
    moves = (np.random.random(num_iters) < move_weights[1]).astype(np.int8)

    # Setup fast random nodes
    global random_nodes
    random_nodes = list(np.random.randint(0, num_nodes, num_iters * 2))

    # Setup initial guesses
    global num_edges


    A = np.zeros((num_nodes, num_nodes))
    P = utils.PartitionManager(n=num_nodes, chunk_size=100_000)
 

    if A0 is not None:
        A = A0
    
    num_edges = np.count_nonzero(A)


    # Setup for iters
    bic, ML_data = score_DAG_full(A, P)


    if mode == "bic":
        best_A = A.copy()
        best_P = utils.sorted_partition(P.P)
        best_bic = bic
        best_iter = 0
        num_fails = 0

        # Run MCMC iters
        for i in range(num_iters):
            move = moves[i]
            A, P, bic, ML_data, fail = MCMC_iteration(move, A, P, bic, ML_data)  
            if bic > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P.P)
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
        for i in range(num_iters):
            move = moves[i]
            A, P, bic, ML_data, fail = MCMC_iteration(move, A, P, bic, ML_data)
            cashe[utils.hash_DAG(A, P.P)] += 1
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
    
def MCMC_iteration(move, A, P, bic, ML_data):
    global num_edges

    # Create new colored DAG based on move
    if move:
        A, edge, did_change = change_edge(A)
        if did_change:
            potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)  
        else:
            return A, P, bic, ML_data, 0
    else: 
        node, old_color, new_color = P.propose_move()
        P.perform_move(node, old_color, new_color)
        potential_bic, potential_ML_data = score_DAG_color_edit(P, ML_data, node, old_color, new_color)
     
   
    # Metropolis algorithm to accept or reject new colored DAG
    if random.random() <= np.exp(potential_bic - bic):
        new_bic = potential_bic
        new_ML_data = potential_ML_data
        failed = 0
        if not move:
            P.commit_move(node, old_color, new_color)
    else:
        if move:
            A[edge] = 1 - A[edge]
            num_edges += 2*A[edge] - 1
        else:
            P.undo_move(node, old_color, new_color)
            
        new_bic = bic
        new_ML_data = ML_data
        failed = 1

    return A, P, new_bic, new_ML_data, failed



def change_edge(A):
    global num_edges

    did_change = False
    n1 = random_nodes.pop()
    n2 = random_nodes.pop()
    while n2 == n1:
        n2 = random_nodes.pop()
    edge = (n1, n2)

    if A[edge] == 1:
        A[edge] = 0
        did_change = True
        num_edges -= 1
    elif not utils.is_reachable(A, n1, n2):
        A[edge] = 1
        did_change = True
        num_edges += 1
    
    return A, edge, did_change



# For DAG heuristic
def score_DAG_full(A, P):

    # Calculate ML-eval via least sqares
    omegas_ML = [0] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        _, ss_res = calc_lstsq(node, tuple(parents))
        omegas_ML[node] = ss_res / num_samples


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    block_sums = [0] * num_nodes

    for i, block in enumerate(P.P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_sums[i] = tot
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (math.log(block_omega) + 1)
    
    # Calculate full BIC
    bic = sum(bic_decomp)/2 - BIC_constant * (num_edges + len(P.active_blocks))
    
    return bic, [omegas_ML, bic_decomp, block_sums]

def score_DAG_color_edit(P, ML_data, node, old_color, new_color):
    
    # ML data is the same
    omegas_ML, bic_decomp, block_sums = ML_data
    bic_decomp = bic_decomp.copy()
    block_sums = block_sums.copy()

    # Update decomposed BIC
    old_block = P.P[old_color]
    
    if len(old_block) != 0:
        block_sums[old_color] -= omegas_ML[node]
        old_block_omega = block_sums[old_color] / len(old_block)
        bic_decomp[old_color] = -len(old_block) * (math.log(old_block_omega) + 1)
    else:
        block_sums[old_color] = 0
        bic_decomp[old_color] = 0

    new_block = P.P[new_color]
    block_sums[new_color] += omegas_ML[node]
    new_block_omega = block_sums[new_color] / len(new_block)
    bic_decomp[new_color] = -len(new_block) * (math.log(new_block_omega) + 1)

    # Calculate full BIC
    bic = sum(bic_decomp)/2 - BIC_constant * (num_edges + len(P.active_blocks))
    
    return bic, [omegas_ML, bic_decomp, block_sums]

def score_DAG_edge_edit(A, P, ML_data, changed_edge):

    # Get old ML-eval
    omegas_ML, bic_decomp, block_sums = ML_data
    omegas_ML = omegas_ML.copy()
    bic_decomp = bic_decomp.copy()
    block_sums = block_sums.copy()


    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    _, ss_res = calc_lstsq(active_node, tuple(parents))
    old_omega_ML = omegas_ML[active_node]
    new_omega_ML = ss_res / num_samples
    omegas_ML[active_node] = new_omega_ML
  

    # Update decomposed BIC
    active_block_id = P.node_to_block[active_node]
    active_block = P.P[active_block_id]
    block_sums[active_block_id] -= old_omega_ML
    block_sums[active_block_id] += new_omega_ML
    block_omega = block_sums[active_block_id] / len(active_block)

    bic_decomp[active_block_id] = -len(active_block) * (math.log(block_omega) + 1)


    # Calculate full BIC
    bic = sum(bic_decomp)/2 - BIC_constant * (num_edges + len(P.active_blocks))

    return bic, [omegas_ML, bic_decomp, block_sums]


@functools.cache
def calc_lstsq(node, parents):
    if len(parents) == 0:
        b = my_data[node,:]
        return [], np.dot(b,b)
    
    a = my_data[parents,:]
    b = my_data[node,:]
    beta = np.linalg.solve(a @ a.T, a @ b)
    x = b - a.T @ beta
    ss_res = np.dot(x,x)
    return beta, ss_res

def main():
    pass



if __name__ == "__main__":
    main()
