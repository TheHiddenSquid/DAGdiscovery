import copy
import functools
import math
import pickle
import random

import numpy as np
import utils

# Main MCMC functions

def CausalMCMC(sample, num_iters = None, move_weights = None, debug = False):

    #Clear cache for new run of algorithm
    calc_lstsq.cache_clear()

    # Setup constants
    global data
    global num_samples
    global num_nodes
    global data_S
    global BIC_constant
    data = sample.T
    num_samples = data.shape[1]
    num_nodes = data.shape[0]
    data_S = data @ data.T / num_samples
    BIC_constant = np.log(num_samples)/(num_samples*2)
   

    # Calculate number of iterations
    if num_iters is None:
        num_iters = 25 * 8**num_nodes
        if num_nodes > 6:
            print("Warning! Default number of iterations on more than 6 variables is too big. Set to 25*8^6")
            num_iters = 25 * 8**6
    elif not isinstance(num_iters, int):
        raise TypeError("num_iters needs to be an int")
    elif num_iters < 0:
        raise ValueError("num_iters needs to be positive")
    


    # Check that wieghts are legal
    move_weights = [0.3, 0.3, 0.4]
    moves = random.choices([0, 1, 2], k=num_iters, weights=move_weights)


    # Starting DAG
    A = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    PE = []
    PN = [[[i]] for i in range(num_nodes)]
    

    # Setup for iters
    score, *ML_data = score_DAG_full(A, PE, PN_flat = [sum(x,[]) for x in PN])
    best_A = A.copy()
    best_PE = copy.deepcopy(PE)
    best_PN = copy.deepcopy(PN)
    best_score = score
    best_iter = 0
    num_fails = 0

    # Run MCMC iters    
    for i in range(num_iters):
        move = moves[i]
        A, PE, PN, score, ML_data, fail = MCMC_iteration(move, A, PE, PN, score, ML_data) 

        if score >= best_score:
            best_A = A.copy()
            best_PE = copy.deepcopy(PE)
            best_PN = copy.deepcopy(PN)
            best_score = score
            best_iter = i
        num_fails += fail

    # Flatten node partition
    best_PN = [sum(x,[]) for x in best_PN]
    if debug: 
        return best_A, best_PE, best_PN, best_score, best_iter, num_fails
    else:
        return best_A, best_PE, best_PN, best_score
      
def MCMC_iteration(move, A, PE, PN, score, ML_data):

    # Check what moves are possible and pick one at random
    old_A = A.copy()
    old_PE = pickle.loads(pickle.dumps(PE, -1))
    old_PN = pickle.loads(pickle.dumps(PN, -1))


    # Create new colored DAG based on move
    match move:
        case 0:
            PE, PN = change_edge_partiton(PE, PN)
            potential_score, *potential_ML_data = score_DAG_color_edit(PE, [sum(x,[]) for x in PN], ML_data)

        case 1:
            PN = change_node_partiton(PN)
            potential_score, *potential_ML_data = score_DAG_color_edit(PE, [sum(x,[]) for x in PN], ML_data)

        case 2:
            # ADD A DID-CHANGE CLAUSE
            A, PE, PN, edge, did_change = add_remove_edge(A, PE, PN)
            if did_change:
                potential_score, *potential_ML_data = score_DAG_edge_edit(A, PE, [sum(x,[]) for x in PN], ML_data, edge)
            else:
                potential_score, potential_ML_data = score, ML_data

    # Metropolis Hastings to accept or reject new colored DAG
    if random.random() <= np.exp(potential_score - score):
        new_score = potential_score
        new_ML_data = potential_ML_data
        failed = 0
    else:
        A = old_A
        PE = old_PE
        PN = old_PN 
        new_score = score
        new_ML_data = ML_data
        failed = 1
 
    return A, PE, PN, new_score, new_ML_data, failed



# For moves
def change_edge_partiton(PE, PN):
    # Find edge to change
    num_edges = sum(len(x) for x in PE)

    if num_edges == 0 or num_edges == 1:
        return PE, PN
    else:
        # Can be optimized to not use edges_in_DAG
        rand_block = random.choices(PE, weights=[len(x) for x in PE])[0]
        edge_to_change = random.choice(rand_block)

    # Find super node with edge[1] in it and remove it from PN
    old_PN_part = None
    old_super_node = None
    for part in PN:
        for super_node in part:
            if edge_to_change[1] in super_node:
                old_PN_part = part
                old_super_node = super_node.copy()
                part.remove(super_node)
                break

    # Make the change of edge color
    old_color = None
    other_colors = []

    for i, part in enumerate(PE):
        if edge_to_change in part:
            old_color = i   
        else:
            other_colors.append(i)

    if len(PE[old_color]) != 1:
        other_colors.append(len(PE))
        PE.append([])

    PE[old_color].remove(edge_to_change)
    new_color = random.choice(other_colors)
    PE[new_color].append(edge_to_change)
    PE = [x for x in PE if len(x)>0]


    # Add back split super nodes to PN and remove duplicates
    new_super_nodes = utils.get_supnodes(PE, num_nodes)
    changed_super_nodes = []
    for new_super_node in new_super_nodes:
        for node in old_super_node:
            if node in new_super_node and new_super_node not in changed_super_nodes:
                changed_super_nodes.append(new_super_node)

    changed_nodes = []
    for new_super_node in changed_super_nodes:
        changed_nodes += new_super_node

    for part in PN:
        for super_node in part:
            for node in changed_nodes:
                if node in super_node:
                    part.remove(super_node)
                    break

    for new_super_node in changed_super_nodes:
        old_PN_part.append(new_super_node)

    PN = [x for x in PN if len(x)>0]

    return PE, PN

def change_node_partiton(PN):
    num_sup_nodes = sum([len(x) for x in PN])
    num = random.randrange(num_sup_nodes)
    i = 0
    for part in PN:
        for supnode in part:
            if i == num:
                supnode_to_change = supnode
            i += 1
    

    old_color = None
    other_colors = []

    for i, part in enumerate(PN):
        if supnode_to_change in part:
            old_color = i   
        else:
            other_colors.append(i)

    if len(PN[old_color]) != 1:
        other_colors.append(len(PN))
        PN.append([])

    PN[old_color].remove(supnode_to_change)
    new_color = random.choice(other_colors)
    PN[new_color].append(supnode_to_change)

    PN = [x for x in PN if len(x)>0]

    return PN

def add_remove_edge(A, PE, PN):
    num_nodes = A.shape[0]
    
    did_change = True
    edge = (random.randrange(num_nodes), random.randrange(num_nodes))

    if A[edge] == 1:
        # Find super node with edge[1] in it and remove it from PN
        old_part = None
        old_super_node = None
        for part in PN:
            for super_node in part:
                if edge[1] in super_node:
                    old_part = part
                    old_super_node = super_node.copy()
                    part.remove(super_node)
                    break


        # Remove edge from A and PE
        A[edge] = 0
        for part in PE:
            if edge in part:
                part.remove(edge)
                break
        PE = [x for x in PE if len(x)>0]


        # Add back split super nodes to PN
        new_super_nodes = utils.get_supnodes(PE, num_nodes)
        changed_super_nodes = []
        for new_super_node in new_super_nodes:
            for node in old_super_node:
                if node in new_super_node and new_super_node not in changed_super_nodes:
                    changed_super_nodes.append(new_super_node)

        for new_super_node in changed_super_nodes:
            old_part.append(new_super_node)

    else:
        tmp = A.copy()
        tmp[edge] = 1
        if utils.is_DAG(tmp):
            A = tmp
            PE.append([edge])
        else:
            did_change = False
    
    return A, PE, PN, edge, did_change



# For DAG heuristic
def score_DAG_full(A, PE, PN_flat):

    # Calculate ML-eval
    edges_ML_ungrouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    omegas_ML_ungrouped = [0] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        beta, ss_res = calc_lstsq(node, tuple(parents))
        edges_ML_ungrouped[parents, node] = beta
        omegas_ML_ungrouped[node] = ss_res / num_samples

    # Block the lambdas as averages
    edges_ML_grouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_ungrouped[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_grouped[edge] = block_lambda

    # Block the omegas as averages
    omegas_ML_grouped = [None] * num_nodes
    for part in PN_flat:
        tot = 0
        for node in part:
            tot += omegas_ML_ungrouped[node]
        block_omega = tot/len(part)
        for node in part:
            omegas_ML_grouped[node] = block_omega
       
    # Calculate BIC 
    x = np.eye(num_nodes)-edges_ML_grouped
    log_likelihood = -math.log(np.prod(omegas_ML_grouped)) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S)
    bic = log_likelihood/2 - BIC_constant * (len(PN_flat) + len(PE))

    return bic, edges_ML_ungrouped, omegas_ML_ungrouped

def score_DAG_edge_edit(A, PE, PN_flat, ML_data, changed_edge):

    # Load old ML-eval
    edges_ML_ungrouped, omegas_ML_ungrouped = ML_data
    edges_ML_ungrouped = edges_ML_ungrouped.copy()
    omegas_ML_ungrouped = omegas_ML_ungrouped.copy()

    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    beta, ss_res = calc_lstsq(active_node, tuple(parents))
    edges_ML_ungrouped[:, active_node] = np.zeros(num_nodes)
    edges_ML_ungrouped[parents, active_node] = beta
    omegas_ML_ungrouped[active_node] = ss_res / num_samples

    # Block the lambdas as averages
    edges_ML_grouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_ungrouped[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_grouped[edge] = block_lambda

    # Block the omegas as averages
    omegas_ML_grouped = [None] * num_nodes
    for part in PN_flat:
        tot = 0
        for node in part:
            tot += omegas_ML_ungrouped[node]
        block_omega = tot/len(part)
        for node in part:
            omegas_ML_grouped[node] = block_omega

    # Calculate BIC 
    x = np.eye(num_nodes)-edges_ML_grouped
    log_likelihood = -math.log(np.prod(omegas_ML_grouped)) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S)
    bic = log_likelihood/2 - BIC_constant * (len(PN_flat) + len(PE))
 
    return bic, edges_ML_ungrouped, omegas_ML_ungrouped

def score_DAG_color_edit(PE, PN_flat, ML_data):

    edges_ML_ungrouped, omegas_ML_ungrouped = ML_data
   
    # Block the lambdas as averages
    edges_ML_grouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_ungrouped[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_grouped[edge] = block_lambda

  
    # Block the omegas as averages
    omegas_ML_grouped = [None] * num_nodes
    for part in PN_flat:
        tot = 0
        for node in part:
            tot += omegas_ML_ungrouped[node]
        block_omega = tot/len(part)
        for node in part:
            omegas_ML_grouped[node] = block_omega
       
    # Calculate BIC
    x = np.eye(num_nodes)-edges_ML_grouped
    log_likelihood = -math.log(np.prod(omegas_ML_grouped)) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S)
    bic = log_likelihood/2 - BIC_constant * (len(PN_flat) + len(PE))

    return bic, edges_ML_ungrouped, omegas_ML_ungrouped


@functools.cache
def calc_lstsq(node, parents):
    if len(parents) == 0:
        b = data[node,:]
        return [], np.dot(b,b)
    
    a = data[parents,:]
    b = data[node,:]
    beta = np.linalg.solve(a @ a.T, a @ b)
    x = b - a.T @ beta
    ss_res = np.dot(x,x)
    return beta, ss_res


def main():
    pass



if __name__ == "__main__":
    main()
