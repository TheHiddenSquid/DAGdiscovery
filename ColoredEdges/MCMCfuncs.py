import copy
import random

import numpy as np
import utils

# Main MCMC functions

def CausalMCMC(data, num_iters = None, move_weights = None, debug = False):
   
    # Check that wieghts are legal
    move_weights = [0.3, 0.3, 0.4]
    move_weights = [1, 1, 1]
    num_nodes = data.shape[1]
    

    # Fully random colored DAG
    A = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    PE = []
    PN = [[[i]] for i in range(num_nodes)]
    

    # Setup for iters
    score = score_DAG(data, A, PE, PN_flat = [sum(x,[]) for x in PN])
    best_A = A.copy()
    best_PE = copy.deepcopy(PE)
    best_PN = copy.deepcopy(PN)
    best_score = score
    best_iter = 0
    num_fails = 0

    # Run MCMC iters    
    for i in range(num_iters):
        A, PE, PN, score, fail = MCMC_iteration(data, A, PE, PN, score, move_weights) 

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
      
def MCMC_iteration(samples, A, PE, PN, score, move_weights):
    # Check what moves are possible and pick one at random
    old_A = A.copy()
    old_PE = copy.deepcopy(PE)
    old_PN = copy.deepcopy(PN)


    moves = ["change_edge_color", "change_node_color",  "change_edge"]
    move = random.choices(moves, weights=move_weights)[0]
    
    # Create new colored DAG based on move
    match move:
        case "change_edge_color":
            PE, PN = change_edge_partiton(A, PE, PN)

        case "change_node_color":
            PN = change_node_partiton(PN)

        case "change_edge":
            A, PE, PN = add_remove_edge(A, PE, PN)
        
    potential_score = score_DAG(samples, A, PE, PN_flat = [sum(x,[]) for x in PN])


    # Metropolis Hastings to accept or reject new colored DAG
    if random.random() <= np.exp(potential_score - score):
        new_score = potential_score
        failed = 0
    else:
        A = old_A
        PE = old_PE
        PN = old_PN 
        new_score = score
        failed = 1
 
    return A, PE, PN, new_score, failed



# For moves
def change_edge_partiton(A, PE, PN):
    # Find edge to change
    num_nodes = A.shape[0]
    edges_in_DAG, _, _ = utils.get_sorted_edges(A)

    if len(edges_in_DAG) == 0 or len(edges_in_DAG) == 1:
        return PE, PN
    else:
        edge_to_change = random.choice(edges_in_DAG)

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
    
    return A, PE, PN



# For DAG heuristic
def score_DAG(data, A, PE, PN_flat):
    data = data.T
    
    global num_nodes
    global num_samples
    global BIC_constant
    num_nodes = data.shape[0]
    num_samples = data.shape[1]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval of the different lambdas
    edges_ML_tmp = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = utils.get_parents(i, A)
        ans = np.linalg.lstsq(data[parents,:].T, data[i,:].T, rcond=None)[0]
        edges_ML_tmp[parents, i] = ans

    # Block the lambdas as averages
    edges_ML_real = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_tmp[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_real[edge] = block_lambda

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * num_nodes
    bic_decomp = [0] * num_nodes

    for i, part in enumerate(PN_flat):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, A)
            tot += np.dot(x:=(data[node,:] - edges_ML_tmp[parents,node].T @ data[parents,:]), x)
        omegas_ML[i] = tot / (num_samples * len(part))


        # Calculate BIC
        bic_decomp[i] = -len(part) * (np.log(omegas_ML[i]) + 1)
    
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (np.sum(A) + len(PN_flat) + len(PE))

    return bic



def main():
    pass



if __name__ == "__main__":
    main()
