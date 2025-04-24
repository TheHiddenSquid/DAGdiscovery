import copy
import pickle
import random

import numpy as np
import utils

# Main MCMC functions

def CausalGreedySearch(data, num_waves = 5):

    # Setup constants
    global num_samples
    global num_nodes
    global data_S
    num_samples = data.shape[0]
    num_nodes = data.shape[1]
    data_S = data.T @ data / num_samples
   

    # Setup for iters
    best_A = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    best_PE = []
    best_PN = [[[i]] for i in range(num_nodes)]
    best_score = -np.inf

    edge_probs = list(np.linspace(0,1,num_waves))

    for i in range(num_waves):
        # Generate random colored DAG to start with
        PE, _, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_nodes**2, num_nodes, edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)
        supnodes = utils.get_supnodes(PE, num_nodes)
        PN = [[] for _ in range(num_nodes)]
        for supnode in supnodes:
            block = random.choice(PN)
            block.append(supnode)
        PN = [x for x in PN if len(x)>0]

     
        # Setup for search
        score, *ML_data = score_DAG_full(data, A, PE, PN_flat = [sum(x,[]) for x in PN])
    
        done = False
        j=0
        while not done:
            print(j)
            A, PE, PN, score, ML_data, done = greedy_iteration(data, A, PE, PN, score, ML_data) 

            if score >= best_score:
                best_A = A.copy()
                best_PE = copy.deepcopy(PE)
                best_PN = copy.deepcopy(PN)
                best_score = score
            j+=1

    # Flatten node partition
    best_PN = [sum(x,[]) for x in best_PN]

    return best_A, best_PE, best_PN, best_score
      
def greedy_iteration(samples, A, PE, PN, score, ML_data):
    old_PE = pickle.dumps(PE, -1)
    old_PN = pickle.dumps(PN, -1)

    best_A = A.copy()
    best_PE = pickle.loads(old_PE)
    best_PN = pickle.loads(old_PN)
    best_score = score
    best_ML_data = ML_data

    num_trys = 100
    for _ in range(num_trys):
        pot_A = A.copy()
        pot_PE = pickle.loads(old_PE)
        pot_PN = pickle.loads(old_PN)
        pot_score = score
        pot_ML_data = ML_data
        pot_PE, pot_PN = change_edge_partiton(pot_PE, pot_PN)
        pot_score, *pot_ML_data = score_DAG_color_edit(samples, pot_A, pot_PE, [sum(x,[]) for x in pot_PN], pot_ML_data)

        if pot_score > best_score:
            best_A = pot_A
            best_PE = pot_PE
            best_PN = pot_PN
            best_score = pot_score
            best_ML_data = pot_ML_data


    for _ in range(num_trys):
        pot_A = A.copy()
        pot_PE = pickle.loads(old_PE)
        pot_PN = pickle.loads(old_PN)
        pot_score = score
        pot_ML_data = ML_data
        pot_PN = change_node_partiton(pot_PN)
        pot_score, *pot_ML_data = score_DAG_color_edit(samples, pot_A, pot_PE, [sum(x,[]) for x in pot_PN], pot_ML_data)

        if pot_score > best_score:
            best_A = pot_A
            best_PE = pot_PE
            best_PN = pot_PN
            best_score = pot_score
            best_ML_data = pot_ML_data

    for _ in range(num_trys):
        pot_A = A.copy()
        pot_PE = pickle.loads(old_PE)
        pot_PN = pickle.loads(old_PN)
        pot_score = score
        pot_ML_data = ML_data
        pot_A, pot_PE, pot_PN = add_remove_edge(pot_A, pot_PE, pot_PN)
        pot_score, *pot_ML_data = score_DAG_full(samples, pot_A, pot_PE, [sum(x,[]) for x in pot_PN])

        if pot_score > best_score:
            best_A = pot_A
            best_PE = pot_PE
            best_PN = pot_PN
            best_score = pot_score
            best_ML_data = pot_ML_data


    # Metropolis Hastings to accept or reject new colored DAG
    if best_score > score:
        new_A = best_A
        new_PE = best_PE
        new_PN = best_PN
        new_score = best_score
        new_ML_data = best_ML_data
        
        return new_A, new_PE, new_PN, new_score, new_ML_data, False
    
    else:
        return A, PE, PN, score, ML_data, True



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
def score_DAG_full(data, A, PE, PN_flat):
    data = data.T
    global data_S
    global num_nodes
    global num_samples

    # Calculate ML-eval of the different lambdas
    edges_ML_ungrouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        parents = utils.get_parents(i, A)
        edges_ML_ungrouped[parents, i] = np.linalg.lstsq(data[parents,:].T, data[i,:].T, rcond=None)[0]

    # Block the lambdas as averages
    edges_ML_grouped = np.zeros((num_nodes,num_nodes), dtype=np.float64)
    for block in PE:
        tot = 0
        for edge in block:
            tot += edges_ML_ungrouped[edge]
        block_lambda = tot/len(block)
        for edge in block:
            edges_ML_grouped[edge] = block_lambda

    # Calculate ML-eval of the different color omegas
    omegas_ML_ungrouped = [None] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        omegas_ML_ungrouped[node] = np.dot(x:=(data[node,:] - edges_ML_ungrouped[parents,node].T @ data[parents,:]), x) / num_samples

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
    log_likelihood = (num_samples/2) * (-np.log(np.prod(omegas_ML_grouped)) + np.log(np.linalg.det(x:=(np.eye(num_nodes)-edges_ML_grouped))**2) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S))

    bic = (1/num_samples) * (log_likelihood - (np.log(num_samples)/2) * (np.sum(A) + len(PN_flat) + len(PE)))

    return bic, edges_ML_ungrouped, omegas_ML_ungrouped

def score_DAG_color_edit(data, A, PE, PN_flat, ML_data):
    data = data.T
    global data_S
    global num_nodes
    global num_samples

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
    log_likelihood = (num_samples/2) * (-np.log(np.prod(omegas_ML_grouped)) + np.log(np.linalg.det(x:=(np.eye(num_nodes)-edges_ML_grouped))**2) - np.trace(x @ np.diag([1/w for w in omegas_ML_grouped]) @ x.T @ data_S))

    bic = (1/num_samples) * (log_likelihood - (np.log(num_samples)/2) * (np.sum(A) + len(PN_flat) + len(PE)))

    return bic, edges_ML_ungrouped, omegas_ML_ungrouped


def main():
    pass



if __name__ == "__main__":
    main()
