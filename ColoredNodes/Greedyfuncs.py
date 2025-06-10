import copy
import math

import numpy as np
import utils

# Main functions

def CausalGreedySearch(samples, num_waves = 5):
    
    # Setup global variables
    global my_data
    global num_nodes
    global num_samples
    global BIC_constant

    my_data = samples.T
    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes))
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

        bic, ML_data = score_DAG_full(A, P)
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            A, P, bic, ML_data, sorted_edges, done = Greedyiteration(A, P, bic, ML_data, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = bic

    CPDAG_A = utils.getCPDAG(best_A, best_P)
    return CPDAG_A, best_P, best_bic
    
def Greedyiteration(A, P, bic, ML_data, sorted_edges):
    best_move = None
    best_A = None
    best_P = None
    best_bic = bic
    best_ML_data = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges



    # Check all neighboring colorings
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


            potential_bic, potential_ML_data = score_DAG_color_edit(A, P, ML_data, old_color, new_color)

            if potential_bic > best_bic:
                best_P = copy.deepcopy(P)
                best_bic = potential_bic
                best_ML_data = potential_ML_data
                best_move = "change_color"


            P[new_color].remove(node)
        P[old_color].add(node)


    # Check all potential edge adds
    for edge in edges_giving_DAGs:
        A[edge] = 1

        potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)

        if potential_bic > best_bic:
            best_A = A.copy()
            best_bic = potential_bic
            best_ML_data = potential_ML_data
            best_move = "add_edge"
            best_saved_edge = edge
    
        A[edge] = 0


    # Check all potential edge removals
    for edge in edges_in_DAG:
        A[edge] = 0
        
        potential_bic, potential_ML_data = score_DAG_edge_edit(A, P, ML_data, edge)

        if potential_bic > best_bic:
            best_A = A.copy()
            best_bic = potential_bic
            best_ML_data = potential_ML_data
            best_move = "remove_edge"
            best_saved_edge = edge
        
        A[edge] = 1



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
            new_sorted_edges = update_sorted_edges_ADD(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "remove_edge":
            new_A = best_A
            new_P = P
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)

        
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
        a = my_data[parents,:]
        b = my_data[node,:]
        beta = np.linalg.solve(a @ a.T, a @ b)
        x = b - a.T @ beta
        ss_res = np.dot(x,x)
        omegas_ML[node] = ss_res / num_samples


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (math.log(block_omega) + 1)
    
    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))
    
    return bic, [omegas_ML, bic_decomp]

def score_DAG_color_edit(A, P, ML_data, old_color, new_color):
    
    # ML data is the same
    omegas_ML, bic_decomp = ML_data
    omegas_ML = omegas_ML.copy()
    bic_decomp = bic_decomp.copy()

    # Update decomposed BIC
    for block_index in [old_color, new_color]:
        block = P[block_index]
        if len(block) == 0:
             bic_decomp[block_index] = 0
             continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[block_index] = -len(block) * (math.log(block_omega) + 1)

    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))

    return bic, [omegas_ML, bic_decomp]

def score_DAG_edge_edit(A, P, ML_data, changed_edge):

    # Get old ML-eval
    omegas_ML, bic_decomp = ML_data
    omegas_ML = omegas_ML.copy()
    bic_decomp = bic_decomp.copy()

    
    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    a = my_data[parents,:]
    b = my_data[active_node,:]
    beta = np.linalg.solve(a @ a.T, a @ b)
    x = b - a.T @ beta
    ss_res = np.dot(x,x)
    omegas_ML[active_node] = ss_res / num_samples

   
    # Update decomposed BIC
    for i, block in enumerate(P):
        if active_node in block:
            tot = 0
            for node in block:
                tot += omegas_ML[node]
            omega = tot / len(block)

            bic_decomp[i] = -len(block) * (math.log(omega) + 1)


    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))

    return bic, [omegas_ML, bic_decomp]


def main():
    pass



if __name__ == "__main__":
    main()

