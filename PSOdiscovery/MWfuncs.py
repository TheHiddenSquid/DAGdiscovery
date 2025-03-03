import random
import numpy as np
import copy
import utils

# Main MCMC function

def CausalMW(samples, num_iters):

    # Other settings
    num_nodes = samples.shape[1]


    global_best_bic = -np.infty
    global_best_A = None
    global_best_P = None
    

    P, A, _ = utils.generate_colored_DAG(num_nodes, num_nodes, 0.5)
    A = np.array(A != 0, dtype="int")
    score_info = score_DAG(samples, A, P)
       

    if score_info[0] > global_best_bic:
        global_best_bic = score_info[0]
        global_best_A = A.copy()
        global_best_P = copy.deepcopy(P)


    alledgemoves = [(k,i,j) for i in range(num_nodes) for j in range(i, num_nodes) for k in range(3) if i != j]

    allcolormoves = [(node, color) for node in range(num_nodes) for color in range(num_nodes)]
    allmoves = alledgemoves + allcolormoves
    weights = [1] * len(allmoves)

    for i in range(num_iters):
        print(weights)
        move = random.choices(allmoves, weights=weights)[0]
        
        if len(move) == 3:
            edge = (move[1], move[2])
            rev = (move[2], move[1])
            if move[0] == 0:
                A[edge] = 1
                A[rev] = 0
                score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], edge])  
            elif move[0] == 1:
                A[edge] = 0
                A[rev] = 1
                score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], rev]) 
            else:
                if A[edge] == 1:
                    A[edge] = 0
                    score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], edge])
                else:
                    A[rev] = 0
                    score_info = score_DAG_edge_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], rev])
            
            
        if len(move) == 2:
            node = move[0]
            new_color = move[1]
            for j, part in enumerate(P):
                if node in part:
                    old_color = j
                    break
            P[old_color].remove(node)
            P[new_color].append(node)

            score_info = score_DAG_color_edit(samples, A, P, [score_info[1], score_info[2], score_info[3], [node, old_color, new_color]])        
        
        weights[allmoves.index(move)] *= score_info[0]
        
        if score_info[0] > global_best_bic:
            global_best_bic = score_info[0]
            global_best_A = A.copy()
            global_best_P = copy.deepcopy(P)


    return global_best_A, global_best_P, global_best_bic

    
    


# For DAG heuristic

def score_DAG(samples, edge_array, partition):
    samples = np.transpose(samples)

    num_nodes = samples.shape[0]
    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)

    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype="float")
    for i in range(num_nodes):
        parents = utils.get_parents(i, edge_array)
        ans = np.linalg.lstsq(np.transpose(samples[parents,:]), np.transpose(samples[i,:]), rcond=None)[0]
        edges_ML[parents, i] = ans

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, edge_array)
            tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
        omegas_ML[i] = tot / (num_samples * len(part))


    # Calculate BIC

    bic_decomp = [0]*len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        bic_decomp[i] = -len(part) * np.log(omegas_ML[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    
    bic = sum(bic_decomp) / 2
    bic = bic - np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_color_edit(samples, edge_array, partition, last_change_data):
    samples = np.transpose(samples)
    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)
    

    # Edge ML is the same
    edges_ML = last_change_data[0]

    # Node ML needs local update
    omegas_ML = last_change_data[1].copy()
    
    node, old_color, new_color = last_change_data[3]

    parents = utils.get_parents(node, edge_array)
    node_ml_contribution = np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2

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
        bic_decomp[i] = -len(part) * np.log(omegas_ML[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    
    bic = sum(bic_decomp) / 2
    bic -= np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_ML, bic_decomp]

def score_DAG_edge_edit(samples, edge_array, partition, last_change_data):
    samples = np.transpose(samples)

    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)
    

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
    new_ml = np.linalg.lstsq(np.transpose(samples[new_parents,:]), np.transpose(samples[new_child,:]), rcond=None)[0]
    edges_ML[new_parents, new_child] = new_ml


    # Calculate ML-eval of the different color omegas
    omegas_ML = last_change_data[1].copy()

    for i, part in enumerate(partition):
        if new_child in part:
            current_color = i
            break

    part = partition[current_color]
    tot = omegas_ML[current_color] * num_samples * len(part)
    tot -= np.linalg.norm(samples[new_child,:]-np.matmul(np.transpose(old_ml), samples[old_parents,:]))**2
    tot += np.linalg.norm(samples[new_child,:]-np.matmul(np.transpose(new_ml), samples[new_parents,:]))**2
    omegas_ML[current_color] = tot / (num_samples * len(part))



    # Calculate BIC
    bic_decomp = last_change_data[2].copy()

    part = partition[current_color]
    bic_decomp[current_color] = -len(part) * np.log(omegas_ML[current_color]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    
    bic = sum(bic_decomp) / 2
    bic -= np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_ML, bic_decomp]




def main():
    pass



if __name__ == "__main__":
    main()
