import random
import numpy as np
import ges
import copy
from collections import defaultdict
import utils

# Main MCMC function

def CausalMCMC(samples, num_iters, mode = "bic", start_from_GES = False, move_weights = None, start_edge_array = None, start_partition = None, debug = False):

    # Check that mode is legal
    if mode not in ["bic", "map"]:
        raise ValueError("Mode not supported")
    
    # Check that wieghts are legal
    if move_weights is not None:
        if len(move_weights) != 3:
            raise ValueError("Lenght of weights must be 3")
        p_change_color, p_add, p_remove = move_weights
        if p_change_color<0 or p_add<0 or p_remove<0 or not np.isclose(sum(move_weights),1):
            raise ValueError("Invalid move probabilities")
    else:
        move_weights = [1/3]*3

    
    # Other settings
    num_nodes = samples.shape[1]

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
        partition = [[i] for i in range(num_nodes)]

    else:
        # Fully random colored DAG
        partition, A, _ = utils.generate_colored_DAG(num_nodes, num_nodes, 0.5)
        A = np.array(A != 0, dtype="int")
        partition = [[i] for i in range(num_nodes)]
    

    if start_partition is not None:
        partition = copy.deepcopy(start_partition)

    if start_edge_array is not None:
        A = start_edge_array


    # Setup for iters
    sorted_edges = get_sorted_edges(A)
    bic = score_DAG(samples, A, partition)


    if mode == "bic":
        best_A = A.copy()
        best_partition = copy.deepcopy(partition)
        best_bic = bic[0]
        best_iter = 0
        num_fails = 0

        # Run MCMC iters    
        for i in range(num_iters):
            A, partition, bic, sorted_edges, fail = MCMC_iteration(samples, A, partition, bic, sorted_edges, move_weights)    
            if bic[0] > best_bic:
                best_A = A.copy()
                best_partition = utils.sorted_partition(partition)
                best_bic = bic[0]
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
        for i in range(num_iters):

            A, partition, bic, sorted_edges, fail = MCMC_iteration(samples, A, partition, bic, sorted_edges, move_weights)

            h1 = A.tobytes()
            h2 = tuple(tuple(x) for x in utils.sorted_partition(partition))

            cashe[(h1,h2)] += 1
            num_fails += fail

        most_visited = max(cashe, key=cashe.get)
        num_visits = cashe[most_visited]

        best_A = most_visited[0]
        best_A = np.frombuffer(best_A, dtype="int")
        best_A = np.reshape(best_A, (num_nodes,num_nodes))

        best_partition = [list(x) for x in most_visited[1]]

        if debug: 
            return best_A, best_partition, num_visits, num_fails
        else:
            return best_A, best_partition, num_visits
    


def MCMC_iteration(samples, edge_array, partition, bic, sorted_edges, move_weights):
    
    edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs = sorted_edges
    num_edges = len(edges_in_DAG)


    # Check what moves are possible and pick one at random
    p_change_color, p_add, p_remove = move_weights
    
    moves = [ "change_color", "add_edge", "remove_edge"]
    weights = move_weights.copy()

    if num_edges == 0:
        weights[2] = 0  
    elif len(edges_giving_DAGs) == 0 :
        weights[1] = 0

    move = random.choices(moves, weights = weights, k = 1)[0]


    # Create new colored DAG based on move
    if move == "change_color":
        potential_edge_array = edge_array
        potential_partition, node, old_color, new_color = change_partiton(partition)

        q_quotient = 1
        potential_bic = score_DAG_color_edit(samples, potential_edge_array, potential_partition, [bic[1], bic[2], [node, old_color, 
        new_color]])


    if move == "add_edge":
        potential_partition = partition

        potential_edge_array = edge_array.copy()
        old_num_addible_edges = len(edges_giving_DAGs)  # Number of edges that can be added

        edge = random.choice(edges_giving_DAGs)
        potential_edge_array[edge] = 1

        q_quotient = (p_remove*old_num_addible_edges) / (p_add*(num_edges+1))
        potential_bic = score_DAG_edge_edit(samples, potential_edge_array, potential_partition, [bic[1], bic[2], edge[1]])
    

    if move == "remove_edge":
        potential_partition = partition

        potential_edge_array = edge_array.copy()
        edge = random.choice(edges_in_DAG)
        potential_edge_array[edge] = 0

        potential_sorted_edges = update_sorted_edges_REMOVE(potential_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        new_num_addible_edges = len(potential_sorted_edges[1])

        q_quotient = (p_add*num_edges) / (p_remove*new_num_addible_edges)
        potential_bic = score_DAG_edge_edit(samples, potential_edge_array, potential_partition, [bic[1], bic[2], edge[1]])


    # Metropolis Hastings to accept or reject new colored DAG
    if random.random() <= np.exp(potential_bic[0] - bic[0]) * q_quotient:
        new_edge_array = potential_edge_array
        new_partition = potential_partition
        new_bic = potential_bic

        if move == "change_color":
            new_sorted_edges = sorted_edges
        elif move == "add_edge":
            new_sorted_edges = update_sorted_edges_ADD(new_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        elif move == "remove_edge":
            new_sorted_edges = potential_sorted_edges

        failed = 0
    else:
        new_edge_array = edge_array
        new_partition = partition
        new_bic = bic
        new_sorted_edges = sorted_edges
        failed = 1

    return new_edge_array, new_partition, new_bic, new_sorted_edges, failed



# For moves
def change_partiton(partition):
    num_nodes = sum(len(x) for x in partition)

    node_to_change = random.randrange(num_nodes)
    old_color = None
    other_colors = []
    found_old_color = False

    for i, part in enumerate(partition):
        if (not found_old_color) and (node_to_change in part):
            found_old_color = True
            old_color = i
        elif len(part) != 0:
            other_colors.append(i)
        else:
            empty_color = i

    if len(partition[old_color]) != 1:
        other_colors.append(empty_color)

    changed_partition = copy.deepcopy(partition)
    changed_partition[old_color].remove(node_to_change)

    new_color = random.choice(other_colors)
    changed_partition[new_color].append(node_to_change)
    return changed_partition, node_to_change, old_color, new_color

    

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
    omegas_for_color = [None] * len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, edge_array)
            tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
        omegas_for_color[i] = tot / (num_samples * len(part))


    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    
    bic = tot / 2
    bic = bic - np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_for_color]

def score_DAG_color_edit(samples, edge_array, partition, last_change_data):
    samples = np.transpose(samples)
    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)
    

    # Edge ML is the same
    edges_ML = last_change_data[0]

    # Node ML needs local update
    omegas_for_color = last_change_data[1].copy()
    
    node, old_color, new_color = last_change_data[2]

    parents = utils.get_parents(node, edge_array)
    node_ml_contribution = np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2

    if len(partition[old_color]) == 0:
        omegas_for_color[old_color] = None
    else:
        tot = omegas_for_color[old_color] * num_samples * (len(partition[old_color]) + 1)
        tot -= node_ml_contribution
        omegas_for_color[old_color] = tot / (num_samples * len(partition[old_color]))
    
    if len(partition[new_color]) == 1:
        tot = 0
    else:
        tot = omegas_for_color[new_color] * num_samples * (len(partition[new_color]) - 1)
    tot += node_ml_contribution
    omegas_for_color[new_color] = tot / (num_samples * len(partition[new_color]))



    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    bic = tot / 2
    bic = bic - np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_for_color]

def score_DAG_edge_edit(samples, edge_array, partition, last_change_data):
    samples = np.transpose(samples)

    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)
    

    # Calculate ML-eval of the different lambdas
    edges_ML = last_change_data[0].copy()
    node_with_new_parents = last_change_data[2]
    parents = utils.get_parents(node_with_new_parents, edge_array)
    ans = np.linalg.lstsq(np.transpose(samples[parents,:]), np.transpose(samples[node_with_new_parents,:]), rcond=None)[0]
    edges_ML[parents, node_with_new_parents] = ans


    # Calculate ML-eval of the different color omegas
    omegas_for_color = last_change_data[1].copy()

    for i, part in enumerate(partition):
        if node_with_new_parents in part:
            current_color = i
            break

    part = partition[current_color]
    tot = 0
    for node in part:
        parents = utils.get_parents(node, edge_array)
        tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
    omegas_for_color[current_color] = tot / (num_samples * len(part))



    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    bic = tot / 2
    bic = bic - np.log(num_samples)/(num_samples*2) * num_colors


    return [bic, edges_ML, omegas_for_color]




def main():
    pass



if __name__ == "__main__":
    main()
