import random
import numpy as np
import networkx as nx
import ges
import copy
from generateDAGs import generate_colored_DAG

# Main MCMC function

def MCMC(samples, num_iters, move_list = None, start_from_GES = False):
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
            if is_DAG(new_edges):
                A = new_edges
                continue

            new_edges = A.copy()
            new_edges[edge[1], edge[0]] = 1
            if is_DAG(new_edges):
                A = new_edges
                continue

            raise ValueError("Could not create DAG")

        # Every node has its own color
        partition = [[i] for i in range(num_nodes)]

    else:
        # Fully random colored DAG
        partition, A, _ = generate_colored_DAG(num_nodes, num_nodes, 0.5)
        A = np.array(A != 0, dtype="int")
    



    # Setup for iters
    sorted_edges = get_sorted_edges(A)
    bic = score_DAG(samples, A, partition)


    # Space for best found DAG
    best_A = A.copy()
    best_partition = partition.copy()
    best_bic = bic
    best_iter = 0


    # Run MCMC iters    
    for i in range(num_iters):

        A, partition, bic, sorted_edges = MCMC_iteration(samples, A, partition, bic, sorted_edges, move_list)

        if bic > best_bic:
            best_A = A.copy()
            best_partition = partition.copy()
            best_bic = bic
            best_iter = i

    return best_A, best_partition, best_bic, best_iter





def MCMC_iteration(samples, edge_array, partition, bic, sorted_edges, possible_moves=None):
    old_edge_array = edge_array.copy()
    old_partition = copy.deepcopy(partition)
    old_bic = bic


    potential_edge_array = edge_array.copy()
    potential_partition = copy.deepcopy(partition)

    
    edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs = sorted_edges

    m = len(edges_in_DAG)                   # Number of current edges
    no_nodes = np.shape(edge_array)[0]      # Number of current noded
    no_colors = len(potential_partition)    # Number of possible colors

    if possible_moves == None:
        moves = [ "change_color", "add_edge", "remove_edge"]
    else:
        moves = possible_moves

    if m == 0:
        moves.remove("remove_edge")
        print("could not remove edge")

    if len(edges_giving_DAGs) == 0:
        moves.remove("add_edge")
        print("could not add edge")

    move = random.choice(moves)



    if move == "change_color":
        node = random.randrange(no_nodes)
        for i, part in enumerate(potential_partition):
            if node in part:
                current_color = i
                break
        potential_partition[current_color].remove(node)

        new_color = current_color
        while new_color == current_color:
            new_color = random.randrange(no_colors)
        potential_partition[new_color].append(node)

        # Relative probability of jumping back
        q_quotient = 1


    if move == "add_edge":
        k_old = len(edges_giving_DAGs)  # Number of edges that can be added

        edge = random.choice(edges_giving_DAGs)
        potential_edge_array[edge] = 1

        # Relative probability of jumping back
        q_quotient = k_old / (m+1)
    

    if move == "remove_edge":
        edge = random.choice(edges_in_DAG)
        potential_edge_array[edge] = 0

        # Relative probability of jumping back
        potential_sorted_edges = update_sorted_edges_REMOVE(potential_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        k_new = len(potential_sorted_edges[1])

        q_quotient = m / k_new


    # Metropolis hastings algorithm

    potential_bic = score_DAG(samples, potential_edge_array, potential_partition)
    
    if random.random() <= (potential_bic / old_bic) * q_quotient:
        new_edge_array = potential_edge_array
        new_partition = potential_partition
        new_bic = potential_bic

        if move == "change_color":
            new_sorted_edges = sorted_edges
        elif move == "add_edge":
            new_sorted_edges = update_sorted_edges_ADD(new_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        elif move == "remove_edge":
            new_sorted_edges = potential_sorted_edges
    else:
        new_edge_array = old_edge_array
        new_partition = old_partition
        new_bic = old_bic
        new_sorted_edges = sorted_edges

    return new_edge_array, new_partition, new_bic, new_sorted_edges


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
            if is_DAG(tmp_edge_array):
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
        if is_DAG(tmp_edge_array):
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
        if is_DAG(tmp_edge_array):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]


def is_DAG(A):
    numnodes = A.shape[0]
    P = np.linalg.matrix_power(A, numnodes-1)
    if np.argmax(P) != 0:
        return False
    return not P[0,0]



# For DAG heuristic
# Speedup by NOT doing edges of only color was changed and vice versa

def score_DAG(samples, edge_array, partition):
    samples = np.transpose(samples)

    num_nodes = samples.shape[0]
    num_samples = samples.shape[1]
    
    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype="float")
    for i in range(num_nodes):
        parents = get_parents(i, edge_array)
        ans = np.linalg.lstsq(np.transpose(samples[parents,:]), np.transpose(samples[i,:]), rcond=None)[0]
        edges_ML[parents, i] = ans


    # Calculate ML-eval of the different color omegas
    omegas_for_color = [None] * len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = get_parents(node, edge_array)
            tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
        omegas_for_color[i] = tot / (num_samples * len(part))


    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(get_parents(x, edge_array)) for x in part)
    bic = tot / 2


    return bic

def get_parents(node, edge_array):
    parents = []
    n = np.shape(edge_array)[0]
    for i in range(n):
        if edge_array[i, node] == 1:
            parents.append(i)
    return parents


def generate_partition(no_nodes, no_colors):
    partition = [[] for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].append(node)
    return partition

def main():
    pass



if __name__ == "__main__":
    main()