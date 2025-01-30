import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
from scipy import stats
import ges
import time

def generate_colored_DAG(no_nodes, no_colors, edge_probability):
    
    # Add edges and make sure it is a DAG
    G = nx.DiGraph()
    nodes = [*range(no_nodes)]
    random.shuffle(nodes)

    for node in nodes:
        G.add_node(node)
        others = [*range(no_nodes)]
        others.remove(node)
        random.shuffle(others)

        for other in others:
            if random.random() < edge_probability:
                if random.random() < 0.5:
                    G.add_edge(node, other)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(node, other)
                else:
                    G.add_edge(other, node)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(other, node)


    # Create partition for colors
    partition = generate_partition(no_nodes, no_colors)

    # Generate lambda matrix
    lambda_matrix = nx.adjacency_matrix(G).todense().astype("float64")
    for i in range(no_nodes):
        for j in range(no_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(-1,1)

    # Generate omega matrix
    choices = [random.random() for _ in range(no_colors)]
    omega_matrix = [None] * no_nodes
    for i, part in enumerate(partition):
        for node in part:
            omega_matrix[node] = choices[i]


    return partition, lambda_matrix, omega_matrix

def generate_sample(size, lambda_matrix, omega_matrix):
    no_nodes = len(omega_matrix)

    errors = np.zeros((no_nodes,size), dtype="float64")
    for i, omega in enumerate(omega_matrix):
        rv = stats.norm(scale = omega)
        errors[i,:] = rv.rvs(size=size)

    X = np.transpose(np.linalg.inv(np.identity(no_nodes) - lambda_matrix))
    
    sample = np.zeros((no_nodes,size), dtype="float64")
    for i in range(size):
        sample[:,i] = np.matmul(X, errors[:,i])
    
    return np.transpose(sample)


def generate_partition(no_nodes, no_colors):
    partition = [[] for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].append(node)
    return partition

def generate_color_map(partition):
    if len(partition) > 10:
        raise ValueError("Too many colors needed for color map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink"]
    length = sum([len(x) for x in partition])
    color_map = [None] * length

    for i, part in enumerate(partition):
        for node in part:
            color_map[node] = colors[i]

    return color_map


def MCMC_iteration(edge_array, partition, samples, sorted_edges):
    old_edge_array = edge_array.copy()
    old_partition = partition.copy()

    potential_edge_array = edge_array.copy()
    potential_partition = partition.copy()

    
    edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs = sorted_edges

    m = len(edges_in_DAG)                   # Number of current edges
    no_nodes = np.shape(edge_array)[0]      # Number of current noded
    no_colors = len(potential_partition)    # Number of possible colors


    moves = [ "change_color", "add_edge", "remove_edge"]

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
        new_edges_giving_DAGs = update_addable_edges_REMOVED(potential_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        k_new = len(new_edges_giving_DAGs[1])

        q_quotient = m / k_new


    # Metropolis hastings algorithm

    potential_bic = score_DAG(samples, potential_edge_array, potential_partition)
    old_bic = score_DAG(samples, old_edge_array, old_partition)

    if random.random() <= (potential_bic / old_bic) * q_quotient: 
        new_edge_array = potential_edge_array
        new_partition = potential_partition
        new_bic = potential_bic

        if move == "change_color":
            new_sorted_edges = sorted_edges
        elif move == "add_edge":
            new_sorted_edges = update_addable_edges_ADD(new_edge_array, sorted_edges[0], sorted_edges[1], sorted_edges[2], edge)
        elif move == "remove_edge":
            new_sorted_edges = new_edges_giving_DAGs
    else:
        new_edge_array = old_edge_array
        new_partition = old_partition
        new_bic = old_bic
        new_sorted_edges = sorted_edges


    return new_edge_array, new_partition, new_bic, new_sorted_edges


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
            G = nx.DiGraph(tmp_edge_array)
            if nx.is_directed_acyclic_graph(G):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_addable_edges_REMOVED(edge_array, edges_in, addable_edges, not_addable_edges, removed_edge):
    
    tmp_edge_array = edge_array.copy()

    edges_in_DAG = edges_in.copy()
    edges_in_DAG.remove(removed_edge)

    edges_giving_DAGs = addable_edges.copy() + [removed_edge]
    edges_not_giving_DAGs = []

    for edge in not_addable_edges:
        tmp_edge_array[edge] = 1
        G = nx.DiGraph(tmp_edge_array)
        if nx.is_directed_acyclic_graph(G):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_addable_edges_ADD(edge_array, edges_in, addable_edges, not_addable_edges, added_edge):

    tmp_edge_array = edge_array.copy()

    edges_in_DAG = edges_in.copy() + [added_edge]
    edges_giving_DAGs = []
    edges_not_giving_DAGs = not_addable_edges.copy()

    for edge in addable_edges:
        if edge == added_edge:
            continue

        tmp_edge_array[edge] = 1
        G = nx.DiGraph(tmp_edge_array)
        if nx.is_directed_acyclic_graph(G):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]



def score_DAG(samples, edge_array, partition):
    samples = np.transpose(samples)

    n = sum(len(x) for x in partition)


    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((n,n), dtype="float")
    for i in range(n):
        parents = get_parents(i, edge_array)
        ans = np.linalg.solve(np.matmul(samples[parents,:],np.transpose(samples[parents,:])), np.matmul(samples[parents,:],np.transpose(samples[i,:])))
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
        omegas_for_color[i] = tot / (n * len(part))

    omegas_ML = [None] * n
    for i, part in enumerate(partition):
        for node in part:
            omegas_ML[node] = omegas_for_color[i]


    # Calculate BIC
    tot = 0
    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot += -len(part) * np.log(omegas_for_color[i]) - len(part) - np.log(n) * (1/n) * sum(len(get_parents(x, edge_array)) for x in part)
    bic = tot / 2


    return bic

def get_parents(node, edge_array):
    parents = []
    n = np.shape(edge_array)[0]
    for i in range(n):
        if edge_array[i, node] == 1:
            parents.append(i)
    return parents


def calc_SHD(edge_array1, edge_array2):
    return np.sum(np.abs(edge_array1-edge_array2))

def calc_partition_distance(partition1, partition2):
    pa1 = partition1.copy()
    pa2 = partition2.copy()

    n = sum(len(x) for x in pa1)
    parts = max(len(pa1), len(pa2))

    pa1 += [[]]*(parts - len(pa1))
    pa2 += [[]]*(parts - len(pa2))

    cost_matrix = np.zeros((parts, parts), dtype="int")
    for i in range(parts):
        for j in range(parts):
            cost_matrix[i,j] = len(set(pa1[i]).intersection(set(pa2[j])))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return n - cost_matrix[row_ind, col_ind].sum()


def main():
    no_nodes = 6
    no_colors = 3
    edge_probability = 0.3
    sample_size = 1000
    MCMC_iterations = 10000
    start_with_GES_DAG = False

    real_partition, real_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    

    # MCMC setup
    if start_with_GES_DAG:

        # Take an initial DAG from the given GES CPDAG
        initial_edge_array =  GES_edge_array.copy()
        double = []
        for i in range(no_nodes):
            for j in range(i+1, no_nodes):
                if initial_edge_array[i,j] == 1 and initial_edge_array[j,i] == 1:
                    double.append((i,j))
                    initial_edge_array[i,j] = 0
                    initial_edge_array[j,i] = 0

        for edge in double:
            new_edges = initial_edge_array.copy()
            new_edges[edge[0], edge[1]] = 1
            G = nx.DiGraph(new_edges)
            if nx.is_directed_acyclic_graph(G):
                initial_edge_array = new_edges
                continue

            new_edges = initial_edge_array.copy()
            new_edges[edge[1], edge[0]] = 1
            G = nx.DiGraph(new_edges)
            if nx.is_directed_acyclic_graph(G):
                initial_edge_array = new_edges
                continue

            raise ValueError("Could not create DAG")

        # Random initial coloring guess
        initial_partition = generate_partition(no_nodes, no_nodes)

    else:
        # Fully random colored DAG
        initial_partition, initial_edge_array, _ = generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
    

    

    # RUN MCMC

    current_edge_array = initial_edge_array.copy()
    current_partition = initial_partition.copy()
    current_sorted_edges = get_sorted_edges(current_edge_array)

    best_edge_array = initial_edge_array.copy()
    best_partition = initial_partition.copy()
    best_bic = score_DAG(samples, initial_edge_array, initial_partition)
    best_iter = 0


    t = time.perf_counter()
    for i in range(MCMC_iterations):

        current_edge_array, current_partition, current_bic, current_sorted_edges = MCMC_iteration(current_edge_array, current_partition, samples, current_sorted_edges)

        if current_bic > best_bic:
            best_edge_array = current_edge_array.copy()
            best_partition = current_partition.copy()
            best_bic = current_bic
            best_iter = i


    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", best_bic)
    print("Found on iteration:", best_iter)
    print("SHD to real DAG was:", calc_SHD(best_edge_array, real_edge_array))
    print("The found DAG with correct coloring gives BIC:", score_DAG(samples, best_edge_array, real_partition))

    if start_with_GES_DAG:
        print("A random GES DAG with correct coloring gives BIC:", score_DAG(samples, initial_edge_array, real_partition))
        print("SHD to real DAG for it was:", calc_SHD(initial_edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax3)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(best_partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
