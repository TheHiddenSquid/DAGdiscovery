import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
import random
from scipy import stats
import ges

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


def update_DAG(edge_array, partition):
    tmp_edge_array = edge_array.copy()
    tmp_partition = partition.copy()

    m = np.sum(tmp_edge_array)          # Number of current edges
    no_nodes = np.shape(edge_array)[0]  # Number of current noded
    no_colors = len(tmp_partition)      # Number of possible colors

    moves = [ "change_color", "add_edge", "remove_edge"]
    move = random.choice(moves)


    if move == "change_color":
        node = random.randrange(no_nodes)
        for i, part in enumerate(tmp_partition):
            if node in part:
                current_color = i
                break
        tmp_partition[current_color].remove(node)

        new_color = current_color
        while new_color == current_color:
            new_color = random.randrange(no_colors)
        tmp_partition[new_color].append(node)

        # Relative probability of jumping back
        q_quotient = 1


    if move == "add_edge":
        edges_giving_DAGs = get_addable_edges(tmp_edge_array)
        k_old = len(edges_giving_DAGs[0])  # Number of edges that can be added

        if k_old == 0:
            print("Could not add edge to graph")
            q_quotient = 1
        else:
            index = random.randrange(k_old)
            tmp_edge_array[edges_giving_DAGs[0][index], edges_giving_DAGs[1][index]] = 1

            # Relative probability of jumping back
            q_quotient = k_old / (m+1)
    

    if move == "remove_edge":
        if m == 0:
            print("Could not remove edge")
            q_quotient = 1
        else:
            edges = np.nonzero(tmp_edge_array)
            index = random.randrange(len(edges[0]))
            tmp_edge_array[edges[0][index], edges[1][index]] = 0

            # Relative probability of jumping back
            edges_giving_DAGs = get_addable_edges(tmp_edge_array)
            k_new = len(edges_giving_DAGs[0])

            q_quotient = m / k_new


    return tmp_edge_array, tmp_partition, q_quotient


def get_addable_edges(edge_array):
  
    tmp_edge_array = edge_array.copy()
    non_existing_edges = np.nonzero(tmp_edge_array == 0)

    edges_giving_DAGs = [[],[]]
    for i in range(len(non_existing_edges[0])):
        if tmp_edge_array[non_existing_edges[1][i], non_existing_edges[0][i]] == 1:
            continue
        tmp_edge_array[non_existing_edges[0][i], non_existing_edges[1][i]] = 1
        G = nx.DiGraph(tmp_edge_array)
        if nx.is_directed_acyclic_graph(G):
            edges_giving_DAGs[0].append(non_existing_edges[0][i])
            edges_giving_DAGs[1].append(non_existing_edges[1][i])
        tmp_edge_array[non_existing_edges[0][i], non_existing_edges[1][i]] = 0

    return edges_giving_DAGs

def update_addable_edges(edge_array, edges_giving_DAGs, last_action, last_edge):
    tmp_edge_array = edge_array.copy()

    if last_action == None:
         new_edges_giving_DAGs = edges_giving_DAGs.copy()


    if last_action == "change_color":
         new_edges_giving_DAGs = edges_giving_DAGs.copy()


    if last_action == "add_edge":
        new_edges_giving_DAGs = [[],[]]

        for i in range(len(edges_giving_DAGs[0])):
            start_node = edges_giving_DAGs[0][i]
            end_node = edges_giving_DAGs[1][i]
            if start_node == last_edge[0] or start_node == last_edge[1] or end_node == last_edge[0] or end_node == last_edge[1]:
                tmp_edge_array[start_node, end_node] = 1
                G = nx.DiGraph(tmp_edge_array)
                if nx.is_directed_acyclic_graph(G):
                    new_edges_giving_DAGs[0].append(start_node)
                    new_edges_giving_DAGs[1].append(end_node)
                tmp_edge_array[start_node, end_node] = 0
            else:
                new_edges_giving_DAGs[0].append(start_node)
                new_edges_giving_DAGs[1].append(end_node)

    if last_action == "remove_edge":
        new_edges_giving_DAGs = edges_giving_DAGs.copy()
        new_edges_giving_DAGs[0].append(last_edge[0])
        new_edges_giving_DAGs[1].append(last_edge[1])

        for i in range(np.shape(edge_array)[0]):
            if tmp_edge_array[i, last_edge[0]] == 1:
                continue

            tmp_edge_array[i, last_edge[0]] = 1
            G = nx.DiGraph(tmp_edge_array)
            if nx.is_directed_acyclic_graph(G):
                new_edges_giving_DAGs[0].append(i)
                new_edges_giving_DAGs[1].append(last_edge[0])
            tmp_edge_array[start_node, end_node] = 0


    return new_edges_giving_DAGs


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
    random.seed(100)
    no_nodes = 10
    no_colors = 4
    edge_probability = 0.3
    sample_size = 1000
    start_with_GES_DAG = True

    real_partition, real_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

    global samples
    samples = generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # Create plots
    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(4, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax11)
    G = nx.DiGraph(real_lambda_matrix)
    nx.draw_circular(G, node_color=generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")



     # MCMC setup
    if start_with_GES_DAG:
        # GES estimate of graph
        res = ges.fit_bic(data=samples)
        GES_edge_array = res[0]
        initial_edge_array = GES_edge_array.copy()

       
        # Take an initial DAG from the given GES CPDAG
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


        # Initial coloring guess
        initial_partition = generate_partition(no_nodes, no_nodes)

    else:
        # Start with random DAG
        initial_partition, initial_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_lambda_matrix != 0, dtype="int")
    


    # More MCMC setup
    global current_edge_array
    global current_partition

    current_edge_array = initial_edge_array.copy()
    current_partition = initial_partition.copy()

    plt.axes(ax12)
    G = nx.DiGraph(current_edge_array)
    nx.draw_circular(G, node_color=generate_color_map(current_partition), with_labels=True)
    plt.title("MCMC")


    # Setop for BIC plots
    global bics
    global cumsum
    start_bic = score_DAG(samples, current_edge_array, current_partition)
    bics = [start_bic]
    cumsum = [start_bic]

    global SHDs
    SHDs = [calc_SHD(current_edge_array, real_edge_array)]

    global CHDs
    CHDs = [calc_partition_distance(current_partition, real_partition)]



    def update(frame):
        # Update graph visuals
        plt.axes(ax12)
        ax12.clear()
        global current_edge_array
        global current_partition
        global samples

        old_edge_array = current_edge_array.copy()
        old_partition = current_partition.copy()

        new_edge_array, new_partition, q_quotient = update_DAG(old_edge_array, old_partition)

        new_bic = score_DAG(samples, new_edge_array, new_partition)
        old_bic = score_DAG(samples, old_edge_array, old_partition)

  
        if random.random() <= (new_bic / old_bic) * q_quotient: 
            current_edge_array = new_edge_array
            current_partition = new_partition
        else:
            current_edge_array = old_edge_array
            current_partition = old_partition

        #print(len(get_addable_edges(anim_edges)[0]))
        
        color_map = generate_color_map(current_partition)
        G = nx.DiGraph(current_edge_array)
        nx.draw_circular(G, node_color=color_map, with_labels=True)
        plt.title("MCMC")

        # Update bic graphics
        global bics
        global cumsum
        global SHDs
        global CHDs

        bics.append(new_bic)
        plt.axes(ax21)
        ax21.clear()
        plt.plot(range(len(bics)),bics)
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("BIC")

        plt.axes(ax22)
        ax22.clear()
        cumsum.append(cumsum[-1]+new_bic)

        plt.plot(range(len(bics)),[cumsum[i]/(i+1) for i in range(len(bics))])
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("Rolling Average")


        # SHD for each iteration
        plt.axes(ax31)
        ax31.clear()
        SHDs.append(calc_SHD(real_edge_array, current_edge_array))

        plt.plot(range(len(SHDs)), SHDs)
        plt.xlabel("iterations")
        plt.ylabel("SHD")
        plt.title("SHD")

        # Scatter for intuition
        plt.axes(ax32)
        ax32.clear()

        plt.scatter(bics, SHDs)
        plt.xlabel("BIC")
        plt.ylabel("SHD")
        plt.title("BIC and SHD relation")


        # CHD for each iteration
        plt.axes(ax41)
        ax41.clear()
        CHDs.append(calc_partition_distance(real_partition, current_partition))

        plt.plot(range(len(CHDs)), CHDs)
        plt.xlabel("iterations")
        plt.ylabel("CHD")
        plt.title("CHD")

        # Scatter for intuition
        plt.axes(ax42)
        ax42.clear()

        plt.scatter(bics, CHDs)
        plt.xlabel("BIC")
        plt.ylabel("CHD")
        plt.title("BIC and CHD relation")


    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=10)
    plt.show()



if __name__ == "__main__":
    main()
