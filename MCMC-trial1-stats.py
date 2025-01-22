import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    moves = ["add", "remove", "change_color"]
    move = random.choice(moves)

    no_nodes = sum(len(x) for x in partition)
    no_colors = len(partition)

    if move == "add":
        start_options = [*range(no_nodes)]
        random.shuffle(start_options)
        done = False

        for start in start_options:
            neighbors = edge_array[start]
            if sum(neighbors) == no_nodes-1:
                continue
            options = np.nonzero(neighbors == 0)[0]
            options = list(options)
            options.remove(start)
            random.shuffle(options)
            
            for option in options:
                edge_array[start, option] = 1
                G = nx.DiGraph(edge_array)
                if nx.is_directed_acyclic_graph(G):
                    done = True
                    break
                edge_array[start, option] = 0
            if done:
                break
        else:
            print("failed to add edge")

        new_edge_array = edge_array
        new_partition = partition


    elif move == "remove":
        start_options = [*range(no_nodes)]
        random.shuffle(start_options)

        for node in start_options:
            neighbors = edge_array[node]
            if sum(neighbors) == 0:
                continue
            options = neighbors.nonzero()[0]
            to_remove = random.choice(options)
            edge_array[node, to_remove] = 0
            break
        else:
            print("failed to remove edge")

        new_edge_array = edge_array
        new_partition = partition


    elif move == "change_color":
        node = random.randrange(no_nodes)
        for i, part in enumerate(partition):
            if node in part:
                current_color = i
        partition[current_color].remove(node)

        new_color = current_color
        while new_color == current_color:
            new_color = random.randrange(no_colors)
        partition[new_color].append(node)

        new_edge_array = edge_array
        new_partition = partition

    return new_edge_array.copy(), new_partition.copy()

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


def main():
    no_nodes = 10
    no_colors = 4
    edge_probability = 0.2
    sample_size = 1000

    real_partition, real_lambda_matrix, real_omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax11)
    G = nx.DiGraph(real_lambda_matrix)
    color_map = generate_color_map(real_partition)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    

    # MCMC setup
    # Take an initial DAG from the given GES CPDAG
    
    double = []
    for i in range(no_nodes):
        for j in range(i+1, no_nodes):
            if GES_edge_array[i,j] == 1 and GES_edge_array[j,i] == 1:
                double.append((i,j))
                GES_edge_array[i,j] = 0
                GES_edge_array[j,i] = 0

    for edge in double:
        new_edges = GES_edge_array.copy()
        new_edges[edge[0], edge[1]] = 1
        G = nx.DiGraph(new_edges)
        if nx.is_directed_acyclic_graph(G):
            GES_edge_array = new_edges
            continue

        new_edges = GES_edge_array.copy()
        new_edges[edge[1], edge[0]] = 1
        G = nx.DiGraph(new_edges)
        if nx.is_directed_acyclic_graph(G):
            GES_edge_array = new_edges
            continue

        raise ValueError("Could not create DAG")


    # Initial coloring guess
    real_partition = generate_partition(no_nodes, no_nodes)
    

    # Make random moves
    global anim_samples
    global anim_edges
    global anim_partition
    anim_samples = samples.copy()
    anim_edges = GES_edge_array.copy()
    anim_partition = real_partition.copy()

    plt.axes(ax12)
    G = nx.DiGraph(anim_edges)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    plt.title("MCMC")


    # Setop for BIC plots
    global bics
    global cumsum
    start_bic = score_DAG(anim_samples, anim_edges, anim_partition)
    bics = [start_bic]
    cumsum = [start_bic]


    global SHDs
    SHDs = [calc_SHD(GES_edge_array, real_edge_array)]



    def update(frame):
        # Update graph visuals
        plt.axes(ax12)
        ax12.clear()
        global anim_edges
        global anim_partition
        global anim_samples

        old_anim_edges = anim_edges.copy()
        old_anim_partition = anim_partition.copy()

        new_anim_edges, new_anim_partition = update_DAG(old_anim_edges, old_anim_partition)

        new_bic = score_DAG(anim_samples, new_anim_edges, new_anim_partition)
        old_bic = score_DAG(anim_samples, old_anim_edges, old_anim_partition)

        did_jump = False
        if new_bic > old_bic:
            did_jump = True
            anim_edges = new_anim_edges
            anim_partition = new_anim_partition
        else:
            if random.random() <= new_bic / old_bic: 
                anim_edges = new_anim_edges
                anim_partition = new_anim_partition
                did_jump = True
            else:
                anim_edges = old_anim_edges
                anim_partition = old_anim_partition


        color_map = generate_color_map(anim_partition)
        G = nx.DiGraph(anim_edges)
        nx.draw_circular(G, node_color=color_map, with_labels=True)
        plt.title("MCMC")

        # Update bic graphics
        global bics
        global cumsum
        global SHDs

        if did_jump:

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
            SHDs.append(calc_SHD(real_edge_array, anim_edges))

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
            plt.title("BIC and SHD realtion")





    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=1)
    plt.show()



    
if __name__ == "__main__":
    main()
