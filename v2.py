import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
from scipy import stats
import ges

def generate_colored_DAG(no_nodes, no_colors, edge_probability):
    # Add edges and make sure it is a DAG
    fail = True
    while fail:
        fail = False
        G = nx.DiGraph()

        for node in range(no_nodes):
            G.add_node(node)

        for i in range(no_nodes):
            for j in range(i+1, no_nodes):
                if random.random() < edge_probability:
                    if random.random() < 0.5:
                        G.add_edge(i,j)
                    else:
                        G.add_edge(i,j)       
                    if not nx.is_directed_acyclic_graph(G):
                        fail = True
                if fail:
                    break
            if fail:
                break

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
    if len(partition) > 6:
        raise ValueError("Too many colors needed for color map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown"]
    length = sum([len(x) for x in partition])
    color_map = [None] * length

    for i, part in enumerate(partition):
        for node in part:
            color_map[node] = colors[i]

    return color_map


def update_DAG(edges, partition):
    moves = ["add", "remove", "change_color"]
    move = random.choice(moves)

    no_nodes = sum(len(x) for x in partition)
    no_colors = len(partition)

    if move == "add":
        start_options = [*range(no_nodes)]
        random.shuffle(start_options)
        done = False

        for start in start_options:
            neighbors = edges[start]
            if sum(neighbors) == no_nodes-1:
                continue
            options = np.nonzero(neighbors == 0)[0]
            options = list(options)
            options.remove(start)
            random.shuffle(options)
            
            for option in options:
                edges[start, option] = 1
                G = nx.DiGraph(edges)
                if nx.is_directed_acyclic_graph(G):
                    done = True
                    break
                edges[start, option] = 0
            if done:
                break
        else:
            print("failed to add edge")

        new_edges = edges
        new_partition = partition


    elif move == "remove":
        start_options = [*range(no_nodes)]
        random.shuffle(start_options)

        for node in start_options:
            neighbors = edges[node]
            if sum(neighbors) == 0:
                continue
            options = neighbors.nonzero()[0]
            to_remove = random.choice(options)
            edges[node, to_remove] = 0
            break
        else:
            print("failed to remove edge")

        new_edges = edges
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

        new_edges = edges
        new_partition = partition

    return new_edges.copy(), new_partition.copy()

def main():
    no_nodes = 10
    no_colors = 3
    edge_probability = 0.2

    partition, lambda_matrix, omega_matrix = generate_colored_DAG(no_nodes, no_colors, edge_probability)
    samples = generate_sample(100000, lambda_matrix, omega_matrix)


    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2)


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(lambda_matrix)
    color_map = generate_color_map(partition)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    plt.title("Data generating DAG")




    # GES estimate of graph
    res = ges.fit_bic(data=samples)

    # Take an initial DAG from the given CPDAG
    edges = res[0]
    double = []
    for i in range(no_nodes):
        for j in range(i+1, no_nodes):
            if edges[i,j] == 1 and edges[j,i] == 1:
                double.append((i,j))
                edges[i,j] = 0
                edges[j,i] = 0

    for edge in double:
        new_edges = edges.copy()
        new_edges[edge[0], edge[1]] = 1
        G = nx.DiGraph(new_edges)
        if nx.is_directed_acyclic_graph(G):
            edges = new_edges
            continue

        new_edges = edges.copy()
        new_edges[edge[1], edge[0]] = 1
        G = nx.DiGraph(new_edges)
        if nx.is_directed_acyclic_graph(G):
            edges = new_edges
            continue

        raise ValueError("Could not create graph")


    # Initial coloring guess
    partition = generate_partition(no_nodes, no_colors)
    

    # Make random moves
    global anim_edges
    global anim_partition
    anim_edges = edges.copy()
    anim_partition = partition.copy()

    plt.axes(ax2)
    G = nx.DiGraph(anim_edges)
    nx.draw_circular(G, node_color=color_map, with_labels=True)
    plt.title("MCMC")


    def update(frame):
        ax2.clear()
        global anim_edges
        global anim_partition
        anim_edges, anim_partition = update_DAG(anim_edges, anim_partition)

        color_map = generate_color_map(anim_partition)
        G = nx.DiGraph(anim_edges)
        nx.draw_circular(G, node_color=color_map, with_labels=True)
        plt.title("MCMC")



    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=100)
    plt.show()


    #print("score:", res[1])
    plt.show()

    
    
if __name__ == "__main__":
    main()
