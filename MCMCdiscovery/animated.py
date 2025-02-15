import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import ges
from MCMCfuncs import MCMC_iteration
from MCMCfuncs import score_DAG
from MCMCfuncs import get_sorted_edges
import utils


def main():
    no_nodes = 6
    no_colors = 4
    edge_probability = 0.3
    sample_size = 1000
    start_with_GES_DAG = True

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_probability)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

    global samples
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # Create plots
    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(4, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax11)
    G = nx.DiGraph(real_lambda_matrix)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
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
        initial_partition = utils.generate_random_partition(no_nodes, no_nodes)

    else:
        # Start with random DAG
        initial_partition, initial_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_lambda_matrix != 0, dtype="int")
    


    # More MCMC setup
    global current_edge_array
    global current_partition
    global current_sorted_edges
    global current_bic

    current_edge_array = initial_edge_array.copy()
    current_partition = initial_partition.copy()
    current_sorted_edges = get_sorted_edges(current_edge_array)
    current_bic = score_DAG(samples, current_edge_array, current_partition)

    plt.axes(ax12)
    G = nx.DiGraph(current_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(current_partition), with_labels=True)
    plt.title("MCMC")


    # Setop for BIC plots
    global bics
    global cumsum
    start_bic = score_DAG(samples, current_edge_array, current_partition)[0]
    bics = [start_bic]
    cumsum = [start_bic]

    global SHDs
    SHDs = [utils.calc_SHD(current_edge_array, real_edge_array)]

    global CHDs
    CHDs = [utils.calc_CHD(current_partition, real_partition)]



    def update(frame):
        # Update graph visuals
        plt.axes(ax12)
        ax12.clear()
        global current_edge_array
        global current_partition
        global current_sorted_edges
        global current_bic
        global samples

        current_edge_array, current_partition, current_bic, current_sorted_edges, _ = MCMC_iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges)
        
        G = nx.DiGraph(current_edge_array)
        nx.draw_circular(G, node_color=utils.generate_color_map(current_partition), with_labels=True)
        plt.title("MCMC")

        # Update bic graphics
        global bics
        global cumsum
        global SHDs
        global CHDs

        bics.append(current_bic[0])
        plt.axes(ax21)
        ax21.clear()
        plt.plot(range(len(bics)),bics)
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("BIC")

        plt.axes(ax22)
        ax22.clear()
        cumsum.append(cumsum[-1]+current_bic[0])

        plt.plot(range(len(bics)),[cumsum[i]/(i+1) for i in range(len(bics))])
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("Rolling Average")


        # SHD for each iteration
        plt.axes(ax31)
        ax31.clear()
        SHDs.append(utils.calc_SHD(real_edge_array, current_edge_array))

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
        CHDs.append(utils.calc_CHD(real_partition, current_partition))

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
