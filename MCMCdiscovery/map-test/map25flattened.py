import random
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
from collections import defaultdict

import utils
from MCMCfuncs import MCMC_iteration, score_DAG
from utils import get_sorted_edges


def get_all_3node_DAGs(color = False):
    AllDAGs = defaultdict(lambda: 0)
    i = 0
    if color == True:
        cap = 125
    else:
        cap = 25

    while len(AllDAGs) < cap:
        partition, A, _ = utils.generate_colored_DAG(3,3, random.random())
        A = np.array(A != 0, dtype="int")
        h1 = A.tobytes()
        if color == True:
            h2 = tuple(tuple(x) for x in utils.sorted_partition(partition))
            AllDAGs[(h1,h2)] += 1
        else:
            AllDAGs[h1] += 1

    dags = list(AllDAGs.keys())
    dags.sort()
    return dags


def main():
    random.seed(8)
    np.random.seed(8)
    # General setup
    num_nodes = 3
    num_colors = 3
    sparse = True
    sample_size = 1000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, sparse)
    real_A = np.array(real_lambda_matrix != 0, dtype="int")
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # setup for hypergraph
    global dags
    dags = get_all_3node_DAGs(color = False)


    edge_array = np.zeros((25,25), dtype="int")
    for i, DAG in enumerate(dags):
        A = np.reshape(np.frombuffer(DAG, dtype="int"), (3,3))
        
        edges_in, edges_giving, _ = get_sorted_edges(A)

        for edge in edges_in:
            B = A.copy()
            B[edge] = 0
            B = B.tobytes()
            edge_array[i,dags.index(B)] = 1

        for edge in edges_giving:
            B = A.copy()
            B[edge] = 1
            B = B.tobytes()
            edge_array[i,dags.index(B)] = 1


    
    
    allbics = [score_DAG(samples, np.reshape(np.frombuffer(x, dtype="int"), (3,3)), real_partition)[0] for x in dags]

    print(sorted(allbics))
    print("True was", sorted(allbics).index(score_DAG(samples, real_A, real_partition)[0]), "best of 25")


    # setup for MCMC
  
    fig, ax = plt.subplots()

    def init():
        global current_edge_array
        global current_partition
        global current_bic
        global labels
        current_edge_array = np.zeros((3,3))
        current_partition = real_partition.copy()
        current_bic = score_DAG(samples, current_edge_array, current_partition)

        labels = {i:0 for i in range(25)}
        labels[0] = 1
        
        plt.bar(labels.keys(), [x/sum(labels.values()) for x in labels.values()], label="MCMC")
        plt.plot(labels.keys(), [x/sum(allbics) for x in allbics], color = "C1", label="True value")
        plt.xlabel("DAG ID")
        plt.ylabel("Probabiliry")
        plt.legend(loc = "upper left")

    def update(frame):
        global labels
        global current_edge_array
        global current_partition
        global current_bic
        global current_node

        ax.clear()
        current_edge_array, current_partition, current_bic, _ = MCMC_iteration(samples, current_edge_array, current_partition, current_bic, [0.4,0.6])

        A = current_edge_array.astype("int")
        labels[dags.index(A.tobytes())] += 1
        bar_color = ["C0"]*25
        bar_color[dags.index(A.tobytes())] = "lightpink"
        bar_color[dags.index(real_A.tobytes())] = "gold"
        plt.bar(labels.keys(), [x/sum(labels.values()) for x in labels.values()], label="MCMC", color = bar_color)
        plt.plot(labels.keys(), [x/sum(allbics) for x in allbics], color = "C1", label="True value")
        plt.xlabel("DAG ID")
        plt.ylabel("Probabiliry")
        plt.legend(loc = "upper left")


    ani = animation.FuncAnimation(fig=fig, func=update, frames=10_000, interval=10, init_func=init)
    plt.show()

    
    
if __name__ == "__main__":
    main()
