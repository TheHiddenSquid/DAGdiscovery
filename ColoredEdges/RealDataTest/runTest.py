import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from Greedyfuncs import CausalGreedySearch
from MCMCfuncs import CausalMCMC


def main():
    # Load data "winequality-red.csv" or "sachs.data.txt"
    with open("sachs.data.txt") as f:
        data = []
        var_labels = f.readline().strip().split()   # or split(",")
        for line in f:
            data.append(line.strip().split())
    
    samples = np.zeros((len(data), len(var_labels)))
    for i, row in enumerate(data):
        samples[i,:] =  [eval(x) for x in row]
    

    random.seed(1)
    np.random.seed(1)

    # Create plots
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    print("GES-bic=", res[1])

    plt.axes(ax1)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES")
    

    A, PE, PN, bic = CausalGreedySearch(samples, num_waves=5)
    print("GCCES:", bic)

    plt.axes(ax2)
    G = nx.DiGraph(A)
    nx.draw_circular(G, node_color=utils.generate_node_color_map(PN), edge_color=utils.generate_edge_color_map(G, PE), with_labels=True)
    plt.title("Greedy")


    plt.show()


    
if __name__ == "__main__":
    main()
