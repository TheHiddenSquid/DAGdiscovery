import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from MCMCfuncs import CausalMCMC
from Tabufuncs import CausalTabuSearch


def main():
    with open("student-mat.csv") as f:
        data = []
        var_labels = f.readline().strip().split(",")
        for line in f:
            data.append(line.strip().split(","))
    
    used_varables = ["sex", "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "health", "absences", "G1", "G2", "G3"]

    samples = np.zeros((len(data), len(used_varables)))

    for i, var in enumerate(used_varables):
        index = var_labels.index(var)
        picked_data = [x[index] for x in data]
        samples[:,i] = picked_data
    

    random.seed(1)
    np.random.seed(1)
    MCMC_iterations = 10_000
    disp_labels = {i:used_varables[i] for i in range(len(used_varables))}
    print(disp_labels)


    # Create plots
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    print("GES-bic=", res[1])

    plt.axes(ax1)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, labels=disp_labels, with_labels=True)
    plt.title("GES")
    

    t = time.perf_counter()
    #edge_array, partition, bic, found_iter, _ = CausalMCMC(samples, MCMC_iterations, start_from_GES = False, debug=True)
    edge_array, partition, bic, found_iter, _ = CausalTabuSearch(samples, MCMC_iterations)


    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", found_iter)



    plt.axes(ax2)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), labels=disp_labels, with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
