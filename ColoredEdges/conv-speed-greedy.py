import random
import time

import matplotlib.pyplot as plt
import numpy as np
import utils
from Greedyfuncs import CausalGreedySearch


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    edge_prob = 0.8
    sample_size = 1000
    

    # RUN MCM
    num_chains = 50
    max_num_nodes = 21
    
    required_iters_min = []
    required_iters_mean = []
    required_iters_max = []

    for num_nodes in range(2, max_num_nodes):
        best_times = []
        print(num_nodes)
        for _ in range(num_chains):
            real_EP, real_NP, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, no_colors, no_colors, edge_prob)
            real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

            t = time.perf_counter()
            CausalGreedySearch(samples)
            best_times.append(time.perf_counter()-t)
        
        required_iters_min.append(np.min(best_times))
        required_iters_mean.append(np.mean(best_times))
        required_iters_max.append(np.max(best_times))


    plt.semilogy(range(2,max_num_nodes), required_iters_min, color="C0", linewidth=0.5)
    plt.fill_between(range(2,max_num_nodes), required_iters_max, required_iters_min, color="C0", alpha=.2)
    plt.semilogy(range(2,max_num_nodes), required_iters_mean, color="C0", linestyle="-", marker="s", linewidth=1.5)
    plt.semilogy(range(2,max_num_nodes), required_iters_max, color="C0", linewidth=0.5)

    plt.xlabel("Nodes")
    plt.ylabel("Time (s)")
    plt.title("GCCES: Convergence Speed")
    plt.show()

    
if __name__ == "__main__":
    main()
