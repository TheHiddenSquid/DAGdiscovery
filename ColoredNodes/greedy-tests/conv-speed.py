import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import utils
from Greedyfuncs import CausalGreedySearch, Greedyiteration, get_sorted_edges, score_DAG


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    sparse = True
    sample_size = 1000
    

    # RUN MCM
    num_chains = 50
    max_num_nodes = 8  # 11 takes time
    

    min_time = []
    mean_time = []
    max_time = []


    for num_nodes in range(2, max_num_nodes):
        best_times = []
        print(num_nodes)
        for _ in range(num_chains):
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, no_colors, sparse)
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

            t = time.perf_counter()
            CausalGreedySearch(samples, num_waves=5)      

            best_times.append(time.perf_counter()-t)
        
        min_time.append(np.min(best_times))
        mean_time.append(np.mean(best_times))
        max_time.append(np.max(best_times))


    plt.semilogy(range(2,max_num_nodes), min_time, color="C0", linestyle="dashed", linewidth=0.5)
    plt.semilogy(range(2,max_num_nodes), mean_time, color="C0", linewidth=1.8)
    plt.semilogy(range(2,max_num_nodes), max_time, color="C0", linestyle="dashed", linewidth=0.5)
    plt.fill_between(range(2,max_num_nodes), min_time, mean_time, color="C0", alpha=.35)
    plt.fill_between(range(2,max_num_nodes), mean_time, max_time, color="C0", alpha=.35)



    plt.xlabel("nodes")
    plt.ylabel("time (s)")
    plt.title("Greedy search: Convergence speed")
    plt.show()

    
if __name__ == "__main__":
    main()
