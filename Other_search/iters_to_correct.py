import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import utils
from Tabufuncs import CausalTabuSearch, get_sorted_edges, iteration, score_DAG


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    sparse = True
    sample_size = 1000
    

    # RUN MCM
    num_chains = 100
    max_num_nodes = 6  # 11 takes time
    

    required_iters = []


    for num_nodes in range(2, max_num_nodes):
        best_iters = []
        print(num_nodes)
        for _ in range(num_chains):
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, no_colors, sparse)
            real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

            real_bic = utils.score_DAG(samples, real_edge_array, real_partition)

            current_edge_array = np.zeros((num_nodes, num_nodes), dtype="int")
            current_partition = [{i} for i in range(num_nodes)]
            current_bic = score_DAG(samples, current_edge_array, current_partition)
            current_sorted_edges = get_sorted_edges(current_edge_array)
            best_bic = current_bic[0]       

            CausalTabuSearch(samples, 0)

            iters = 1
            while  best_bic < real_bic and not np.isclose(best_bic, real_bic):
                iters += 1
                current_edge_array, current_partition, current_bic, current_sorted_edges, _ = iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges)

                if current_bic[0] > best_bic:
                    best_bic = current_bic[0]
                
                if iters % 100_000 == 0:
                    print("yo")
                    print(real_bic - best_bic)
                
            best_iters.append(iters)
        
        required_iters.append(np.mean(best_iters))



    plt.semilogy(range(2,max_num_nodes), required_iters)

    plt.xlabel("nodes")
    plt.ylabel("iterations")
    plt.title("Convergence speed")
    plt.show()

    
if __name__ == "__main__":
    main()
