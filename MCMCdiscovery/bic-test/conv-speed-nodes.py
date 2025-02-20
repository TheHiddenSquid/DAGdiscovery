import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append("../")
from MCMCfuncs import MCMC_iteration
from MCMCfuncs import score_DAG
from MCMCfuncs import get_sorted_edges
import utils



def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    edge_probability = 0.3
    sample_size = 1000
    

    # RUN MCM
    num_chains = 5
    max_num_nodes = 10  # 11 takse time
    

    required_iters = []


    for i in range(3, max_num_nodes):
        best_iters = []
        print(i)
        for _ in range(num_chains):
            no_nodes = i
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_probability)
            real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

            real_bic = score_DAG(samples, real_edge_array, real_partition)[0]


            current_partition, current_edge_array, _ = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
            current_edge_array = np.array(current_edge_array != 0, dtype="int")
            current_bic = score_DAG(samples, current_edge_array, current_partition)
            current_sorted_edges = get_sorted_edges(current_edge_array)

            best_bic = current_bic[0]            

            iters = 1
            while real_bic - best_bic > 0.1:
                iters += 1
                current_edge_array, current_partition, current_bic, current_sorted_edges, _ = MCMC_iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges)

                if current_bic[0] > best_bic:
                    best_bic = current_bic[0]
                
                if iters % 100_000 == 0:
                    print(real_bic - best_bic)
                
            best_iters.append(iters)
        
        required_iters.append(np.mean(best_iters))



    plt.semilogy(range(3,max_num_nodes), required_iters)

    plt.xlabel("nodes")
    plt.ylabel("iterations")
    plt.title("Convergence speed")
    plt.show()

    
if __name__ == "__main__":
    main()
