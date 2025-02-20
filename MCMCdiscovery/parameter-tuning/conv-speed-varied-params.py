import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import copy
sys.path.append("../")
from MCMCfuncs import MCMC_iteration
from MCMCfuncs import score_DAG
from MCMCfuncs import get_sorted_edges
import utils



def main():
    random.seed(1)
    np.random.seed(1)
    no_nodes = 6
    no_colors = 3
    edge_probability = 0.3
    sample_size = 1000


    # RUN MCM
    MCMC_iterations = 10_000
    num_chains = 5
    param_list = [0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.499]
    num_params = len(param_list)
    
    chain_rolling_best_bic = [[] for _ in range(num_params)]

    for i in range(num_chains):
        print(i)
        real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_probability)
        real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
        real_bic = score_DAG(samples, real_edge_array, real_partition)[0]

        start_partition, start_edge_array, _ = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
        start_edge_array = np.array(start_edge_array != 0, dtype="int")

        for j in range(num_params):
            print(j)
            
            params = [param_list[j]]*2

            current_partition = copy.deepcopy(start_partition)
            current_edge_array = start_edge_array.copy()

            current_bic = score_DAG(samples, current_edge_array, current_partition)
            current_sorted_edges = get_sorted_edges(current_edge_array)

            best_bic = current_bic[0]
            rolling_best_bic = [best_bic,best_bic]
            

            for k in range(MCMC_iterations):
                current_edge_array, current_partition, current_bic, current_sorted_edges, _ = MCMC_iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges, move_weights=params)

                best_bic = max(best_bic, current_bic[0])
                rolling_best_bic.append(real_bic-best_bic)

            chain_rolling_best_bic[j].append(rolling_best_bic)


 
    param_chains = []
    for group in chain_rolling_best_bic:
        arrays = [np.array(x) for x in group]
        param_chains.append([np.mean(k) for k in zip(*arrays)])


    for k in range(num_params):
        plt.semilogx(range(MCMC_iterations+2), param_chains[k], label = f"P = {param_list[k]}")

    plt.xlabel("iterations")
    plt.ylabel("true bic - best bic")
    plt.ylim((0,20))
    plt.legend()
    plt.title("Convergence speed (6 nodes)")
    plt.show()

    
if __name__ == "__main__":
    main()
