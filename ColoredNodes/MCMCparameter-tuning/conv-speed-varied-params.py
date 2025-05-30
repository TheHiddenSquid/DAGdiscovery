import copy
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import utils
from MCMCfuncs import CausalMCMC, MCMC_iteration, score_DAG_full


def main():
    random.seed(1)
    np.random.seed(1)
    num_nodes = 6
    num_colors = 3
    sparse = True
    sample_size = 1000


    # RUN MCM
    MCMC_iterations = 100_000
    num_chains = 50
    param_list = [0.001, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999]
    num_params = len(param_list)
    
    chain_rolling_best_bic = [[] for _ in range(num_params)]

    for i in range(num_chains):
        print(i)
        real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, sparse)
        real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
        real_bic = utils.score_DAG(samples, real_edge_array, real_partition)

        start_partition, start_edge_array, _ = utils.generate_colored_DAG(num_nodes, num_nodes, 0.5)
        start_edge_array = np.array(start_edge_array != 0, dtype="int")

        for j in range(num_params):
            print(j)
            
            params = [1-param_list[j], param_list[j]]

            current_partition = copy.deepcopy(start_partition)
            current_edge_array = start_edge_array.copy()

            CausalMCMC(samples, 0)
            current_bic, current_ML_data = score_DAG_full(current_edge_array, current_partition)

            best_bic = current_bic
            rolling_best_bic = [real_bic-best_bic, real_bic-best_bic]
            

            for k in range(MCMC_iterations):
                move = random.choices([0, 1], k=1, weights=params)[0]
                current_edge_array, current_partition, current_bic, current_ML_data, _ = MCMC_iteration(move, current_edge_array, current_partition, current_bic, current_ML_data)

                best_bic = max(best_bic, current_bic)
                rolling_best_bic.append(real_bic-best_bic)

            chain_rolling_best_bic[j].append(rolling_best_bic)


 
    param_chains = []
    for group in chain_rolling_best_bic:
        arrays = [np.array(x) for x in group]
        param_chains.append([np.mean(k) for k in zip(*arrays)])


    for k in range(num_params):
        plt.semilogx(range(MCMC_iterations+2), param_chains[k], label = rf"$\pi$ = {param_list[k]}")

    plt.xlabel("iterations")
    plt.ylabel("true bic - best bic")
    plt.legend()
    plt.title(f"Convergence speed ({num_nodes} nodes)")
    plt.show()

    
if __name__ == "__main__":
    main()
