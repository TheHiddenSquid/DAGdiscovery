import random
import sys
import time

import ges
import numpy as np
import pandas as pd

sys.path.append("../")
import MCMCfuncs
import utils


def main():
    random.seed(1)
    np.random.seed(1)

    num_colors = 2  # Change between 2 and int(p/2)

    num_tests = 25
    size_options = [4,6,8]
    edge_probabilities = [0.2,0.4,0.6,0.8]
    sample_options = [100, 500, 1000]

    t = time.perf_counter()

    df = pd.DataFrame(columns=["num_nodes", "num_samples", "edge_prob", "Algorithm", "SHD", "CHD"])
    for num_nodes in size_options: 
        print("num_nodes:", num_nodes)  
        for edge_prob in edge_probabilities:
            print("edge_prob:", edge_prob)  
            for i in range(num_tests):
                print(i)
                real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, int(num_nodes/2), edge_prob)
                real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)

                for num_samples in sample_options:
                    
                    # GES estimate of graph
                    samples = utils.generate_sample(num_samples, real_lambda_matrix, real_omega_matrix)

                    res = ges.fit_bic(data=samples)
                    GES_edge_array = res[0]
                    GES_SHD = utils.calc_SHD(real_edge_array, GES_edge_array)
                    df.loc[-1] = [num_nodes, num_samples, edge_prob, "GES", GES_SHD, None]
                    df.index = df.index + 1
                    df = df.sort_index()
                    
                    # MCMC estimate of graph
                    MCMC_edge_array, MCMC_partition, _ = MCMCfuncs.CausalMCMC(samples)
                    MCMC_SHD = utils.calc_SHD(real_edge_array, MCMC_edge_array)
                    MCMC_CHD = utils.calc_CHD(real_partition, MCMC_partition)
                    df.loc[-1] = [num_nodes, num_samples, edge_prob, "MCMC", MCMC_SHD, MCMC_CHD]
                    df.index = df.index + 1
                    df = df.sort_index()

    df.to_csv("outP/2colors.csv", index=False)
    print(df)
    print(time.perf_counter()-t)


  

if __name__ == "__main__":
    main()