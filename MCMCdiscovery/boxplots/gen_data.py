import random
import sys
import time
from multiprocessing import Pool

import ges
import numpy as np
import pandas as pd

sys.path.append("../")
import MCMCfuncs
import utils


def get_data_df(num):
    size_options = [4,6,8]
    edge_probabilities = [0.2,0.4,0.6,0.8]
    sample_options = [100, 500, 1000]
    color_options = ["2", "num_nodes/2"]

    t_start = time.perf_counter()

    df = pd.DataFrame(columns=["num_nodes", "num_colors", "edge_prob", "num_samples", "Algorithm", "SHD", "CHD"])
    for num_nodes in size_options: 
        for edge_prob in edge_probabilities:
            for nc in color_options:
                nc_used = int(eval(nc))
                real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, nc_used, edge_prob)
                real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)
                for num_samples in sample_options:
                    
                    # GES estimate of graph
                    samples = utils.generate_sample(num_samples, real_lambda_matrix, real_omega_matrix)

                    res = ges.fit_bic(data=samples)
                    GES_edge_array = res[0]
                    GES_SHD = utils.calc_SHD(real_edge_array, GES_edge_array)
                    df.loc[-1] = [num_nodes, nc_used, edge_prob, num_samples, "GES", GES_SHD, None]
                    df.index = df.index + 1
                    df = df.sort_index()
                    
                    # MCMC estimate of graph
                    MCMC_edge_array, MCMC_partition, _ = MCMCfuncs.CausalMCMC(samples)
                    MCMC_SHD = utils.calc_SHD(real_edge_array, MCMC_edge_array)
                    MCMC_CHD = utils.calc_CHD(real_partition, MCMC_partition)
                    df.loc[-1] = [num_nodes, nc_used, edge_prob, num_samples, "MCMC", MCMC_SHD, MCMC_CHD]
                    df.index = df.index + 1
                    df = df.sort_index()
    
    t_end = time.perf_counter()

    return num, df, t_end-t_start





def main():
    random.seed(1)
    np.random.seed(1)

    num_tests = 40
  
    dfs = []
    print("Start")
    t_start = time.perf_counter()
    with Pool() as pool:
        result = pool.imap(get_data_df, [x for x in range(num_tests)])
        for num, df, duration in result:
            print(f"{num}, took, {duration}, s")
            dfs.append(df)

    final_df = pd.concat(dfs)
    t_end = time.perf_counter()
    print(f"All done in {t_end-t_start} s")
    final_df.to_csv("out_MCMC.csv", index=False)




if __name__ == "__main__":
    main()
