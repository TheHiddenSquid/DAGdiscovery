import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import MCMCfuncs
import utils


def test2d(resolution, sample_size, MCMCiterations, savefile=None, loadfile=None):
    if loadfile is not None:
        with open(loadfile, "rb") as f:
            y = pickle.load(f)    
    else:
        random.seed(2)
        np.random.seed(2)
        no_nodes = 20
        no_colors = 3
        sparse = True

        _, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, sparse)
        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

        
        # RUN MCMC
        # Fully random colored DAG
        y = np.zeros((resolution+1))

        for i in range(1,resolution):
            random.seed(2)
            np.random.seed(2)
            s = i/resolution
            print(s)
            _, _, _, _, fails = MCMCfuncs.CausalMCMC(samples, MCMCiterations, move_weights=[1-s,s], debug=True)
            y[i] = fails/MCMCiterations
            
        if savefile is not None:
            with open(savefile, "wb") as f:
                pickle.dump(y, f)

    # Plot data
    end = y.shape[0]

    x = np.linspace(0,1,end)

    plt.plot(x[1:end-1],y[1:end-1])
    plt.xlabel("P(change_color)")
    plt.ylabel("fails")
    plt.title("20 nodes")
    plt.show()


def main():

    save = "lastsave.pkl"
    #save = "2d6nodes.pkl"


    load1 = "2d6nodes.pkl"
    load2 = "2d20nodes.pkl"
    test2d(resolution = 50, sample_size = 1000, MCMCiterations = 10_000, savefile=save, loadfile=None)


    
if __name__ == "__main__":
    main()
    