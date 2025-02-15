import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append("../")
import MCMCfuncs
import utils
from matplotlib import cm
import matplotlib.colors as colors
import pickle


def test3d(resolution, sample_size, MCMCiterations, savefile=None, loadfile=None):
    if loadfile is not None:
        with open(loadfile, "rb") as f:
            z = pickle.load(f)    
    else:


        random.seed(2)
        np.random.seed(2)
        num_nodes = 6
        num_colors = 3
        edge_probability = 0.3

        _, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, edge_probability)
        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

        
        # RUN MCMC
        # Fully random colored DAG
        z = np.zeros((resolution+1,resolution+1))
    
        for i in range(1,resolution):
            s = i/resolution
            for j in range(1,resolution-i):
                random.seed(2)
                np.random.seed(2)
                t = j/resolution
                print(s,t)
                
                _, _, _, _, fails = MCMCfuncs.CausalMCMC(samples, MCMCiterations, move_weights=[s,t], debug=True)
                z[i,j] = fails/MCMCiterations

        if savefile is not None:
            with open(savefile, "wb") as f:
                pickle.dump(z, f)

    # Plot data
    nrows = z.shape[0]
    x = np.linspace(0, nrows-1, nrows)
    y = np.linspace(0, nrows-1, nrows)
    x, y = np.meshgrid(x, y)

    region = np.s_[0:nrows-1, 0:nrows-1]
    x, y, z = x[region].ravel(), y[region].ravel(), z[region].ravel()

    # parameters for bar3d(...) func
    bottom = np.full_like(z, np.min(z))
    width = (np.max(x)-np.min(x))/np.sqrt(np.shape(x)[0])
    depth = (np.max(y)-np.min(y))/np.sqrt(np.shape(y)[0])

    # creating color_values for the figure
    offset = z + np.abs(z.min())
    fracs = offset.astype(float)/offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.jet(norm(fracs.tolist()))

    # Set up and display the plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.bar3d(x,y,bottom,width,depth,z, color=color_values, shade=True)
    ax.set_xlabel("P(add)")
    ax.set_ylabel("P(remove)")
    ax.set_zlabel("fails")
    ax.set_xticks([0,(nrows-1)/4, (nrows-1)/2, 3*(nrows-1)/4, (nrows-1)],[0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0,(nrows-1)/4, (nrows-1)/2, 3*(nrows-1)/4, (nrows-1)],[0, 0.25, 0.5, 0.75, 1])
    plt.title("Two parameter variation")
    plt.show()

def test2d(resolution, sample_size, MCMCiterations, savefile=None, loadfile=None):
    if loadfile is not None:
        with open(loadfile, "rb") as f:
            y = pickle.load(f)    
    else:
        random.seed(2)
        np.random.seed(2)
        no_nodes = 6
        no_colors = 3
        edge_probability = 0.3

        real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_probability)
        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

        
        # RUN MCMC
        # Fully random colored DAG
        y = np.zeros((resolution))

        for i in range(1,resolution):
            random.seed(2)
            np.random.seed(2)
            s = i/(2*resolution)
            print(s)

            _, _, _, _, fails = MCMCfuncs.CausalMCMC(samples, MCMCiterations, move_weights=[s,s], debug=True)
            y[i] = fails/MCMCiterations
            
        if savefile is not None:
            with open(savefile, "wb") as f:
                pickle.dump(y, f)

    # Plot data
    x = np.linspace(0,0.5,resolution)
    plt.plot(x,y)
    plt.xlabel("P(add)=P(remove)")
    plt.ylabel("fails")
    plt.title("One parameter variation")
    plt.show()


def main():
    load1 = "plot3d40res.pkl"
    load2 = "plot2d50res.pkl"
    save = None #"lastsave.pkl"
    test3d(resolution = 10, sample_size = 1000, MCMCiterations = 1000, savefile=save, loadfile=None)

    test2d(resolution = 50, sample_size = 1000, MCMCiterations = 1000, savefile=save, loadfile=None)


    

    
if __name__ == "__main__":
    main()
