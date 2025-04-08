import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../")
import MCMCfuncs
import utils


def main():

    cols = [f"n = {x}" for x in [100,500,1000]]
    rows = [f"p = {x}" for x in [4,6,8]]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    plt.setp(axes.flat, xlabel='edge probability', ylabel='SHD')

    pad = 5 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)



    size_options = [4,6,8]
    sample_options = [100, 500, 1000]

    df = pd.read_csv("out2colors.csv")


    for row in [0,1,2]:
        tmp1_df = df[df["num_nodes"] == size_options[row]]

        for col in [0,1,2]:
            plt.subplot(3,3, 3*row+col+1)
            tmp2_df = tmp1_df[tmp1_df["num_samples"] == sample_options[col]]
            sns.boxplot(data=tmp2_df, x="edge_prob", y="SHD", hue="Algorithm")
  
    plt.show()

if __name__ == "__main__":
    main()