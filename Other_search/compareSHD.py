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
import utils
from Tabufuncs import CausalTabuSearch


def pa(i, A):
    return set(np.where(np.logical_and(A[:, i] != 0, A[i, :] == 0))[0])

def ch(i, A):
    return set(np.where(np.logical_and(A[i, :] != 0, A[:, i] == 0))[0])

def is_DAG(A):
    numnodes = A.shape[0]
    if numnodes < 8:
        P = A
    else:
        P = A.astype(np.float32)
    power = 1
    while power < numnodes:
        P = P @ P
        power *= 2
    if np.argmax(P) != 0:
        return False
    return not P[0,0]

def topological_ordering(A):
    # Run the algorithm from the 1962 paper "Topological sorting of
    # large networks" by AB Kahn
    A = A.copy()
    sinks = list(np.where(A.sum(axis=0) == 0)[0])
    ordering = []
    while len(sinks) > 0:
        i = sinks.pop()
        ordering.append(i)
        for j in ch(i, A):
            A[i, j] = 0
            if len(pa(j, A)) == 0:
                sinks.append(j)
    # If A still contains edges there is at least one cycle
    if A.sum() > 0:
        raise ValueError("The given graph is not a DAG")
    else:
        return ordering
    
def order_edges(G):
    if not is_DAG(G):
        raise ValueError("The given graph is not a DAG")
    # i.e. if i -> j, then i appears before j in order
    order = topological_ordering(G)
    # You can check the above by seeing that np.all([i == order[pos[i]] for i in range(p)]) is True
    # Unlabelled edges as marked with -1
    ordered = (G != 0).astype(int) * -1
    i = 1
    while (ordered == -1).any():
        # let y be the lowest ordered node that has an unlabelled edge
        # incident to it
        froms, tos = np.where(ordered == -1)
        with_unlabelled = np.unique(np.hstack((froms, tos)))
        y = sort(with_unlabelled, reversed(order))[0]
        # let x be the highest ordered node s.t. the edge x -> y
        # exists and is unlabelled
        unlabelled_parents_y = np.where(ordered[:, y] == -1)[0]
        x = sort(unlabelled_parents_y, order)[0]
        ordered[x, y] = i
        i += 1
    return ordered

def sort(L, order=None):
    L = list(L)
    if order is None:
        return sorted(L)
    else:
        order = list(order)
        pos = np.zeros(len(order), dtype=int)
        pos[order] = range(len(order))
        positions = [pos[l] for l in L]
        return [tup[1] for tup in sorted(zip(positions, L))]
    
def label_edges(ordered):
    # Validate the input
    if not is_DAG(ordered):
        raise ValueError("The given graph is not a DAG")
    no_edges = (ordered != 0).sum()
    if sorted(ordered[ordered != 0]) != list(range(1, no_edges + 1)):
        raise ValueError("The ordering of edges is not valid:", ordered[ordered != 0])
    # define labels: 1: compelled, -1: reversible, -2: unknown
    COM, REV, UNK = 1, -1, -2
    labelled = (ordered != 0).astype(int) * UNK
    # while there are unknown edges
    while (labelled == UNK).any():
        # print(labelled)
        # let (x,y) be the unknown edge with lowest order
        # (i.e. appears last in the ordering, NOT has smalles label)
        # in ordered
        unknown_edges = (ordered * (labelled == UNK).astype(int)).astype(float)
        unknown_edges[unknown_edges == 0] = -np.inf
        # print(unknown_edges)
        (x, y) = np.unravel_index(np.argmax(unknown_edges), unknown_edges.shape)
        # print(x,y)
        # iterate over all edges w -> x which are compelled
        Ws = np.where(labelled[:, x] == COM)[0]
        end = False
        for w in Ws:
            # if w is not a parent of y, label all edges into y as
            # compelled, and finish this pass
            if labelled[w, y] == 0:
                labelled[list(pa(y, labelled)), y] = COM
                end = True
                break
            # otherwise, label w -> y as compelled
            else:
                labelled[w, y] = COM
        if not end:
            # if there exists an edge z -> y such that z != x and z is
            # not a parent of x, label all unknown edges (this
            # includes x -> y) into y with compelled; label with
            # reversible otherwise.
            z_exists = len(pa(y, labelled) - {x} - pa(x, labelled)) > 0
            unknown = np.where(labelled[:, y] == UNK)[0]
            assert x in unknown
            labelled[unknown, y] = COM if z_exists else REV
    return labelled

def dag_to_cpdag(G):
    # 1. Perform a total ordering of the edges
    ordered = order_edges(G)
    # 2. Label edges as compelled or reversible
    labelled = label_edges(ordered)
    # 3. Construct CPDAG
    cpdag = np.zeros_like(labelled)
    # set compelled edges
    cpdag[labelled == 1] = labelled[labelled == 1]
    # set reversible edges
    fros, tos = np.where(labelled == -1)
    for (x, y) in zip(fros, tos):
        cpdag[x, y], cpdag[y, x] = 1, 1
    return cpdag




def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 6
    no_colors = 3
    sparse = True
    MCMC_iterations = 10_000

    num_tests = 10


    df = pd.DataFrame(columns=["DAG_ID", "sample size", "Algorithm", "SHD"])


    for dag_id in range(num_tests):
        print(dag_id)
        for num_samples in [100, 500, 1000]:
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, sparse)
            real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)


            # GES estimate of graph
            samples = utils.generate_sample(num_samples, real_lambda_matrix, real_omega_matrix)
            res = ges.fit_bic(data=samples)
            GES_edge_array = res[0]

            MCMC_edge_array, partition, bic, _, _ = CausalTabuSearch(samples, MCMC_iterations)


            MCMC_error = utils.calc_SHD(dag_to_cpdag(real_edge_array), dag_to_cpdag(MCMC_edge_array))
            GES_error = utils.calc_SHD(dag_to_cpdag(real_edge_array), GES_edge_array)
            
            df.loc[-1] = [dag_id, num_samples, "MCMC", GES_error]
            df.index = df.index + 1
            df = df.sort_index()

            df.loc[-1] = [dag_id, num_samples, "GES", MCMC_error]
            df.index = df.index + 1
            df = df.sort_index()

    print(df)
    #df.to_csv('out.csv', index=False) 
    
    # Creating plot
    sns.boxplot(data=df, x="sample size", y="SHD", hue="Algorithm")
    plt.title("Boxplot: p=6, sparse, tests=100, iters=10_000")
    
    plt.show()

if __name__ == "__main__":
    main()