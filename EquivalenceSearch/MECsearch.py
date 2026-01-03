import itertools
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utils


def bell(n):
    tot = 0
    for k in range(n+1):
        tot1 = 0
        for i in range(k+1):
            tot1 += (-1)**(k-i) * math.comb(k, i) * i**n
        tot += tot1 / math.factorial(k)
    return int(tot)

def partition(collection):
    if len(collection) == 1 or len(collection) == 0:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller


def main():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 4
    
    
    _, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, 2, 0.8)
    samples = utils.generate_sample(100, real_lambda_matrix, real_omega_matrix)

    # A = np.array([  [0, 1, 0, 0],
    #                 [0, 0, 0, 0],
    #                 [0, 0, 0, 0],
    #                 [0, 0, 0, 0]])
    # B = np.array([  [0, 0, 0, 0],
    #                 [1, 0, 0, 0],
    #                 [0, 0, 0, 0],
    #                 [0, 0, 0, 0]])
    # PE = [[(0,1)]]
    # PN = [[x] for x in range(num_nodes)]
    # score = np.round(utils.score_DAG(samples, A, PE, PN), 11)
    # print(score)

    # PE = [[(1,0)]]
    # PN = [[x] for x in range(num_nodes)]
    # score = np.round(utils.score_DAG(samples, B, PE, PN), 11)
    # print(score)

    # quit()

    possible_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge = (i,j)
            possible_edges.append(edge)
    

    for num_edges in range(int(num_nodes*(num_nodes-1)/2)+1):
        for chosen_edges in itertools.combinations(possible_edges, num_edges):
            A = np.zeros((num_nodes,num_nodes))
            for edge in chosen_edges:
                A[edge] = 1

            if not utils.is_DAG(A):
                continue

            MEClist = utils.enum_MEC(A)
            MECsize = len(MEClist)
            print("MECsize:", MECsize)

            print("Number of edge partitions:", bell(len(chosen_edges)))

            i = 0
            for DAG_B in MEClist:
                i += 1
                print(f"DAG {i}:")

                for edge_partition in partition(list(chosen_edges)):
                    
                    # NEED TO CHANGE TO PARTITION OF UNORDERED PAIRS (i,j) 


                    PE = [x for x in edge_partition if len(x)>0]
                    PN = [[x] for x in range(num_nodes)]
                    print(PE)
                    print(PN)
                    score = np.round(utils.score_DAG(samples, DAG_B, PE, PN), 11)
                    print(DAG_B)
                    print(score)


            time.sleep(0.1)
            




 

    
if __name__ == "__main__":
    main()
