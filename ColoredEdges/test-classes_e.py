import itertools
import random
import time

import numpy as np
import utils


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
    random.seed(4)
    np.random.seed(4)

    visits = {}

    num_nodes = 4
    sample_size = 100

    _, _, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, 2,2, 0.8)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    possible_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge = (i,j)
            possible_edges.append(edge)


    t = time.perf_counter()
    for num_edges in range(int(num_nodes*(num_nodes-1)/2)+1):
        for chosen_edges in itertools.combinations(possible_edges, num_edges):
            A = np.zeros((num_nodes,num_nodes))
            for edge in chosen_edges:
                A[edge] = 1

            if not utils.is_DAG(A):
                continue

            for edge_partition in partition(list(chosen_edges)):
                PE = [x for x in edge_partition if len(x)>0]

                if len(PE) == num_edges:    # Only look through "true" compcolored models
                    continue

                PN = [[x] for x in range(num_nodes)]
                score = np.round(utils.score_DAG(samples, A, PE, PN), 11)

                if np.abs( score+ 3.5810377596) < 0.00000001:
                    print(PE)
                    print(utils.score_DAG(samples, A, PE, PN))


                if score in visits:
                    visits[score] += 1

                    # if visits[score] == 2:
                    #     for part in PE:
                    #         end = part[0][1]
                    #         for edge in part:
                    #             if edge[1] != end:
                    #                 print(PE)
                    #                 print(score)
                else:
                    visits[score] = 1
                    

    #print(visits.values())
    print("Max:", max(visits.values()))
    print("Count:", list(visits.values()).count(max(visits.values())))
    print("It took", time.perf_counter()-t)


if __name__ == "__main__":
    main()
