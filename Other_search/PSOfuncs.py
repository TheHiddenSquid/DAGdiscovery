import random
import numpy as np
import copy
import utils

# Main MCMC function

def CausalPSO(samples, num_iters, num_particles):

    # Other settings
    num_nodes = samples.shape[1]


    global_best_bic = -np.infty
    global_best_A = None
    global_best_P = None
    
    particles = []
    # Particle is [DAG, Coloring, score_info, BEST_DAG, BEST_coloring]
    for _ in range(num_particles):
        P, A, _ = utils.generate_colored_DAG(num_nodes, num_nodes, 0.5)
        A = np.array(A != 0, dtype="int")
        bic = score_DAG(samples, A, P)
        particles.append([A, P, bic, A.copy(), copy.deepcopy(P)])

        if bic > global_best_bic:
            global_best_bic = bic
            global_best_A = A
            global_best_P = copy.deepcopy(P)


    alledges = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    moves = ["stay", "goto_global_best", "goto_local_best"]
    for k in range(num_iters):
        print(k)
        for i in range(num_particles):
            A = particles[i][0]
            P = particles[i][1]
            bic_locbest = particles[i][2]
            A_locbest = particles[i][3]
            P_locbest = particles[i][4]

            r_choices = random.choices(moves, weights =[1/3,1/3,1/3], k = len(alledges)+num_nodes)
            for edge in alledges:
                move = r_choices.pop()
                A_tmp = A.copy()
                if move == "goto_global_best":
                    A_tmp[edge] = global_best_A[edge]
                    A_tmp[(edge[1],edge[0])] = global_best_A[(edge[1],edge[0])]

                if move == "goto_local_best":
                    A_tmp[edge] = A_locbest[edge]
                    A_tmp[(edge[1],edge[0])] = A_locbest[(edge[1],edge[0])]

                if utils.is_DAG(A_tmp):
                    A = A_tmp
                
            for node in range(num_nodes):
                move = r_choices.pop()
                if move == "goto_global_best":
                    for j, part in enumerate(P):
                        if node in part:
                            current_color = j
                            break
                    P[current_color].remove(node)
                    
                    for j, part in enumerate(global_best_P):
                        if node in part:
                            current_color = j
                            break
                    P[current_color].append(node)

                if move == "goto_local_best":
                    for j, part in enumerate(P):
                        if node in part:
                            current_color = j
                            break
                    P[current_color].remove(node)
                    
                    for j, part in enumerate(P_locbest):
                        if node in part:
                            current_color = j
                            break
                    P[current_color].append(node)


            bic = score_DAG(samples, A, P)
            if bic > bic_locbest:
                particles[i][2] = bic
                particles[i][3] = A
                particles[i][4] = P

                if bic > global_best_bic:
                    global_best_bic = bic
                    global_best_A = A
                    global_best_P = P

        

    return global_best_A, global_best_P, global_best_bic

    
    


# For DAG heuristic

def score_DAG(samples, edge_array, partition):
    samples = np.transpose(samples)

    num_nodes = samples.shape[0]
    num_samples = samples.shape[1]
    num_colors = sum(1 for x in partition if len(x)>0)

    # Calculate ML-eval of the different lambdas
    edges_ML = np.zeros((num_nodes,num_nodes), dtype="float")
    for i in range(num_nodes):
        parents = utils.get_parents(i, edge_array)
        ans = np.linalg.lstsq(np.transpose(samples[parents,:]), np.transpose(samples[i,:]), rcond=None)[0]
        edges_ML[parents, i] = ans

    # Calculate ML-eval of the different color omegas
    omegas_ML = [None] * len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        tot = 0
        for node in part:
            parents = utils.get_parents(node, edge_array)
            tot += np.linalg.norm(samples[node,:]-np.matmul(np.transpose(edges_ML[parents,node]), samples[parents,:]))**2
        omegas_ML[i] = tot / (num_samples * len(part))


    # Calculate BIC

    bic_decomp = [0]*len(partition)

    for i, part in enumerate(partition):
        if len(part) == 0:
            continue
        bic_decomp[i] = -len(part) * np.log(omegas_ML[i]) - len(part) - (np.log(num_samples)/num_samples) * sum(len(utils.get_parents(x, edge_array)) for x in part)
    
    bic = sum(bic_decomp) / 2
    bic = bic - np.log(num_samples)/(num_samples*2) * num_colors


    return bic



def main():
    pass



if __name__ == "__main__":
    main()
