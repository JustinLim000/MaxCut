import sys

import copy
import time
from typing import List, Union
import numpy as np
import random
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut

''' 
Simulated Annealing is an algorithm that works based on a given set of information (a function or a set of data) to find the most optimal
solution for some purpose. In metallurgy, annealing is defined as slowly cooling down a heated substance in order to
more easily work with it. 

Simulated annealing is based on this concept, and implements a temperature variable that slowly goes down after
iterations, as well as randomizing values. This temperature variable is initialized at a given value. 
It divides delta_e in: prob = np.exp(- delta_e / (temperature + 1e-6)) to decide on the next value to go to. 
Without the temperature value, it would just be a greedy algorithm in that it chooses the best value to go to at every iteration.

The greedy algorithm is more easily subject to local minima, meaning it can end before reaching the ultimate optimal solution/path. Simulated
annealing can be more useful for this, as it does not choose "the best" value every time, as to not fall into those local minima so early on.
However, as the temperature value goes down, it's effect decreases, and the algorithm becomes more prone to minima.
'''

def simulated_annealing(init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')

    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)

    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    num_nodes = len(init_solution)
    scores = []
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        new_solution = copy.deepcopy(curr_solution)
        idx = np.random.randint(0, num_nodes)
        new_solution[idx] = (new_solution[idx] + 1) % 2
        new_score = obj_maxcut(new_solution, graph)
        scores.append(new_score)
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
        else:
            prob = np.exp(- delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
    print("score, init_score of simulated_annealing", curr_score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':


    # run alg
    # init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))

    # read data
    graph = read_nxgraph('./data/syn/syn_50_176.txt')
    init_temperature = 4
    num_steps = 2000
    sa_score, sa_solution, sa_scores = simulated_annealing(init_temperature, num_steps, graph)






