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
Overview of Simulated Annealing:
Simulated Annealing is an algorithm that works based on a given set of information (a function or a set of data) to find the most optimal
solution for some purpose. In metallurgy, annealing is defined as slowly cooling down a heated substance in order to
more easily work with it. 

Simulated annealing is based on this concept, and implements a temperature variable that slowly goes down after
iterations, as well as randomizing values. This temperature variable is initialized at a given value. 
It divides delta_e in: prob = np.exp(- delta_e / (temperature + 1e-6)) (Boltzmann Distrubtion) to decide on the next value to go to. 
Without the temperature value, it would just be a greedy algorithm in that it chooses the best value to go to at every iteration.

The greedy algorithm is more easily subject to local optima, meaning it can end before reaching the ultimate optimal solution/path. Simulated
annealing can be more useful for this, as it does not choose "the best" value every time, as to not fall into those local optima so early on.
However, as the temperature value goes down, it's effect decreases, and the algorithm becomes more prone to optima.
'''

'''
Function description(s):
The function below takes in these parameters: initial temperature, the number of steps we will take on a graph, and the graph itself.
The graph is then set up as a section of an equal number of 0's and 1's, which will be used to partition the MaxCut graph; can be denoted as 
S or T as well. The function will loop the number of times as given by the num_steps parameter. In the loop, it decrements the temperature
value based on the current step number k, and the total number of steps: temperature = init_temperature * (1 - (k + 1) / num_steps).
Next it creates a temporary copy of the initial solution graph, and the succeeding code before the if statement is basically to find a
random value to explore next, store the current score value in a list, and computes the difference between the current score and the new
solution's score (delta_e). Using this, we can determine if the new solution's score is better or worse than the curr score. If it is better,
we just update the solution with the new solution, else we use probability to determine whether to keep the solution or replace with the new
one, so as to avoid getting stuck in local optima. This is where temperature comes in, where the variable prob is calculated.
If prob is greater than a randomly generated value (from random.random()), we accept the worse solution, encouraging exploration in the 
early stages of the algorithm instead of choosing the best option (possibly leading to a local optima).

Output/Print Statements:
The output of the program is as follows:
1) The score achieved after the program run + the initial score. The score is the cut value (how many edges are cut when dividing the
graph into two sets).

2) A list of the scores over time/iterations. This shows how to score grew and was altered for each step.

3) A list representing the final partition of nodes 1 and 0.

4) Running duration (the shorter the better)
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






