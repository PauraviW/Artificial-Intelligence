#!/usr/local/bin/python3
#
# choose_team.py : Choose a team of maximum skill under a fixed budget
#
# Code by: [vimohan pwagh vkvats Vidit Mohaniya Pauravi Wagh Vibhas Vats]
#
# Based on skeleton code by D. Crandall, September 2019
#
import sys


def load_people(filename):
    people = {}
    with open(filename, "r") as file:
        for line in file:
            l = line.split()
            people[l[0]] = [float(i) for i in l[1:]]
    return people

"""
This function implements breadth- first search algorithm on a list that is sorted on the basis of the skill/cost rate.
"""
def generate_successor_nodes(sorted_list, person, skill, cost, budget, index):
    fringe = [([person], skill, cost, index)]
    highestX = ([person], skill, cost, index)

    while len(fringe) > 0:
        # pop on the basis of the highest rate
        curr_node = fringe.pop(0)
        for i in range(curr_node[3]+1, len(sorted_list)):
            (p1, (s1, c1)) = sorted_list[i]
            people_list = curr_node[0].copy()
            if curr_node[2] + c1 <= budget:
                people_list.append(p1)
                new_skill = curr_node[1] + s1
                new_cost = curr_node[2] + c1
                new_index = i
                new_node = (people_list, new_skill, new_cost, new_index)
                fringe.append(new_node)
                if new_skill > highestX[1]:
                    highestX = new_node
    return highestX


# This function implements a greedy solution to the problem:
#  It adds people in decreasing order of "skill per dollar,"
#  until the budget is exhausted. It exactly exhausts the budget
#  by adding a fraction of the last person.
#
def approx_solve(people, budget):
    # initial state
    solution = ()
    best_combination = [[],0,0,0]
    sorted_list = sorted(people.items(),
                         key=lambda x: x[1][0] / x[1][1], reverse=True)
    for i, (person, (skill, cost)) in enumerate(sorted_list):
        """
        generate successor states for every person in the list
        """
        if cost <= budget:
            combination = generate_successor_nodes(
                    sorted_list, person, skill, cost, budget, i)
            if combination[1] > best_combination[1]:
                best_combination = combination

    for i in best_combination[0]:
        solution += ((i, 1),)
    return solution


if __name__ == "__main__":

    if (len(sys.argv) != 3):
        raise Exception('Error: expected 2 command line arguments')

    budget = float(sys.argv[2])
    people = load_people(sys.argv[1])
    solution = approx_solve(people, budget)
    if len(solution) < 1:
        print("Inf")
    else:
        print("Found a group with %d people costing %f with total skill %f" % \
            (len(solution), sum(people[p][1] * f for p, f in solution), sum(people[p][0] * f for p, f in solution)))

    for s in solution:
        print("%s %f" % s)
