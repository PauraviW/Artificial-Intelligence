#!/usr/local/bin/python3
# solve_luddy.py : Sliding tile puzzle solver
#
# Code by: [Vidit Mohaniya - Pauravi Wagh - Vibhas Vats vimohan - pwagh - vkvats]
#
# Based on skeleton code by D. Crandall, September 2019
#
import heapq
import sys
# import time
# from queue import PriorityQueue

MOVES_original = {(0, -1): "R", (0, 1): "L", (-1, 0): "D", (1, 0): "U"}

MOVES_circular = {(0, -1): "R", (0, 1): "L", (-1, 0): "D", (1, 0): "U",
                  (0, 3): "L", (0, -3): "R", (-3, 0): "U", (3, 0): "D"}

MOVES_Luddy = {(1, 2): "E", (1, -2): "F", (-1, 2): "G", (-1, -2): "H",
               (-2, -1): "D", (-2, 1): "C", (2, -1): "B", (2, 1): "A"}


def rowcol2ind(row, col):
    return row*4 + col


def ind2rowcol(ind):
    return int(ind/4), ind % 4


def valid_index(row, col):
    return 0 <= row <= 3 and 0 <= col <= 3


def swap_ind(list0, ind1, ind2):
    return list0[0:ind1] + (list0[ind2],) + list0[ind1+1:ind2] + (list0[ind1],) + list0[ind2+1:]


def swap_tiles(state, row1, col1, row2, col2):
    return swap_ind(state, *(sorted((rowcol2ind(row1, col1), rowcol2ind(row2, col2)))))


def printable_board(row):
    return ['%3d %3d %3d %3d' % (row[j:(j+4)]) for j in range(0, 16, 4)]


def original(state, row, col):
    return [(swap_tiles(state, row, col, row + i, col + j), c)
            for ((i, j), c) in MOVES_original.items() if valid_index(row + i, col + j)]


def circular(state, row, col):
    return [(swap_tiles(state, row, col, row + i, col + j), c)
            for ((i, j), c) in MOVES_circular.items() if valid_index(row + i, col + j)]


def luddy(state, row, col):
    return [(swap_tiles(state, row, col, row + i, col + j), c)
            for ((i, j), c) in MOVES_Luddy.items() if valid_index(row + i, col + j)]


# return a list of possible successor states
def successors(state, move):
    (empty_row, empty_col) = ind2rowcol(state.index(0))  # find the empty tile index number.
    return move(state, empty_row, empty_col)


# check if we've reached the goal
def is_goal(state):
    return sorted(state[:-1]) == list(state[:-1]) and state[-1] == 0


def circular_manhattan_dist(board):  # to be used with circular method only.
    final_board = sorted(board)[1:]
    final_board.append(0)
    m_distance = 0
    for i in board:
        r1, c1 = ind2rowcol(board.index(i))
        r2, c2 = ind2rowcol(final_board.index(i))
        if abs(r1-r2) == 3:
            m_distance += 1 + abs(c2-c1)
        elif abs(c2-c1) == 3:
            m_distance += 1 + abs(r2-r1)
        else:
            m_distance += abs(r1 - r2) + abs(c1 - c2)
    return m_distance


def misplaced_tile(g, board):  # to be used with original method only.
    final_board = sorted(board)[1:]
    misplaced = 0
    for value in final_board:
        if final_board.index(value) != board.index(value):
            misplaced += 1
    return g + misplaced


def luddy_misplaced_tile(g, board):  # to be used with original method only.
    final_board = sorted(board)[1:]
    misplaced = 0
    for value in final_board:
        if final_board.index(value) != board.index(value):
            misplaced += 1
    return misplaced


# The solver! - using BFS right now


def solve(initial_board, method0):
    fringe = []
    steps_so_far = 0
    heapq.heappush(fringe, (steps_so_far, (initial_board, "")))

    # Parity Check
    if misplaced_tile(steps_so_far, initial_board) % 2 != 0 and method0 == "original":
        return False
    method1 = eval(method0)
    while fringe:  # PriorityQueue is not iterable and it has no length, need to find out when to break while loop.
        (state, route_so_far) = heapq.heappop(fringe)[-1]
        steps_so_far += 1
        for (succ, move) in successors(state, method1):  # current state is being sent to successor function
            if is_goal(succ):
                return route_so_far + move
            elif method0 == "circular":
                heapq.heappush(fringe, (circular_manhattan_dist(succ), (succ, route_so_far + move)))
            elif method0 == "original":
                heapq.heappush(fringe, (misplaced_tile(steps_so_far, succ), (succ, route_so_far + move)))
    return False


def solve_luddy(initial_board, method0):
    fringe = []
    steps_so_far = 0
    number_of_misplaced_tiles = luddy_misplaced_tile(0, initial_board)
    heapq.heappush(fringe, (steps_so_far + number_of_misplaced_tiles, (initial_board, steps_so_far, "")))

    method1 = eval(method0)
    visited1 = {initial_board: True}
    while fringe:
        element = list(heapq.heappop(fringe))
        state = element[1][0]
        steps_so_far = element[1][1]
        route_so_far = element[1][2]
        for (succ, move) in successors(state, method1):
            if is_goal(succ):
                return route_so_far + move
            elif not visited1.get(succ, False):
                heapq.heappush(fringe, (luddy_misplaced_tile(0, succ)+steps_so_far+1, (succ, steps_so_far+1, route_so_far + move)))
                visited1[succ] = True
    return False


# test cases
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise(Exception("Error: expected 2 arguments"))
    # start_time = time.time()
    start_state = []
    with open(sys.argv[1], 'r') as file:  # sys.argv[1]
        for line in file:
            start_state += [int(i) for i in line.split()]
    # method = "circular"   # hard coded for now.
    method = sys.argv[2].lower()  # assign the original, circular and luddy.

    if len(start_state) != 16:
        raise(Exception("Error: couldn't parse start state file"))
    print("Start state: \n" + "\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    if method != "luddy":
        route = solve(tuple(start_state), method)
    else:
        route = solve_luddy(tuple(start_state), method)
    if route:
        print("Solution found in " + str(len(route)) + " moves:" + "\n" + route)
    else:
        print("Inf")
    # print("--- %s seconds ---" % (time.time() - start_time))
