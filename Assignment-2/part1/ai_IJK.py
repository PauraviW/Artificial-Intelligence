#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: [vmohan Vidit Mohaniya pwagh Pauravi Wagh vkvats Vibhas Vats]

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
import random


# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game 
#
# This function should analyze the current state of the game and determine the 
# best move for the current player. It should then call "yield" on that move.

def next_move(game: Game_IJK) -> None:
    '''board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    '''

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()

    # You'll want to put in your fancy AI code here. For right now this just 
    # returns a random move.
    yield MINIMAX_Decision(game, deterministic, h=5)


def successors(board):
    return [(board.makeMove(move), move) for move in ['U', 'D', 'L', 'R']]


def MINIMAX_Decision(board, deterministic, h=3):
    succ = successors(board)
    if deterministic:
        heuristic_values = [MAX_Value(s[0], h, s[1], -1000, 1000) for s in succ]
        decision = succ[heuristic_values.index(max(heuristic_values))]
        return decision[-1]
    else:
        heuristic_values = [MAX_Value_Non_det(s[0], h, s[1]) for s in succ]
        decision = succ[heuristic_values.index(max(heuristic_values))]
        return decision[-1]


def MAX_Value(board, h, move, alpha, beta):
    h = h - 1
    max_value = -1000
    if h <= 0 or board.state() != 0:
        return heuristic(board)
    else:
        for s in successors(board):
            value = MIN_Value(s[0], h, s[1], alpha, beta)
            if value > max_value:
                max_value = value

            if max_value >= beta:
                return max_value

            if max_value > alpha:
                alpha = max_value
        return max_value


def MIN_Value(board, h, move, alpha, beta):
    h = h - 1
    min_value = 1000
    if h <= 0 or board.state() != 0:
        return heuristic(board)
    else:
        for s in successors(board):
            value = MAX_Value(s[0], h, s[1], alpha, beta)
            if value < min_value:
                min_value = value

            if min_value <= alpha:
                return min_value

            if min_value < beta:
                beta = min_value
        return min_value


def heuristic(board):
    sum_AK = 0
    sum_ak = 0
    board = board.getGame()
    for i in range(len(board)):
        for j in range(len(board[0])):
            if 'A' <= board[i][j] <= 'Z':
                sum_AK += ord(board[i][j])
            elif 'a' <= board[i][j] <= 'z':
                sum_ak += ord(board[i][j]) - 32
    return sum_AK - sum_ak


def MAX_Value_Non_det(board, h, move):
    h = h - 1
    max_value = -1000
    if h <= 0 or board.state() != 0:
        return heuristic(board)
    else:
        for s in successors(board):
            value = Chance_Value_Non_det(s[0], h, s[1], 'min')
            if value > max_value:
                max_value = value
        return max_value


def MIN_Value_Non_det(board, h, move):
    h = h - 1
    min_value = 1000
    if h <= 0 or board.state() != 0:
        return heuristic(board)
    else:
        for s in successors(board):
            value = Chance_Value_Non_det(s[0], h, s[1], 'max')
            if value < min_value:
                min_value = value
        return min_value


def Chance_Value_Non_det(board, h, move, next_layer):
    if next_layer == 'min':
        return MIN_Value_Non_det(board, h, move) * (1/number_of_empty_slots(board))
    elif next_layer == 'max':
        return MAX_Value_Non_det(board, h, move) * (1/number_of_empty_slots(board))


def number_of_empty_slots(board):
    open_slots = 1
    board = board.getGame()
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == '':
                open_slots += 1
    return open_slots
