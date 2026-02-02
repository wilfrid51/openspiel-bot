import numpy as np
from collections import deque

def score_moves(board, player):
    scores = {}
    opponent = 3 - player
    for move in legal_moves(board):
        temp_board = board.copy()
        temp_board.play(move, player)
        my_conn = connectivity_score(temp_board, player)
        opp_conn = connectivity_score(temp_board, opponent)
        template_score = template_bonus(temp_board, player)
        forcing = forcing_bonus(temp_board, player, opponent)
        center_bonus = positional_weight(move)
        scores[move] = (my_conn - opp_conn) + template_score + forcing + center_bonus
    return scores

best_move = max(score_moves(board, current_player), key=lambda m: scores[m])