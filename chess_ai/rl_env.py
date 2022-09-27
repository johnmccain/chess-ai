import logging
from optparse import Option
from os import stat
from typing import List, Dict, Set, Tuple, Optional, Iterator, Union
from numpy.typing import ArrayLike
import numpy as np
from enum import IntEnum

import torch

from chess_ai.game import Piece, PieceColor, Game, Board, Move, MoveValidator, GameResolution


RULE_PENALTY = -500
WIN_REWARD = 200
LOSE_REWARD = -200


class ChessBoardScorer:

    def __init__(self, game: Game):
        self.game = game

    @staticmethod
    def get_piece_score(piece: Union[Piece, int]) -> float:
        if piece in (Piece.B_PAWN, Piece.W_PAWN):
            return 1.0
        elif piece in (Piece.B_KNIGHT, Piece.W_KNIGHT):
            return 3.0
        elif piece in (Piece.B_BISHOP, Piece.W_BISHOP):
            return 3.0
        elif piece in (Piece.B_ROOK, Piece.W_ROOK):
            return 5.0
        elif piece in (Piece.B_QUEEN, Piece.W_QUEEN):
            return 9.0
        else:
            return 0.0

    @staticmethod
    def score_board(board: Board, color: PieceColor) -> float:
        score = 0.0
        for x, y, piece in board.get_pieces(color):
            score += ChessBoardScorer.get_piece_score(piece)
        opp_score = 0.0
        for x, y, piece in board.get_pieces(PieceColor.opposite(color)):
            opp_score += ChessBoardScorer.get_piece_score(piece)
        
        return score - opp_score


class ChessEnv:
    """
    Environment class for interaction with RL agent.
    """

    def __init__(self):
        self.game = Game()

    def reset(self):
        self.game = Game()

    def get_state(self) -> torch.Tensor:
        # (8 x 8)
        board = self.game.board.board
        board_one_hot = np.zeros((8, 8, 13))
        board_one_hot[np.arange(8), np.arange(8), board] = 1
        # remove dim for empty space indicator, implicit
        # (8 x 8 x 13) -> (8 x 8 x 12)
        board_one_hot = board_one_hot[:, :, 1:]
        # (8 x 8 x 12) -> (12 x 8 x 8)
        return torch.tensor(board_one_hot.astype(np.float32), dtype=torch.float32).permute(2, 0, 1)

    def get_reward(self) -> float:
        resolution = self.game.get_resolution()
        if resolution == GameResolution.WHITE_WINS:
            return WIN_REWARD
        elif resolution == GameResolution.BLACK_WINS:
            return LOSE_REWARD
        else:
            return ChessBoardScorer.score_board(self.game.board, self.game.turn)

    def take_and_score_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Take an action and return the new state, reward, and whether the game is over.
        An action is defined as two coordinates (piece to select and space to move to).
        If the action would violate the rules, then a penalty reward is returned and the state remains the same.
        """
        x1, y1 = round(action[0, 0].item()), round(action[0, 1].item())
        x2, y2 = round(action[1, 0].item()), round(action[1, 1].item())
        move = Move(x1, y1, x2, y2, self.game.board.get_piece(x1, y1))
        if self.game.move(move):
            # move was legal
            # get reward
            reward = self.get_reward()
            # get new state
            state = self.get_state()
            # check if game is over
            done = self.game.get_resolution() != GameResolution.UNDECIDED
            # swap to other player
            self.game = self.game.swap()
            return state, reward, done
        else:
            # move was illegal
            # return penalty reward
            return self.get_state(), RULE_PENALTY, False
