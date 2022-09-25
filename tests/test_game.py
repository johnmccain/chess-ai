import pytest

from chess_ai.game import Piece, PieceColor, Game, Board, Move, MoveValidator

def test_notation_to_space_is_reversable():
    all_notations = []
    for row in "12345678":
        for col in "abcdefgh":
            all_notations.append(f"{col}{row}")

    for notation in all_notations:
        coords = Board.notation_to_space(notation)
        assert Board.space_to_notation(*coords) == notation


def test_space_to_notation_is_reversable():
    all_spaces = []
    for row in range(8):
        for col in range(8):
            all_spaces.append((col, row))

    for space in all_spaces:
        notation = Board.space_to_notation(*space)
        assert Board.notation_to_space(notation) == space
