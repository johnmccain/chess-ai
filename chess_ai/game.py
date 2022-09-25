from multiprocessing.sharedctypes import Value
from typing import List, Dict, Set, Tuple, Optional, Iterator
import numpy as np


class PieceColor:
    WHITE = 0
    BLACK = 1

    @staticmethod
    def opposite(color):
        return PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE


class Piece:
    EMPTY = 0

    # Black
    B_PAWN = 1
    B_ROOK = 2
    B_BISHOP = 3
    B_KNIGHT = 4
    B_QUEEN = 5
    B_KING = 6

    # White
    W_PAWN = 7
    W_ROOK = 8
    W_BISHOP = 9
    W_KNIGHT = 10
    W_QUEEN = 11
    W_KING = 12

    @staticmethod
    def color(piece: int) -> Optional[int]:
        if piece in (Piece.B_PAWN, Piece.B_ROOK, Piece.B_BISHOP, Piece.B_KNIGHT, Piece.B_QUEEN, Piece.B_KING):
            return PieceColor.BLACK
        elif piece in (Piece.W_PAWN, Piece.W_ROOK, Piece.W_BISHOP, Piece.W_KNIGHT, Piece.W_QUEEN, Piece.W_KING):
            return PieceColor.WHITE
        elif piece == Piece.EMPTY:
            return None
        else:
            raise ValueError("Invalid piece")

    @staticmethod
    def name(piece: int) -> str:
        if piece == Piece.B_PAWN:
            return "Black Pawn"
        elif piece == Piece.B_ROOK:
            return "Black Rook"
        elif piece == Piece.B_BISHOP:
            return "Black Bishop"
        elif piece == Piece.B_KNIGHT:
            return "Black Knight"
        elif piece == Piece.B_QUEEN:
            return "Black Queen"
        elif piece == Piece.B_KING:
            return "Black King"
        elif piece == Piece.W_PAWN:
            return "White Pawn"
        elif piece == Piece.W_ROOK:
            return "White Rook"
        elif piece == Piece.W_BISHOP:
            return "White Bishop"
        elif piece == Piece.W_KNIGHT:
            return "White Knight"
        elif piece == Piece.W_QUEEN:
            return "White Queen"
        elif piece == Piece.W_KING:
            return "White King"
        elif piece == Piece.EMPTY:
            return "Empty"
        else:
            raise ValueError("Invalid piece")

    @staticmethod
    def short_name(piece: int) -> str:
        """
        Get the short name for a piece.
        """
        if piece == Piece.B_PAWN:
            return "p"
        elif piece == Piece.B_ROOK:
            return "r"
        elif piece == Piece.B_BISHOP:
            return "b"
        elif piece == Piece.B_KNIGHT:
            return "n"
        elif piece == Piece.B_QUEEN:
            return "q"
        elif piece == Piece.B_KING:
            return "k"
        elif piece == Piece.W_PAWN:
            return "P"
        elif piece == Piece.W_ROOK:
            return "R"
        elif piece == Piece.W_BISHOP:
            return "B"
        elif piece == Piece.W_KNIGHT:
            return "N"
        elif piece == Piece.W_QUEEN:
            return "Q"
        elif piece == Piece.W_KING:
            return "K"
        elif piece == Piece.EMPTY:
            return " "
        else:
            raise ValueError("Invalid piece")


class Move:
    """
    Represents a single move.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int, piece: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.piece = piece

    def __repr__(self):
        return f"Move(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, piece={self.piece})"


class Board:
    """
    Represents a chess board.
    """

    def __init__(self, board: Optional[np.ndarray] = None):
        self.board = board or np.zeros((8, 8), dtype=np.int8)
        self.reset()

    def __repr__(self) -> str:
        return f"Board(board={self.board})"

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """
        Iterate over the board, yielding the x, y, and piece at each position.
        """
        for y in range(8):
            for x in range(8):
                yield x, y, self.get_piece(x, y)

    def reset(self):
        """
        Reset the board for the start of a game.
        """
        self.board[0] = [Piece.B_ROOK, Piece.B_KNIGHT, Piece.B_BISHOP, Piece.B_QUEEN, Piece.B_KING, Piece.B_BISHOP,
                         Piece.B_KNIGHT, Piece.B_ROOK]
        self.board[1] = [Piece.B_PAWN] * 8
        self.board[6] = [Piece.W_PAWN] * 8
        self.board[7] = [Piece.W_ROOK, Piece.W_KNIGHT, Piece.W_BISHOP, Piece.W_QUEEN, Piece.W_KING, Piece.W_BISHOP,
                         Piece.W_KNIGHT, Piece.W_ROOK]

    def get_piece(self, x: int, y: int) -> int:
        """
        Get the piece at the given position.
        """
        return self.board[y, x]

    def set_piece(self, x: int, y: int, piece: int):
        """
        Set the piece at the given position.
        """
        self.board[y, x] = piece

    def find(self, piece: int) -> List[Tuple[int, int]]:
        """
        Find all the positions of the given piece.
        """
        return list(zip(*np.where(self.board == piece)))

    def move(self, move: Move) -> int:
        """
        Make a move on the board. Returns the piece that was captured (or empty space if no piece was captured).
        Assumes validity of the move.
        """
        captured_piece = self.get_piece(move.x2, move.y2)
        self.set_piece(move.x2, move.y2, move.piece)
        self.set_piece(move.x1, move.y1, Piece.EMPTY)
        return captured_piece

    def notation_to_space(self, notation: str) -> Tuple[int, int]:
        """
        Convert a chess notation to a board space.
        """
        x = ord(notation[0]) - ord("a")
        y = int(notation[1]) - 1
        return x, y

    def space_to_notation(self, x: int, y: int) -> str:
        """
        Convert a board space to a chess notation.
        """
        return chr(x + ord("a")) + str(y + 1)

    def pprint(self) -> None:
        """
        Pretty print the board.
        """
        print(self.pformat())

    def pformat(self) -> str:
        """
        Pretty format the board.
        """
        output = ""
        for y in range(8):
            yname = str(y + 1)
            output += yname + " "
            for x in range(8):
                xname = chr(x + ord("a"))
                output += Piece.short_name(self.get_piece(x, y)) + " "
            output += "\n"
        output += "  "
        for x in range(8):
            xname = chr(x + ord("a"))
            output += xname + " "
        return output

    def copy(self) -> "Board":
        """
        Return a copy of the board.
        """
        return Board(self.board.copy())


class MoveValidator:
    """
    Validates moves. Does not check for checks.
    """

    def __init__(self, board: Board, history: List[Move] = []):
        self.board = board
        self.history = history

    def _is_move_in_bounds(self, move: Move) -> bool:
        """
        Check if the move is in bounds.
        """
        return 0 <= move.x1 < 8 and 0 <= move.y1 < 8 and 0 <= move.x2 < 8 and 0 <= move.y2 < 8

    def _is_path_clear(self, move: Move) -> bool:
        """
        Check if there are any obstructions in the move's path.
        Assumes straight line movement (vertical, diagonal, horizontal).
        Allows for a piece of the opposing side's color at the end state.
        """
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            return False
        if move.x1 == move.x2:
            # Vertical move
            start = min(move.y1, move.y2)
            end = max(move.y1, move.y2)
            for y in range(start + 1, end):
                if self.board.get_piece(move.x1, y) != Piece.EMPTY:
                    return False
            return True
        elif move.y1 == move.y2:
            # Horizontal move
            start = min(move.x1, move.x2)
            end = max(move.x1, move.x2)
            for x in range(start + 1, end):
                if self.board.get_piece(x, move.y1) != Piece.EMPTY:
                    return False
            return True
        elif abs(move.x1 - move.x2) == abs(move.y1 - move.y2):
            # Diagonal move
            x_start = min(move.x1, move.x2)
            x_end = max(move.x1, move.x2)
            y_start = min(move.y1, move.y2)
            y_end = max(move.y1, move.y2)
            x = x_start + 1
            y = y_start + 1
            while x < x_end and y < y_end:
                if self.board.get_piece(x, y) != Piece.EMPTY:
                    return False
                x += 1
                y += 1
            return True
        else:
            return False

    def _check_en_passant(self, move) -> bool:
        """
        Return true if the move is a valid en passant
        """
        if move.piece == Piece.W_PAWN and move.y1 == 4 and move.y2 == 5:
            if self.board.get_piece(move.x2, move.y2) == Piece.EMPTY:
                if self.board.get_piece(move.x2, move.y1) == Piece.B_PAWN:
                    if self.history[-1].piece == Piece.B_PAWN:
                        if abs(self.history[-1].y1 - self.history[-1].y2) == 2:
                            if self.history[-1].x1 == move.x2:
                                return True
        elif move.piece == Piece.B_PAWN and move.y1 == 3 and move.y2 == 2:
            if self.board.get_piece(move.x2, move.y2) == Piece.EMPTY:
                if self.board.get_piece(move.x2, move.y1) == Piece.W_PAWN:
                    if self.history[-1].piece == Piece.W_PAWN:
                        if abs(self.history[-1].y1 - self.history[-1].y2) == 2:
                            if self.history[-1].x1 == move.x2:
                                return True
        return False

    def _is_valid_pawn_move(self, move: Move) -> bool:
        """
        Check if a pawn move is valid.
        """
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            return False
        if abs(move.y2 - move.y1) == 1:
            if move.x2 == move.x1:
                # regular move
                return self.board.get_piece(move.x2, move.y2) == Piece.EMPTY
            elif abs(move.x2 - move.x1) == 1:
                if self._check_en_passant(move):
                    # en passant
                    return True
                else:
                    # normal diagonal attack
                    return self.board.get_piece(move.x2, move.y2) != Piece.EMPTY
        elif abs(move.y2 - move.y1) == 2 and move.x1 == move.x2:
            # double move
            path_clear = self._is_path_clear(move)

            if Piece.color(move.piece) == PieceColor.WHITE:
                return move.y1 == 6 and path_clear
            else:
                return move.y1 == 1 and path_clear
        return False

    def _is_valid_knight_move(self, move: Move) -> bool:
        """
        Check if a knight move is valid.
        """
        x_delta = abs(move.x2 - move.x1)
        y_delta = abs(move.y2 - move.y1)
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            return False
        return (x_delta == 1 and y_delta == 2) or (x_delta == 2 and y_delta == 1)

    def _is_valid_bishop_move(self, move: Move) -> bool:
        """
        Check if a bishop move is valid.
        """
        x_delta = abs(move.x2 - move.x1)
        y_delta = abs(move.y2 - move.y1)
        return x_delta == y_delta and x_delta != 0 and self._is_path_clear(move)

    def _is_valid_rook_move(self, move: Move) -> bool:
        """
        Check if a rook move is valid.
        """
        return (move.x1 == move.x2 or move.y1 == move.y2) and self._is_path_clear(move)

    def _is_valid_queen_move(self, move: Move) -> bool:
        """
        Check if a queen move is valid.
        """
        return self._is_valid_rook_move(move) or self._is_valid_bishop_move(move)

    def _is_valid_king_move(self, move: Move) -> bool:
        """
        Check if a king move is valid.
        """
        x_delta = abs(move.x2 - move.x1)
        y_delta = abs(move.y2 - move.y1)
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            return False
        return (x_delta <= 1 and y_delta <= 1)

    def is_valid_move(self, move: Move, turn: Optional[int]=None) -> bool:
        """
        Check if a move is valid.
        :param move: The move to check.
        :param turn: The turn to check the move for. If None, no turn check is performed.
        """
        piece = self.board.get_piece(move.x1, move.y1)

        # universal checks
        if piece == Piece.EMPTY:
            return False
        if turn is not None and Piece.color(piece) != turn:
            return False
        if not self._is_move_in_bounds(move):
            return False
        if piece != move.piece:
            return False

        # piece specific checks
        if piece in (Piece.W_PAWN, Piece.B_PAWN):
            return self._is_valid_pawn_move(move)
        elif piece in (Piece.W_KNIGHT, Piece.B_KNIGHT):
            return self._is_valid_knight_move(move)
        elif piece in (Piece.W_BISHOP, Piece.B_BISHOP):
            return self._is_valid_bishop_move(move)
        elif piece in (Piece.W_ROOK, Piece.B_ROOK):
            return self._is_valid_rook_move(move)
        elif piece in (Piece.W_QUEEN, Piece.B_QUEEN):
            return self._is_valid_queen_move(move)
        elif piece in (Piece.W_KING, Piece.B_KING):
            return self._is_valid_king_move(move)
        else:
            raise ValueError("Invalid piece: {}".format(piece))

    def get_valid_moves(self, x: int, y: int) -> List[Move]:
        """
        Get all valid moves for the piece in the specified location.
        """
        piece = self.board.get_piece(x, y)
        if piece == Piece.EMPTY:
            return []
        elif piece == Piece.B_PAWN:
            candidate_moves = [
                Move(x, y, x, y + 1, piece),
                Move(x, y, x, y + 2, piece),
                Move(x, y, x + 1, y + 1, piece),
                Move(x, y, x - 1, y + 1, piece),
            ]
        elif piece == Piece.W_PAWN:
            candidate_moves = [
                Move(x, y, x, y - 1, piece),
                Move(x, y, x, y - 2, piece),
                Move(x, y, x + 1, y - 1, piece),
                Move(x, y, x - 1, y - 1, piece),
            ]
        elif piece in (Piece.B_KNIGHT, Piece.W_KNIGHT):
            candidate_moves = [
                Move(x, y, x + 1, y + 2, piece),
                Move(x, y, x + 1, y - 2, piece),
                Move(x, y, x - 1, y + 2, piece),
                Move(x, y, x - 1, y - 2, piece),
                Move(x, y, x + 2, y + 1, piece),
                Move(x, y, x + 2, y - 1, piece),
                Move(x, y, x - 2, y + 1, piece),
                Move(x, y, x - 2, y - 1, piece),
            ]
        elif piece in (Piece.B_BISHOP, Piece.W_BISHOP):
            candidate_moves = []
            for i in range(1, 8):
                candidate_moves.append(Move(x, y, x + i, y + i, piece))
                candidate_moves.append(Move(x, y, x + i, y - i, piece))
                candidate_moves.append(Move(x, y, x - i, y + i, piece))
                candidate_moves.append(Move(x, y, x - i, y - i, piece))
        elif piece in (Piece.B_ROOK, Piece.W_ROOK):
            candidate_moves = []
            for i in range(1, 8):
                candidate_moves.append(Move(x, y, x + i, y, piece))
                candidate_moves.append(Move(x, y, x - i, y, piece))
                candidate_moves.append(Move(x, y, x, y + i, piece))
                candidate_moves.append(Move(x, y, x, y - i, piece))
        elif piece in (Piece.B_QUEEN, Piece.W_QUEEN):
            candidate_moves = []
            for i in range(1, 8):
                candidate_moves.append(Move(x, y, x + i, y, piece))
                candidate_moves.append(Move(x, y, x - i, y, piece))
                candidate_moves.append(Move(x, y, x, y + i, piece))
                candidate_moves.append(Move(x, y, x, y - i, piece))
                candidate_moves.append(Move(x, y, x + i, y + i, piece))
                candidate_moves.append(Move(x, y, x + i, y - i, piece))
                candidate_moves.append(Move(x, y, x - i, y + i, piece))
                candidate_moves.append(Move(x, y, x - i, y - i, piece))
        elif piece in (Piece.B_KING, Piece.W_KING):
            candidate_moves = [
                Move(x, y, x + 1, y, piece),
                Move(x, y, x - 1, y, piece),
                Move(x, y, x, y + 1, piece),
                Move(x, y, x, y - 1, piece),
                Move(x, y, x + 1, y + 1, piece),
                Move(x, y, x + 1, y - 1, piece),
                Move(x, y, x - 1, y + 1, piece),
                Move(x, y, x - 1, y - 1, piece),
            ]
        else:
            raise ValueError("Invalid piece: {}".format(piece))
        
        valid_moves = []
        for move in candidate_moves:
            if self.is_valid_move(move, Piece.color(piece)):
                valid_moves.append(move)
        return valid_moves


class Game:
    """
    Represents a game of chess.
    """

    def __init__(self, board: Optional[Board] = None, turn=PieceColor.WHITE):
        self.board = board or Board()
        self.turn = turn or PieceColor.WHITE
        self.history: List[Move] = []
        self.validator = MoveValidator(self.board, self.history)

    def __repr__(self):
        return f"Game(board={self.board}, turn={self.turn})"

    def check_for_check(self, color: int) -> bool:
        """
        Check if the specified color is in check.
        """
        king_piece = Piece.W_KING if color == PieceColor.WHITE else Piece.B_KING
        king_x, king_y = self.board.find(king_piece)[0]
        for x, y, piece in self.board:
            if Piece.color(piece) != color:
                if self.validator.is_valid_move(Move(x, y, king_x, king_y, piece)):
                    return True
        return False

    def validate_move(self, move: Move) -> bool:
        """
        Check if the specified move is valid.
        """
        valid = self.validator.is_valid_move(move, self.turn)
        if not valid:
            return False
        # Test if move would leave us in check.
        test_game = Game(self.board.copy(), self.turn)
        test_game.board.move(move)
        if test_game.check_for_check(self.turn):
            return False
        return True

    def move(self, move: Move) -> bool:
        """
        Make a move. Returns True if the move was successful, False otherwise.
        """
        if self.validate_move(move):
            self.board.move(move)
            self.history.append(move)
            self.turn = PieceColor.opposite(self.turn)
            return True
        return False
