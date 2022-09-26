import logging
from optparse import Option
from typing import List, Dict, Set, Tuple, Optional, Iterator
from numpy.typing import ArrayLike
import numpy as np
from enum import IntEnum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GameResolution(IntEnum):
    """
    End state of a game
    """
    UNDECIDED = 0
    WHITE_WINS = 1
    BLACK_WINS = 2
    DRAW = 3


class PieceColor(IntEnum):
    WHITE = 0
    BLACK = 1

    @staticmethod
    def opposite(color):
        return PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE


class MoveType(IntEnum):
    NORMAL = 0
    CASTLE = 1
    EN_PASSANT = 2


class Piece(IntEnum):
    EMPTY = 0

    # Black
    B_PAWN = 1
    B_ROOK = 2
    B_KNIGHT = 3
    B_BISHOP = 4
    B_QUEEN = 5
    B_KING = 6

    # White
    W_PAWN = 7
    W_ROOK = 8
    W_KNIGHT = 9
    W_BISHOP = 10
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
    def get_name(piece: int) -> str:
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

    def __init__(
            self,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            piece: int,
            move_type: MoveType = MoveType.NORMAL,
            start_notation: Optional[str] = None,
            end_notation: Optional[str] = None
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.piece = piece
        self.move_type = move_type
        self.start_notation = start_notation or Board.space_to_notation(x1, y1)
        self.end_notation = end_notation or Board.space_to_notation(x2, y2)

    def __repr__(self):
        return f"Move(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, piece={self.piece}, move_type={self.move_type}, start_notation='{self.start_notation}', end_notation='{self.end_notation}')"


class Board:
    """
    Represents a chess board.
    """

    def __init__(self, board: Optional[ArrayLike] = None):
        if board is None:
            self.board = np.zeros((8, 8), dtype=np.int8)
            self.reset()
        else:
            self.board = np.array(board, dtype=np.int8)
        self.validate_board_state()

    def __repr__(self) -> str:
        board_list = self.board.tolist()
        board_str = "[\n"
        for idx, row in enumerate(board_list):
            board_str += "\t"
            board_str += f"{row},\n"
        board_str += "]"
        return f"Board(board={board_str})"

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """
        Iterate over the board, yielding the x, y, and piece at each position.
        """
        for y in range(8):
            for x in range(8):
                yield x, y, self.get_piece(x, y)

    def validate_board_state(self):
        """
        Check that board state is valid (i.e. exactly one king of each color).
        Raises a ValueError if the board state is invalid.
        """
        b_king_count = len(np.where(self.board == Piece.B_KING)[0])
        w_king_count = len(np.where(self.board == Piece.W_KING)[0])
        if b_king_count != 1:
            raise ValueError(f"Invalid board state: {b_king_count} black kings")
        if w_king_count != 1:
            raise ValueError(f"Invalid board state: {w_king_count} white kings")

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
        ys, xs = np.where(self.board == piece)
        return list(zip(xs, ys))

    def move(self, move: Move) -> int:
        """
        Make a move on the board. Returns the piece that was captured (or empty space if no piece was captured).
        Assumes validity of the move.
        """
        if move.move_type == MoveType.CASTLE:
            # this is a castle, swap positions
            target_piece = self.get_piece(move.x2, move.y2)
            self.set_piece(move.x2, move.y2, move.piece)
            self.set_piece(move.x1, move.y1, target_piece)
            self.validate_board_state()
            return Piece.EMPTY
        elif move.move_type == MoveType.EN_PASSANT:
            # this is an en passant, swap positions and remove captured piece
            self.set_piece(move.x2, move.y2, move.piece)
            self.set_piece(move.x1, move.y1, Piece.EMPTY)
            self.set_piece(move.x2, move.y1, Piece.EMPTY)
            captured_piece = self.get_piece(move.x2, move.y1)
            self.validate_board_state()
            return captured_piece
        else:
            # Check for pawn promotion
            if move.piece == Piece.W_PAWN and move.y2 == 0:
                # promote to queen
                move.piece = Piece.W_QUEEN
            elif move.piece == Piece.B_PAWN and move.y2 == 7:
                # promote to queen
                move.piece = Piece.B_QUEEN
            captured_piece = self.get_piece(move.x2, move.y2)
            self.set_piece(move.x2, move.y2, move.piece)
            self.set_piece(move.x1, move.y1, Piece.EMPTY)
            self.validate_board_state()
            return captured_piece

    @staticmethod
    def notation_to_space(notation: str) -> Tuple[int, int]:
        """
        Convert a chess notation to a board space.
        """
        x = ord(notation[0]) - ord("a")
        try:
            y = "87654321".index(notation[1])
        except ValueError:
            # Invalid notation, substitute with -1
            y = -1
        return x, y

    @staticmethod
    def space_to_notation(x: int, y: int) -> str:
        """
        Convert a board space to a chess notation.
        """
        try:
            y_part = "87654321"[y]
        except IndexError:
            # Invalid space, substitute with "x"
            y_part = "x"
        return chr(x + ord("a")) + y_part

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

    def __init__(self, game: "Game"):
        self.game = game
        self.board = game.board
        self.history = game.history

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
            logger.debug(f"Invalid move: move to same color piece: {move}")
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
            if (move.x1 > move.x2) == (move.y1 > move.y2):
                # upwards diagonal
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
                # downwards diagonal
                x_start = min(move.x1, move.x2)
                x_end = max(move.x1, move.x2)
                y_start = max(move.y1, move.y2)
                y_end = min(move.y1, move.y2)
                print(f"start ({x_start, y_start}) {self.board.space_to_notation(x_start, y_start)}")
                print(f"end ({x_end, y_end}) {self.board.space_to_notation(x_end, y_end)}")
                x = x_start + 1
                y = y_start -1
                while x < x_end and y > y_end:
                    print("actual:")
                    print("\t", (x, y))
                    print("\t", self.board.space_to_notation(x, y) + "; " + Piece.get_name(self.board.get_piece(x, y)))

                    if self.board.get_piece(x, y) != Piece.EMPTY:
                        return False
                    x += 1
                    y -= 1
                return True
        else:
            return False

    def _check_en_passant(self, move: Move) -> bool:
        """
        Return true if the move is a valid en passant
        """
        if move.piece == Piece.W_PAWN and move.y2 - move.y1 == -1 and abs(move.x2 - move.x1) == 1:
            if self.board.get_piece(move.x2, move.y2) == Piece.EMPTY:
                if self.board.get_piece(move.x2, move.y1) == Piece.B_PAWN:
                    if self.history[-1].piece == Piece.B_PAWN:
                        if abs(self.history[-1].y1 - self.history[-1].y2) == 2:
                            if self.history[-1].x1 == move.x2:
                                logger.info(f"Valid en passant: {move}")
                                move.move_type = MoveType.EN_PASSANT
                                return True
        if move.piece == Piece.B_PAWN and move.y2 - move.y1 == 1 and abs(move.x2 - move.x1) == 1:
            if self.board.get_piece(move.x2, move.y2) == Piece.EMPTY:
                if self.board.get_piece(move.x2, move.y1) == Piece.W_PAWN:
                    if self.history[-1].piece == Piece.W_PAWN:
                        if abs(self.history[-1].y1 - self.history[-1].y2) == 2:
                            if self.history[-1].x1 == move.x2:
                                logger.info(f"Valid en passant: {move}")
                                move.move_type = MoveType.EN_PASSANT
                                return True
        return False

    def _check_castle(self, move: Move) -> bool:
        """
        Check if the move is a valid castle.
        Requirements for a castle:
            - There cannot be any pieces between the king and the rook
            - Your king can not have moved
            - Your rook can not have moved
            - Your king can not be in check
            - Your king can not pass through check
        """
        if move.piece == Piece.W_KING or move.piece == Piece.B_KING:
            king_piece = move.piece
            king_location = (move.x1, move.y1)
            rook_piece = self.board.get_piece(move.x2, move.y2)
            rook_location = (move.x2, move.y2)
            if rook_piece not in (Piece.W_ROOK, Piece.B_ROOK) or Piece.color(rook_piece) != Piece.color(king_piece):
                return False
        elif move.piece == Piece.W_ROOK or move.piece == Piece.B_ROOK:
            rook_piece = move.piece
            rook_location = (move.x1, move.y1)
            king_piece = self.board.get_piece(move.x2, move.y2)
            king_location = (move.x2, move.y2)
            if king_piece not in (Piece.W_KING, Piece.B_KING) or Piece.color(rook_piece) != Piece.color(king_piece):
                return False
        else:
            return False

        # Check that rook and king are in the correct starting positions
        if king_piece == Piece.W_KING and king_location != (4, 7):
            return False
        elif king_piece == Piece.B_KING and king_location != (4, 0):
            return False
        elif rook_piece == Piece.W_ROOK and rook_location not in ((0, 7), (7, 7)):
            return False
        elif rook_piece == Piece.B_ROOK and rook_location not in ((0, 0), (7, 0)):
            return False
        
        # Check if king has moved
        king_moved = any([move.piece == king_piece for move in self.game.history])
        if king_moved:
            logger.debug(f"Invalid move: Cannot castle after King has moved: {move}")
            return False
        
        # Check if rook has moved
        rook_moved = any([
            move.piece == rook_piece and (move.x1, move.y1) == rook_location
            for move in self.game.history
        ])
        if rook_moved:
            logger.debug(f"Invalid move: Cannot castle after Rook has moved: {move}")
            return False
        
        # Check if king is in check
        if self.check_for_check(Piece.color(king_piece)):
            logger.debug(f"Invalid move: Cannot castle while King is in check: {move}")
            return False
        
        # Check if king passes through check
        if king_location[0] < rook_location[0]:
            # King side castle
            for x in range(king_location[0] + 1, rook_location[0]):
                test_move = Move(king_location[0], king_location[1], x, king_location[1], king_piece)
                if self.test_move_for_check(test_move, turn=Piece.color(king_piece)):
                    logger.debug(f"Invalid move: Cannot castle through check: {move}")
                    return False
        else:
            # Queen side castle
            for x in range(rook_location[0] + 1, king_location[0]):
                test_move = Move(king_location[0], king_location[1], x, king_location[1], king_piece)
                if self.test_move_for_check(test_move, turn=Piece.color(king_piece)):
                    logger.debug(f"Invalid move: Cannot castle through check: {move}")
                    return False
        
        # check if path is clear
        if king_location[0] < rook_location[0]:
            # King side castle
            for x in range(king_location[0] + 1, rook_location[0]):
                if self.board.get_piece(x, king_location[1]) != Piece.EMPTY:
                    logger.debug(f"Invalid move: Cannot castle through pieces: {move}")
                    return False
        else:
            # Queen side castle
            for x in range(rook_location[0] + 1, king_location[0]):
                if self.board.get_piece(x, king_location[1]) != Piece.EMPTY:
                    logger.debug(f"Invalid move: Cannot castle through pieces: {move}")
                    return False

        # Check if king ends up in check
        test_move = Move(king_location[0], king_location[1], rook_location[0], rook_location[1], king_piece)
        if self.test_move_for_check(test_move, turn=Piece.color(king_piece)):
            logger.debug(f"Invalid move: Cannot castle into check: {move}")
            return False

        move.move_type = MoveType.CASTLE
        return True

    def _is_valid_pawn_move(self, move: Move) -> bool:
        """
        Check if a pawn move is valid.
        """
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            logger.debug(f"Invalid move: move to same color piece: {move}")
            return False
        if abs(move.y2 - move.y1) == 1:
            if move.x2 == move.x1:
                # regular move
                valid = self.board.get_piece(move.x2, move.y2) == Piece.EMPTY
                if not valid:
                    logger.debug(f"Invalid move: vertical pawn move to occupied space: {move}")
                return valid
            elif abs(move.x2 - move.x1) == 1:
                if self._check_en_passant(move):
                    # en passant
                    return True
                else:
                    # normal diagonal attack
                    valid = self.board.get_piece(move.x2, move.y2) != Piece.EMPTY
                    if not valid:
                        logger.debug(f"Invalid move: diagonal pawn move to empty space: {move}")
                    return valid
        elif abs(move.y2 - move.y1) == 2 and move.x1 == move.x2:
            # double move
            path_clear = self._is_path_clear(move)

            if not path_clear:
                logger.debug(f"Invalid move: pawn double move blocked: {move}")
                return False

            if self.board.get_piece(move.x2, move.y2) != Piece.EMPTY:
                logger.debug(f"Invalid move: pawn double move to occupied space: {move}")
                return False

            if Piece.color(move.piece) == PieceColor.WHITE:
                return move.y1 == 6
            else:
                return move.y1 == 1
        logger.debug(f"Invalid move: other invalid pawn move: {move}")
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
            logger.debug(f"Invalid move: move to same color piece: {move}")
            return False
        valid = (x_delta == 1 and y_delta == 2) or (x_delta == 2 and y_delta == 1)
        if not valid:
            logger.debug(f"Invalid move: invalid knight move: {move}")
        return valid

    def _is_valid_bishop_move(self, move: Move) -> bool:
        """
        Check if a bishop move is valid.
        """
        x_delta = abs(move.x2 - move.x1)
        y_delta = abs(move.y2 - move.y1)
        path_clear = self._is_path_clear(move)
        if not path_clear:
            logger.debug(f"Invalid move: bishop move blocked: {move}")
            return False
        valid = x_delta == y_delta and x_delta != 0
        if not valid:
            logger.debug(f"Invalid move: invalid bishop move: {move}")
        return valid

    def _is_valid_rook_move(self, move: Move) -> bool:
        """
        Check if a rook move is valid.
        """
        path_clear = self._is_path_clear(move)
        if not path_clear:
            if self._check_castle(move):
                # Castle, allowed
                logger.info(f"Valid castle: {move}")
                return True
            else:
                logger.debug(f"Invalid move: rook move blocked: {move}")
                return False
        valid = (move.x1 == move.x2 or move.y1 == move.y2)

        if not valid:
            logger.debug(f"Invalid move: invalid rook move: {move}")
        return valid

    def _is_valid_queen_move(self, move: Move) -> bool:
        """
        Check if a queen move is valid.
        """
        path_clear = self._is_path_clear(move)
        if not path_clear:
            logger.debug(f"Invalid move: queen move blocked: {move}")
            return False
        valid = self._is_valid_rook_move(move) or self._is_valid_bishop_move(move)
        if not valid:
            logger.debug(f"Invalid move: invalid queen move: {move}")
        return valid

    def _is_valid_king_move(self, move: Move) -> bool:
        """
        Check if a king move is valid.
        """
        x_delta = abs(move.x2 - move.x1)
        y_delta = abs(move.y2 - move.y1)

        # check for castle
        if self._check_castle(move):
            # Castle, allowed
            logger.info(f"Valid castle: {move}")
            return True
        # check that we're not moving to the same space as a piece of the same color
        end_piece = self.board.get_piece(move.x2, move.y2)
        if end_piece != Piece.EMPTY and Piece.color(end_piece) == Piece.color(move.piece):
            logger.debug(f"Invalid move: move to same color piece: {move}")
            return False
        valid = (x_delta <= 1 and y_delta <= 1)
        if not valid:
            logger.debug(f"Invalid move: invalid king move: {move}")
        return valid

    def check_for_check(self, color: Optional[int]) -> bool:
        """
        Check if the specified color is in check.
        Return false if the color is None.
        """
        if color is None:
            return False
        king_piece = Piece.W_KING if color == PieceColor.WHITE else Piece.B_KING
        king_x, king_y = self.board.find(king_piece)[0]
        for x, y, piece in self.board:
            if Piece.color(piece) != color and piece != Piece.EMPTY:
                move = Move(x, y, king_x, king_y, piece)
                if self._is_valid_move(move):
                    logger.info(f"Found check for {color}: {move}")
                    return True
        return False

    def test_move_for_check(self, move: Move, turn: Optional[int]) -> bool:
        """
        Check if a move would put the moving player in check.
        Returns true if the move would put the player in check.
        :param move: the move to test
        :param turn: the color of the player making the move. If None, no check is performed.
        """
        # Test if move would leave us in check.
        if turn is None:
            return False
        test_game = Game(self.board.copy(), turn)
        try:
            test_game.board.move(move)
        except:
            raise
        return test_game.validator.check_for_check(Piece.color(move.piece))

    def _is_valid_move(self, move: Move, turn: Optional[int]=None) -> bool:
        """
        Check if a move is valid. Does not check for check.
        :param move: The move to check.
        :param turn: The turn to check the move for. If None, no turn check is performed.
        """
        piece = self.board.get_piece(move.x1, move.y1)

        # universal checks
        if piece == Piece.EMPTY:
            logging.info(f"Invalid move: no piece at start position (move={move})")
            return False
        if turn is not None and Piece.color(piece) != turn:
            logging.info(f"Invalid move: wrong turn (move={move}, turn={turn})")
            return False
        if not self._is_move_in_bounds(move):
            logging.info(f"Invalid move: move out of bounds (move={move})")
            return False
        if piece != move.piece:
            logging.info(f"Invalid move: piece mismatch (move={move})")
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

    def is_valid_move(self, move: Move, turn: Optional[int]=None) -> bool:
        """
        Check if a move is valid. Checks for check.
        :param move: The move to check.
        :param turn: The turn to check the move for. If None, no turn check is performed.
        """
        turn = Piece.color(move.piece) if turn is None else turn
        assert turn is not None
        valid = self._is_valid_move(move, turn)
        if not valid:
            return False
        # Test if move would leave us in check.
        move_would_check = self.test_move_for_check(move, turn)
        if move_would_check:
            logger.debug(f"Invalid move: Move would leave player in check: {move}")
            return False
        return True

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
            # add castling moves
            if piece == Piece.B_ROOK and y == 0 and (x == 0 or x == 7):
                candidate_moves.append(Move(x, y, 4, 0, piece, move_type=MoveType.CASTLE))
            elif piece == Piece.W_ROOK and y == 7 and (x == 0 or x == 7):
                candidate_moves.append(Move(x, y, 4, 7, piece, move_type=MoveType.CASTLE))
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
            # add castling moves
            if piece == Piece.B_KING and x == 4 and y == 0:
                candidate_moves.append(Move(x, y, 7, 0, piece, move_type=MoveType.CASTLE))
                candidate_moves.append(Move(x, y, 0, 0, piece, move_type=MoveType.CASTLE))
            elif piece == Piece.W_KING and x == 4 and y == 7:
                candidate_moves.append(Move(x, y, 7, 7, piece, move_type=MoveType.CASTLE))
                candidate_moves.append(Move(x, y, 0, 7, piece, move_type=MoveType.CASTLE))
        else:
            raise ValueError("Invalid piece: {}".format(piece))
        
        valid_moves = []
        for move in candidate_moves:
            if self.is_valid_move(move, Piece.color(piece)):
                valid_moves.append(move)
        return valid_moves

    def has_valid_moves(self, color: PieceColor) -> bool:
        """
        Check if the specified color has any valid moves.
        """
        for x in range(8):
            for y in range(8):
                if Piece.color(self.board.get_piece(x, y)) == color:
                    if self.get_valid_moves(x, y):
                        return True
        return False


class Game:
    """
    Represents a game of chess.
    """

    def __init__(self, board: Optional[Board] = None, turn=PieceColor.WHITE):
        self.board = board or Board()
        self.turn = turn or PieceColor.WHITE
        self.history: List[Move] = []
        self.validator = MoveValidator(self)

    def __repr__(self):
        return f"Game(board={self.board}, turn={self.turn})"

    def get_resolution(self) -> GameResolution:
        """
        Checks if game is over due to checkmate or stalemate.
        """
        logger.debug("Checking game resolution...")
        if self.validator.check_for_check(PieceColor.BLACK):
            if not self.validator.has_valid_moves(PieceColor.BLACK):
                logger.info("Black is in checkmate!")
                return GameResolution.WHITE_WINS
        elif self.validator.check_for_check(PieceColor.WHITE):
            if not self.validator.has_valid_moves(PieceColor.WHITE):
                logger.info("White is in checkmate!")
                return GameResolution.BLACK_WINS
        elif not self.validator.has_valid_moves(self.turn):
            logger.info("Stalemate!")
            return GameResolution.DRAW
        return GameResolution.UNDECIDED

    def move(self, move: Move) -> bool:
        """
        Make a move. Returns True if the move was successful, False otherwise.
        """
        if self.validator.is_valid_move(move, self.turn):
            self.board.move(move)
            self.history.append(move)
            self.turn = PieceColor.opposite(self.turn)
            return True
        return False
