from typing import List, Dict, Tuple, Optional
import pygame
import pathlib
from chess_ai import base_path
from chess_ai.game import Piece, PieceColor, Game, Board, Move

pygame.init()
pygame.font.init()

my_font = pygame.font.SysFont('Courier New', 32, bold=True)

character_images = {
    char: my_font.render(char, False, (0, 0, 0))
    for char in "abcdefgh12345678"
}

font_padding = 64
screen = pygame.display.set_mode([512 + font_padding, 512 + font_padding])

# load images
images = {}
image_paths = {
    Piece.B_BISHOP: "img/black_bishop.png",
    Piece.B_KING: "img/black_king.png",
    Piece.B_KNIGHT: "img/black_knight.png",
    Piece.B_PAWN: "img/black_pawn.png",
    Piece.B_QUEEN: "img/black_queen.png",
    Piece.B_ROOK: "img/black_rook.png",
    Piece.W_BISHOP: "img/white_bishop.png",
    Piece.W_KING: "img/white_king.png",
    Piece.W_KNIGHT: "img/white_knight.png",
    Piece.W_PAWN: "img/white_pawn.png",
    Piece.W_QUEEN: "img/white_queen.png",
    Piece.W_ROOK: "img/white_rook.png",
}
for piece, path in image_paths.items():
    images[piece] = pygame.image.load(str(base_path / path))

def update_display(
        board: Board,
        selected_square: Optional[Tuple[int, int]]=None,
        legal_moves: Optional[List[Move]]=None
) -> None:
    # blank the screen
    screen.fill((255, 255, 255))
    # draw the board
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (128, 128, 128)
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(font_padding / 2 + 64 * col, font_padding / 2 + 64 * row, 64, 64)
            )
    # then draw legend
    row_names = "87654321"
    for row in range(8):
        # left
        screen.blit(
            character_images[row_names[row]],
            (0, font_padding / 2 + 64 * row)
        )
        # right
        screen.blit(
            character_images[row_names[row]],
            (8 * 64 + font_padding / 2, font_padding / 2 + 64 * row)
        )
    col_names = "abcdefgh"
    for col in range(8):
        # top
        screen.blit(
            character_images[col_names[col]],
            (font_padding / 2 + 64 * col, 0)
        )
        # bottom
        screen.blit(
            character_images[col_names[col]],
            (font_padding / 2 + 64 * col, 8 * 64 + font_padding / 2)
        )
    # then draw highlights
    if selected_square is not None:
        x, y = selected_square
        pygame.draw.rect(screen, (0, 255, 0), (int(font_padding / 2 + x * 64), int(font_padding / 2 + y * 64), 64, 64))
    if legal_moves is not None:
        for move in legal_moves:
            x, y = move.x2, move.y2
            pygame.draw.rect(screen, (255, 0, 0), (int(font_padding / 2 + x * 64), int(font_padding / 2 + y * 64), 64, 64))
    for row in range(8):
        for col in range(8):
            piece = board.get_piece(col, row)
            if piece != Piece.EMPTY:
                # images are 60x60, so we need to center them
                screen.blit(images[piece], (font_padding / 2 + col * 64 + 2, font_padding / 2 + row * 64 + 2))
    pygame.display.flip()


game = Game()

selected_square: Optional[Tuple[int, int]] = None
legal_moves: Optional[List[Move]] = None

running = True
changed = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:
            xpos, ypos = pygame.mouse.get_pos()
            xpos = int(xpos - font_padding / 2)
            ypos = int(ypos - font_padding / 2)
            x = xpos // 64
            y = ypos // 64
            if selected_square is None:
                if Piece.color(game.board.get_piece(x, y)) == game.turn:
                    selected_square = (x, y)
                    legal_moves = game.validator.get_valid_moves(*selected_square)
            else:
                if (x, y) == selected_square:
                    # set down
                    selected_square = None
                    legal_moves = None
                else:
                    move = Move(
                        x1=selected_square[0],
                        y1=selected_square[1],
                        x2=x,
                        y2=y,
                        piece=game.board.get_piece(selected_square[0], selected_square[1])
                    )
                    if game.move(move):
                        # move was legal
                        changed = True
                        selected_square = None
                        legal_moves = None
                    else:
                        if Piece.color(game.board.get_piece(x, y)) == game.turn:
                            selected_square = (x, y)
                            legal_moves = game.validator.get_valid_moves(*selected_square)

    update_display(game.board, selected_square, legal_moves)
    
    # print game state to console
    if changed:
        print(game)
        changed = False

# Done! Time to quit.
pygame.quit()
