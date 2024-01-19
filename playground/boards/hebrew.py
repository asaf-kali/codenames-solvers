from codenames.boards.hebrew import HEBREW_WORDS
from codenames.game.board import Board

HEBREW_BOARDS = [Board.from_vocabulary(language="hebrew", vocabulary=HEBREW_WORDS, seed=i) for i in range(10)]
HEBREW_BOARD_1 = HEBREW_BOARDS[0]
HEBREW_BOARD_2 = HEBREW_BOARDS[1]
HEBREW_BOARD_3 = HEBREW_BOARDS[2]
HEBREW_BOARD_4 = HEBREW_BOARDS[3]
