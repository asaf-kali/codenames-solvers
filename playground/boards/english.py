from codenames.boards import build_board
from codenames.boards.english import ENGLISH_WORDS

ENGLISH_BOARDS = [build_board(vocabulary=ENGLISH_WORDS, seed=i) for i in range(10)]
ENGLISH_BOARD_1 = ENGLISH_BOARDS[0]
ENGLISH_BOARD_2 = ENGLISH_BOARDS[1]
ENGLISH_BOARD_3 = ENGLISH_BOARDS[2]
ENGLISH_BOARD_4 = ENGLISH_BOARDS[3]
