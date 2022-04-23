from codenames.boards import build_board
from codenames.boards.english import ENGLISH_WORDS

ENGLISH_BOARD_1 = build_board(vocabulary=ENGLISH_WORDS, seed=1)
ENGLISH_BOARD_2 = build_board(vocabulary=ENGLISH_WORDS, seed=2)
