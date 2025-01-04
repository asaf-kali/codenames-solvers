from codenames.generic.board import Board
from codenames.generic.card import canonical_format
from codenames.generic.exceptions import CardNotFoundError
from codenames.generic.move import Clue, Guess
from codenames.generic.player import Operative, Spymaster
from codenames.generic.state import OperativeState, SpymasterState


class CLISpymaster(Spymaster):
    @property
    def is_human(self) -> bool:
        return True

    def give_clue(self, game_state: SpymasterState) -> Clue:
        print_board(game_state.board)
        while True:
            try:
                data = input("Please enter your given_clue in the format 'word, card_amount': ")
                print()
                word, card_amount = data.split(",")
                word = word.strip().title()
                card_amount_parsed = int(card_amount.strip())
                return Clue(word=word, card_amount=card_amount_parsed)
            except ValueError:
                pass


class CLIOperative(Operative):
    @property
    def is_human(self) -> bool:
        return True

    def guess(self, game_state: OperativeState) -> Guess:
        print_board(game_state.board)
        index = None
        while index is None:
            data = input("Please enter your guess word or card index: ")
            data = canonical_format(data)
            print()
            try:
                index = int(data.strip())
            except ValueError:
                try:
                    index = game_state.board.find_card_index(data)
                except CardNotFoundError:
                    index = None  # type: ignore
        return Guess(card_index=index)


def print_board(board: Board):
    print("\n", board.printable_string, "\n", sep="")
