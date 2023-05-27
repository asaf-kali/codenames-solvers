from codenames.game.base import canonical_format
from codenames.game.exceptions import CardNotFoundError
from codenames.game.move import Guess, Hint
from codenames.game.player import Guesser, Hinter
from codenames.game.state import GuesserGameState, HinterGameState


class CliHinter(Hinter):
    @property
    def is_human(self) -> bool:
        return True

    def pick_hint(self, game_state: HinterGameState) -> Hint:
        # print(f"State is: {state}")
        while True:
            try:
                data = input("Please enter your given_hint in the format 'word, card_amount': ")
                print()
                word, card_amount = data.split(",")
                word = word.strip().title()
                card_amount_parsed = int(card_amount.strip())
                return Hint(word=word, card_amount=card_amount_parsed)
            except ValueError:
                pass


class CliGuesser(Guesser):
    @property
    def is_human(self) -> bool:
        return True

    def guess(self, game_state: GuesserGameState) -> Guess:
        print("\n", game_state.board.printable_string, "\n", sep="")
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