from codenames.game.base import GuesserGameState, Guess, HinterGameState, Hint
from codenames.game.player import Guesser, Hinter


class CliHinter(Hinter):
    def pick_hint(self, state: HinterGameState) -> Hint:
        print(f"State is: {state}")
        while True:
            try:
                data = input("Please enter your hint in the format 'word, card_amount': ")
                word, card_amount = data.split(",")
                word = word.strip()
                card_amount_parsed = int(card_amount.strip())
                return Hint(word=word, card_amount=card_amount_parsed)
            except ValueError:
                pass


class CliGuesser(Guesser):
    def guess(self, state: GuesserGameState) -> Guess:
        print(f"State is: {state}")
        data = input("Please enter your guess card index: ")
        while True:
            try:
                index = int(data.strip())
                return Guess(card_index=index)
            except ValueError:
                pass
