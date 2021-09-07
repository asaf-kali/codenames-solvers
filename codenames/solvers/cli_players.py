from codenames.game.base import GuesserGameState, Guess, HinterGameState, Hint
from codenames.game.player import Guesser, Hinter


class CliHinter(Hinter):
    def pick_hint(self, state: HinterGameState) -> Hint:
        # print(f"State is: {state}")
        while True:
            try:
                data = input("Please enter your hint in the format 'word, card_amount': ")
                print()
                word, card_amount = data.split(",")
                word = word.strip().title()
                card_amount_parsed = int(card_amount.strip())
                return Hint(word=word, card_amount=card_amount_parsed)
            except ValueError:
                pass


class CliGuesser(Guesser):
    def guess(self, state: GuesserGameState) -> Guess:
        # print(f"State is: {state}")
        while True:
            data = input("Please enter your guess word or card index: ").lower().strip()
            print()
            try:
                index = int(data.strip())
            except ValueError:
                if data not in state.board.all_words:
                    continue
                index = state.board.all_words.index(data)
            return Guess(card_index=index)
