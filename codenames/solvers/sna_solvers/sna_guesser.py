from codenames.game.base import GuesserGameState, Guess
from codenames.game.player import Guesser


class SnaGuesser(Guesser):
    def guess(self, game_state: GuesserGameState) -> Guess:
        return Guess(0)
