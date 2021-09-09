from gensim.models import KeyedVectors

from codenames.game.base import GuesserGameState, Guess, TeamColor, Board
from codenames.game.manager import PASS_GUESS
from codenames.game.player import Guesser
from codenames.solvers.utils.model_loader import load_language


class NaiveGuesser(Guesser):
    def __init__(
        self,
        name: str,
        team_color: TeamColor,
    ):
        super().__init__(name=name, team_color=team_color)
        self.model: KeyedVectors = None  # type: ignore

    def notify_game_starts(self, language: str, board: Board):
        self.model = load_language(language=language)

    def guess(self, game_state: GuesserGameState) -> Guess:
        if game_state.bonus_given:
            return Guess(PASS_GUESS)
        self.model.most_similar()
        return Guess(0)
