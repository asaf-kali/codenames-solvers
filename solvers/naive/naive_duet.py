from codenames.duet.player import DuetPlayer, DuetTeam
from codenames.generic.board import Board
from codenames.generic.move import Clue, GivenClue, GivenGuess, Guess
from codenames.generic.player import Operative, Spymaster
from codenames.generic.state import OperativeState, SpymasterState


class UnifiedDuetPlayer(DuetPlayer):
    def __init__(self, name: str, spymaster: Spymaster, operative: Operative):
        super().__init__(name, team=DuetTeam.MAIN)
        self.spymaster = spymaster
        self.operative = operative

    def give_clue(self, game_state: SpymasterState) -> Clue:
        return self.spymaster.give_clue(game_state)

    def guess(self, game_state: OperativeState) -> Guess:
        return self.operative.guess(game_state)

    def on_game_start(self, board: Board):
        self.spymaster.on_game_start(board)
        self.operative.on_game_start(board)

    def on_clue_given(self, given_clue: GivenClue):
        self.spymaster.on_clue_given(given_clue)
        self.operative.on_clue_given(given_clue)

    def on_guess_given(self, given_guess: GivenGuess):
        self.spymaster.on_guess_given(given_guess)
        self.operative.on_guess_given(given_guess)
