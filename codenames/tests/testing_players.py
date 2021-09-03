from typing import List, NamedTuple, Iterable, Tuple, Dict

from codenames.game.base import GameState, Hint, GivenHint, Guess, TeamColor
from codenames.game.manager import QuitGame, Team, PassGuessTurn
from codenames.game.player import Hinter, Guesser, Player

SKIP_GUESS = -1


class UnexpectedEndOfInput(Exception):
    def __init__(self, player: Player):
        self.player = player


class TestHinter(Hinter):
    def __init__(
        self, team_color: TeamColor, hints: Iterable[Hint], name: str = "Test Hinter", auto_quit: bool = False
    ):
        super().__init__(name=name, team_color=team_color)
        self.hints = iter(hints)
        self.auto_quit = auto_quit

    def pick_hint(self, state: GameState) -> Hint:
        try:
            hint = next(self.hints)
        except StopIteration:
            if self.auto_quit:
                raise QuitGame(self)
            raise UnexpectedEndOfInput(self)
        return hint


class TestGuesser(Guesser):
    def __init__(
        self, team_color: TeamColor, guesses: Iterable[Guess], name: str = "Test Guesser", auto_quit: bool = False
    ):
        super().__init__(name=name, team_color=team_color)
        self.guesses = iter(guesses)
        self.auto_quit = auto_quit

    def guess(self, state: GameState, given_hint: GivenHint, left_guesses: int) -> Guess:
        try:
            guess = next(self.guesses)
        except StopIteration:
            if self.auto_quit:
                raise QuitGame(self)
            raise UnexpectedEndOfInput(self)
        if guess.card_index == SKIP_GUESS:
            raise PassGuessTurn()
        return guess


class PredictedTurn(NamedTuple):
    hint: Hint
    guesses: List[int]


def build_team(team_color: TeamColor, turns: Iterable[PredictedTurn]) -> Team:
    hints = [turn.hint for turn in turns]
    guesses = [Guess(index) for turn in turns for index in turn.guesses]
    hinter = TestHinter(team_color=team_color, hints=hints)
    guesser = TestGuesser(team_color=team_color, guesses=guesses)
    return Team(hinter=hinter, guesser=guesser, team_color=team_color)


def build_teams(all_turns: Iterable[PredictedTurn]) -> Tuple[Team, Team]:
    team_to_turns: Dict[TeamColor, List[PredictedTurn]] = {TeamColor.BLUE: [], TeamColor.RED: []}
    current_team_color = TeamColor.BLUE
    for turn in all_turns:
        team_to_turns[current_team_color].append(turn)
        current_team_color = current_team_color.opponent
    blue_team = build_team(TeamColor.BLUE, turns=team_to_turns[TeamColor.BLUE])
    red_team = build_team(TeamColor.RED, turns=team_to_turns[TeamColor.RED])
    return blue_team, red_team
