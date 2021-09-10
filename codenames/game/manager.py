import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Tuple, List, Optional

from codenames.game.base import (
    TeamColor,
    Card,
    GivenHint,
    GivenGuess,
    HinterGameState,
    CardColor,
    Guess,
    Board,
    Hint,
    GuesserGameState,
)
from codenames.game.player import Guesser, Hinter, Player
from codenames.solvers.utils.models import WordGroup
from codenames.utils import wrap

log = logging.getLogger(__name__)
SEPARATOR = "\n-----\n"
PASS_GUESS = -1
QUIT_GAME = -2


# Models


@dataclass
class Team:
    hinter: Hinter
    guesser: Guesser
    team_color: TeamColor


class WinningReason(Enum):
    TARGET_SCORE = "Target score reached"
    OPPONENT_HITS_BLACK = "Opponent hit black card"
    OPPONENT_QUIT = "Opponent quit"


@dataclass
class Winner:
    team_color: TeamColor
    reason: WinningReason


# Exceptions


class QuitGame(Exception):
    pass


class GameRuleError(Exception):
    pass


class InvalidHint(GameRuleError):
    pass


class InvalidGuess(GameRuleError):
    pass


# Manager


def _determine_first_team(board: Board) -> TeamColor:
    if len(board.blue_cards) >= len(board.red_cards):
        return TeamColor.BLUE
    return TeamColor.RED


class GameManager:
    def __init__(
        self,
        blue_hinter: Hinter,
        red_hinter: Hinter,
        blue_guesser: Guesser,
        red_guesser: Guesser,
    ):
        self.blue_hinter = blue_hinter
        self.red_hinter = red_hinter
        self.blue_guesser = blue_guesser
        self.red_guesser = red_guesser
        self.language = ""
        self.board = Board([])
        self.given_hints: List[GivenHint] = []
        self.given_guesses: List[GivenGuess] = []
        self.current_team_color: TeamColor = TeamColor.BLUE
        self.bonus_given = False
        self.left_guesses = 0
        self.winner: Optional[Winner] = None

    @staticmethod
    def from_teams(blue_team: Team, red_team: Team):
        return GameManager(
            blue_hinter=blue_team.hinter,
            red_hinter=red_team.hinter,
            blue_guesser=blue_team.guesser,
            red_guesser=red_team.guesser,
        )

    @cached_property
    def hinters(self) -> Tuple[Hinter, Hinter]:
        return self.blue_hinter, self.red_hinter

    @cached_property
    def guessers(self) -> Tuple[Guesser, Guesser]:
        return (
            self.blue_guesser,
            self.red_guesser,
        )

    @cached_property
    def players(self) -> Tuple[Player, ...]:
        return *self.hinters, *self.guessers

    @cached_property
    def blue_team(self) -> Team:
        return Team(hinter=self.blue_hinter, guesser=self.blue_guesser, team_color=TeamColor.BLUE)

    @cached_property
    def red_team(self) -> Team:
        return Team(hinter=self.red_hinter, guesser=self.red_guesser, team_color=TeamColor.RED)

    @property
    def hinter_state(self) -> HinterGameState:
        return HinterGameState(board=self.board, given_hints=self.given_hints, given_guesses=self.given_guesses)

    @property
    def guesser_state(self) -> GuesserGameState:
        board = self.board.censured
        return GuesserGameState(
            board=board,
            given_hints=self.given_hints,
            given_guesses=self.given_guesses,
            left_guesses=self.left_guesses,
            bonus_given=self.bonus_given,
        )

    @property
    def last_given_hint(self) -> GivenHint:
        return self.given_hints[-1]

    @property
    def is_game_over(self) -> bool:
        return self.winner is not None

    @property
    def given_hint_words(self) -> WordGroup:
        return tuple(hint.word for hint in self.given_hints)

    def _reset_state(self, language: str, board: Board):
        log.info(f"\n{SEPARATOR}Reset state with {wrap(len(board))} cards, {wrap(language)} language")
        self.language = language
        self.board = board
        for card in self.board:
            card.revealed = False
        self.given_hints = []
        self.given_guesses = []
        self.current_team_color = _determine_first_team(self.board)
        self.bonus_given = False
        self.left_guesses = 0
        self.winner = None

    def _notify_game_starts(self):
        censored_board = self.board.censured
        for hinter in self.hinters:
            hinter.notify_game_starts(language=self.language, board=self.board)
        for guesser in self.guessers:
            guesser.notify_game_starts(language=self.language, board=censored_board)

    def _run_team_turn(self, team: Team) -> bool:
        """
        :param team: the team to play this turn.
        :return: True if the game has ended.
        """
        self.get_hint_from(hinter=team.hinter)
        while self.left_guesses > 0:
            self.get_guess_from(guesser=team.guesser)
        return self.is_game_over

    def _reveal_guessed_card(self, guess: Guess) -> Card:
        if guess.card_index < 0 or guess.card_index >= len(self.board):
            raise InvalidGuess("Given card index is out of range!")
        guessed_card = self.board[guess.card_index]
        if guessed_card.revealed:
            raise InvalidGuess("Given card is already revealed!")
        guessed_card.revealed = True
        return guessed_card

    def _end_turn(self):
        self.left_guesses = 0
        self.bonus_given = False
        self.current_team_color = self.current_team_color.opponent

    def _run_rounds(self):
        while True:
            if self.current_team_color == TeamColor.BLUE:
                if self._run_team_turn(team=self.blue_team):
                    break
            else:
                if self._run_team_turn(team=self.red_team):
                    break

    def _check_winner(self) -> bool:
        score_target = {TeamColor.BLUE: len(self.board.blue_cards), TeamColor.RED: len(self.board.red_cards)}
        for guess in self.given_guesses:
            card_color = guess.guessed_card.color
            if card_color == CardColor.GRAY:
                continue
            if card_color == CardColor.BLACK:
                winner_color = guess.team.opponent
                self.winner = Winner(team_color=winner_color, reason=WinningReason.OPPONENT_HITS_BLACK)
                return True
            team_color = card_color.as_team_color  # type: ignore
            score_target[team_color] -= 1
            if score_target[team_color] == 0:
                self.winner = Winner(team_color=team_color, reason=WinningReason.TARGET_SCORE)
                return True
        return False

    def initialize_game(self, language: str, board: Board):
        self._reset_state(language=language, board=board)
        self._notify_game_starts()

    def _process_hint(self, hint: Hint) -> GivenHint:
        if hint.word in self.given_hint_words:
            raise InvalidHint("Hint word was already used!")
        given_hint = GivenHint(word=hint.word, card_amount=hint.card_amount, team_color=self.current_team_color)
        log.info(f"Hinter: '{hint.word}', {hint.card_amount} card(s)")
        self.given_hints.append(given_hint)
        self.left_guesses = given_hint.card_amount
        return given_hint

    def get_hint_from(self, hinter: Hinter) -> Hint:
        log.info(f"{SEPARATOR}{wrap(self.current_team_color.value)} turn.")
        hint = hinter.pick_hint(game_state=self.hinter_state)
        self._process_hint(hint)
        return hint

    def _process_guess(self, guess: Guess):
        if guess.card_index == PASS_GUESS:
            log.info("Guesser passed the turn")
            return self._end_turn()
        if guess.card_index == QUIT_GAME:
            log.info("Guesser quit the game")
            raise QuitGame()
        guessed_card = self._reveal_guessed_card(guess)
        given_guess = GivenGuess(given_hint=self.last_given_hint, guessed_card=guessed_card)
        log.info(f"Guesser: {given_guess}")
        self.given_guesses.append(given_guess)
        if self._check_winner():
            log.info("Winner is found, turn is over")
            return self._end_turn()
        if not given_guess.was_correct:
            log.info("Guesser wrong, turn is over")
            return self._end_turn()
        self.left_guesses -= 1
        if self.left_guesses == 0 and not self.bonus_given:
            log.info("Giving bonus guess!")
            self.bonus_given = True
            self.left_guesses += 1

    def get_guess_from(self, guesser: Guesser) -> Guess:
        while True:
            guess = guesser.guess(game_state=self.guesser_state)
            try:
                self._process_guess(guess)
            except InvalidGuess:
                continue
            except QuitGame:
                winner_color = guesser.team_color.opponent
                self.winner = Winner(team_color=winner_color, reason=WinningReason.OPPONENT_QUIT)
                self._end_turn()
            return guess

    def run_game(self, language: str, board: Board) -> TeamColor:
        self.initialize_game(language=language, board=board)
        self._run_rounds()
        log.info(
            f"{SEPARATOR}{self.winner.reason.value}, {wrap(self.winner.team_color.value)} team wins!"  # type: ignore
        )
        return self.winner  # type: ignore
