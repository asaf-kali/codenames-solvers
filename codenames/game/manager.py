import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Tuple, List, Optional

from codenames.game.base import TeamColor, Card, GivenHint, GivenGuess, GameState, CardColor, Guess
from codenames.game.player import Guesser, Hinter, Player
from codenames.utils import wrap

log = logging.getLogger(__name__)


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
    def __init__(self, player: Player):
        self.player = player


class PassGuessTurn(Exception):
    pass


class GuessError(ValueError):
    pass


# Manager


class GameManager:
    def __init__(self, blue_hinter: Hinter, red_hinter: Hinter, blue_guesser: Guesser, red_guesser: Guesser):
        self.blue_hinter = blue_hinter
        self.red_hinter = red_hinter
        self.blue_guesser = blue_guesser
        self.red_guesser = red_guesser
        self.language: str = ""
        self.cards: List[Card] = []
        self.given_hints: List[GivenHint] = []
        self.given_guesses: List[GivenGuess] = []
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
        return self.red_hinter, self.blue_hinter

    @cached_property
    def guessers(self) -> Tuple[Guesser, Guesser]:
        return self.red_guesser, self.blue_guesser

    @cached_property
    def players(self) -> Tuple[Player, ...]:
        return *self.hinters, *self.guessers

    @cached_property
    def red_team(self) -> Team:
        return Team(hinter=self.red_hinter, guesser=self.red_guesser, team_color=TeamColor.RED)

    @cached_property
    def blue_team(self) -> Team:
        return Team(hinter=self.blue_hinter, guesser=self.blue_guesser, team_color=TeamColor.BLUE)

    @cached_property
    def red_cards(self) -> Tuple[Card, ...]:
        return tuple(card for card in self.cards if card.color == CardColor.RED)

    @cached_property
    def blue_cards(self) -> Tuple[Card, ...]:
        return tuple(card for card in self.cards if card.color == CardColor.BLUE)

    @property
    def state(self) -> GameState:
        return GameState(cards=self.cards, given_hints=self.given_hints, given_guesses=self.given_guesses)

    def _reset_state(self, language: str, cards: List[Card]):
        log.info(f"Reset state with {wrap(len(cards))} cards, {wrap(language)} language")
        self.language = language
        self.cards = cards
        for card in self.cards:
            card.revealed = False
        self.given_hints = []
        self.given_guesses = []
        self.winner = None

    def _notify_game_starts(self):
        state = self.state
        guesser_censored = state.guesser_censored
        for hinter in self.hinters:
            hinter.notify_game_starts(language=self.language, state=state)
        for guesser in self.guessers:
            guesser.notify_game_starts(language=self.language, state=guesser_censored)

    def _guess_until_success(self, team: Team, given_hint: GivenHint, left_guesses: int) -> Card:
        while True:
            try:
                guess = team.guesser.guess(state=self.state, given_hint=given_hint, left_guesses=left_guesses)
                return self._reveal_guessed_card(guess)
            except GuessError:
                pass

    def _check_winner(self) -> bool:
        score_target = {TeamColor.RED: len(self.red_cards), TeamColor.BLUE: len(self.blue_cards)}
        for guess in self.given_guesses:
            card_color = guess.guessed_card.color
            if card_color == CardColor.GRAY:
                continue
            if card_color == CardColor.BLACK:
                winner_color = guess.team.opponent
                self.winner = Winner(team_color=winner_color, reason=WinningReason.OPPONENT_HITS_BLACK)
                return True
            team_color = card_color.as_team_color
            score_target[team_color] -= 1
            if score_target[team_color] == 0:
                self.winner = Winner(team_color=team_color, reason=WinningReason.TARGET_SCORE)
                return True
        return False

    def _run_team_turn(self, team: Team) -> bool:
        """
        :param team: the team to play this turn.
        :return: True if the game has ended.
        """
        log.info(f"\n-----\n{wrap(team.team_color.value)} turn")
        hint = team.hinter.pick_hint(self.state)
        given_hint = GivenHint(word=hint.word, card_amount=hint.card_amount, team_color=team.team_color)
        log.info(f"Hinter: '{hint.word}', {hint.card_amount} card(s)")
        self.given_hints.append(given_hint)
        left_guesses = hint.card_amount
        bonus_given = False
        while left_guesses > 0:
            try:
                guessed_card = self._guess_until_success(team=team, given_hint=given_hint, left_guesses=left_guesses)
            except PassGuessTurn:
                log.info("Guesser passed the turn")
                break
            given_guess = GivenGuess(given_hint=given_hint, guessed_card=guessed_card)
            log.info(f"Guesser: {given_guess}")
            self.given_guesses.append(given_guess)
            if self._check_winner():
                return True
            if not given_guess.was_correct:
                log.info("Guesser wrong, turn is over")
                break
            left_guesses -= 1
            if left_guesses == 0 and not bonus_given:
                log.info("Giving extra round!")
                bonus_given = True
                left_guesses += 1
        return False

    def _reveal_guessed_card(self, guess: Guess) -> Card:
        if guess.card_index < 0 or guess.card_index >= len(self.cards):
            raise GuessError("Given card index is of range!")
        guessed_card = self.cards[guess.card_index]
        if guessed_card.revealed:
            raise GuessError("Given card is already revealed!")
        guessed_card.revealed = True
        return guessed_card

    def _run_rounds(self):
        while True:
            if self._run_team_turn(team=self.blue_team):
                break
            if self._run_team_turn(team=self.red_team):
                break

    def run_game(self, language: str, cards: List[Card]) -> TeamColor:
        self._reset_state(language=language, cards=cards)
        self._notify_game_starts()
        try:
            self._run_rounds()
        except QuitGame as e:
            log.info(f"Player {e.player} quit the game!")
            winner_color = e.player.team_color.opponent
            self.winner = Winner(team_color=winner_color, reason=WinningReason.OPPONENT_QUIT)
        log.info(f"\n-----\n{self.winner.reason.value}")  # type: ignore
        log.info(f"{wrap(self.winner.team_color.value)} team wins!")  # type: ignore
        return self.winner  # type: ignore
