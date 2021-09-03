from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, Set, Optional

from codenames.game.player import Guesser, Hinter, Player
from codenames.game.state import TeamColor, Card, GivenHint, GivenGuess, GameState, CardColor, Guess

SKIP_TURN = -1


@dataclass
class Team:
    hinter: Hinter
    guesser: Guesser
    team_color: TeamColor


class GuessError(ValueError):
    pass


class GameManager:
    def __init__(
        self,
        red_hinter: Hinter,
        blue_hinter: Hinter,
        red_guesser: Guesser,
        blue_guesser: Guesser,
        language: str,
        cards: List[Card],
    ):
        self.red_hinter = red_hinter
        self.blue_hinter = blue_hinter
        self.red_guesser = red_guesser
        self.blue_guesser = blue_guesser
        self.language = language
        self.cards = cards
        self.given_hints: List[GivenHint] = []
        self.given_guesses: List[GivenGuess] = []
        self.winner: Optional[TeamColor] = None

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
    def red_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.RED}

    @cached_property
    def blue_cards(self) -> Set[Card]:
        return {card for card in self.cards if card.color == CardColor.BLUE}

    @property
    def state(self) -> GameState:
        return GameState(cards=self.cards, given_hints=self.given_hints, given_guesses=self.given_guesses)

    def _reset_state(self):
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

    def get_winner(self) -> Optional[TeamColor]:
        return self.winner

    def _guess_until_success(self, team: Team, given_hint: GivenHint, left_guesses: int) -> Optional[Card]:
        while True:
            try:
                guess = team.guesser.guess(state=self.state, given_hint=given_hint, left_guesses=left_guesses)
                if guess.card_index == SKIP_TURN:
                    return None
                return self._reveal_guessed_card(guess)
            except GuessError:
                pass

    def _check_winner(self) -> bool:
        score_target = {TeamColor.RED: len(self.red_cards), TeamColor.BLUE: len(self.blue_cards)}
        for guess in self.given_guesses:
            if guess.guessed_card.color == CardColor.GRAY:
                continue
            if guess.guessed_card.color == CardColor.BLACK:
                self.winner = TeamColor.opponent(guess.team)
                return True
            score_target[guess.team] -= 1
            if score_target[guess.team] == 0:
                self.winner = guess.team
                return True
        return False

    def _run_team_turn(self, team: Team) -> bool:
        """
        :param team: the team to play this turn.
        :return: True if the game has ended.
        """
        hint = team.hinter.pick_hint(self.state)
        given_hint = GivenHint(word=hint.word, card_amount=hint.card_amount, team=team.team_color)
        self.given_hints.append(given_hint)
        left_guesses = hint.card_amount
        while left_guesses >= 0:
            guessed_card = self._guess_until_success(team=team, given_hint=given_hint, left_guesses=left_guesses)
            given_guess = GivenGuess(given_hint=given_hint, guessed_card=guessed_card)
            self.given_guesses.append(given_guess)
            if self._check_winner():
                return True
            if not given_guess.was_correct:
                break
            left_guesses -= 1
        return False

    def _reveal_guessed_card(self, guess: Guess) -> Card:
        if guess.card_index < 0 or guess.card_index >= len(self.cards):
            raise GuessError("Given card index is of range!")
        guessed_card = self.cards[guess.card_index]
        if guessed_card.revealed:
            raise GuessError("Given card is already revealed!")
        guessed_card.revealed = True
        return guessed_card

    def run_game(self) -> TeamColor:
        self._reset_state()
        self._notify_game_starts()
        while True:
            if self._run_team_turn(team=self.red_team):
                break
            if self._run_team_turn(team=self.blue_team):
                break
        return self.get_winner()
