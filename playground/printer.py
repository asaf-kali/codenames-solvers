import logging
from collections import defaultdict
from typing import Dict, List

from codenames.game import GameRunner, GameState, GivenGuess, GivenHint

log = logging.getLogger(__name__)


def print_results(game_runner: GameRunner):
    if game_runner is None:
        return
    state = game_runner.state
    _print_board(state)
    _print_moves(state)
    _print_result(state)


def _print_board(state: GameState):
    log.info("")
    log.info(f"{state.board}")


def _print_moves(state: GameState):
    guesses_by_hints = _get_guesses_by_hints(state)
    log.info("")
    log.info("Game moves:")
    for i, given_hint in enumerate(state.given_hints):
        given_guesses = guesses_by_hints[given_hint]
        hint = state.raw_hints[i]
        log.info(f"{given_hint.team_color} team turn, hinter said '{given_hint.word}' for words: {hint.for_words}")
        for guess in given_guesses:
            log.info(f"   Guesser said: {guess}")


def _get_guesses_by_hints(state: GameState):
    guesses_by_hints: Dict[GivenHint, List[GivenGuess]] = defaultdict(list)
    for guess in state.given_guesses:
        guesses_by_hints[guess.given_hint].append(guess)
    return guesses_by_hints


def _print_result(state: GameState):
    red_score = len([card for card in state.board.red_cards if card.revealed])
    blue_score = len([card for card in state.board.blue_cards if card.revealed])
    log.info("")
    log.info(f"Score: 🟦 {blue_score} - {red_score} 🟥")
    log.info(f"Winner: {state.winner}")
