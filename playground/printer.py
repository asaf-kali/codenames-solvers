import logging
from collections import defaultdict
from typing import List

from codenames.game import GameRunner, GivenHint, GivenGuess

log = logging.getLogger(__name__)


def print_results(game_runner: GameRunner):
    if game_runner is None:
        return
    log.info(f"\n{game_runner.state.board}")
    guesses_by_hints = defaultdict(list)
    for guess in game_runner.state.given_guesses:
        guesses_by_hints[guess.given_hint].append(guess)
    log.info("\nGame moves:")
    for hint, guesses in guesses_by_hints.items():
        hint: GivenHint
        guesses: List[GivenGuess]
        log.info(f"{hint.team_color} team turn, hinter said: {hint}")
        for guess in guesses:
            log.info(f"   Guesser said: {guess}")
    red_score = len([card for card in game_runner.state.board.red_cards if card.revealed])
    blue_score = len([card for card in game_runner.state.board.blue_cards if card.revealed])
    log.info(f"\nScore: Blue {blue_score} - {red_score} Red")
    log.info(f"\nWinner: {game_runner.state.winner}")
