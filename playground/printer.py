import logging

from codenames.game import GameManager

log = logging.getLogger(__name__)


def print_results(game_manager: GameManager):
    if game_manager is None:
        return
    log.info(f"\n{game_manager.board}")
    hint_strings = [str(hint) for hint in game_manager.raw_hints]
    hints_string = "\n".join(hint_strings)
    log.info(f"\nHints:\n{hints_string}")
    # log.info("Guesses:")
    # for guess in game_manager.given_guesses:
    #     log.info(guess)
    log.info(f"\nWinner: {game_manager.winner}")
