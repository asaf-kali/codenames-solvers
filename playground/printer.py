import logging

from codenames.game.runner import GameRunner

log = logging.getLogger(__name__)


def print_results(game_manager: GameRunner):
    if game_manager is None:
        return
    log.info(f"\n{game_manager.state.board}")
    hint_strings = [str(hint) for hint in game_manager.state.raw_hints]
    hints_string = "\n".join(hint_strings)
    log.info(f"\nHints:\n{hints_string}")
    # log.info("Guesses:")
    # for guess in game_manager.given_guesses:
    #     log.info(guess)
    log.info(f"\nWinner: {game_manager.state.winner}")
