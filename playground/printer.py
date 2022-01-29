import logging

from codenames.game import GameManager

log = logging.getLogger(__name__)


def print_results(game_manager: GameManager):
    if game_manager is None:
        return
    log.info(f"Winner: {game_manager.winner}")
    print("Hints:")
    for hint in game_manager.raw_hints:
        print(hint)
    print(game_manager.board)
