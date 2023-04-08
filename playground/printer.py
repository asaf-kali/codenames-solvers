import logging

from codenames.game.move import GuessMove, HintMove, PassMove
from codenames.game.runner import GameRunner
from codenames.game.state import GameState

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
    log.info("")
    log.info("Game moves:")
    hint_count = 0
    for move in state.moves:
        if isinstance(move, HintMove):
            given_hint = move.given_hint
            hint = state.raw_hints[hint_count]
            hint_count += 1
            log.info(f"{given_hint.team_color} team turn, hinter said '{hint.word}' for words: {hint.for_words}")
        elif isinstance(move, GuessMove):
            given_guess = move.given_guess
            log.info(f"   Guesser said: {given_guess}")
        elif isinstance(move, PassMove):
            log.info("   Guesser passed the turn.")


def _print_result(state: GameState):
    red_score = len([card for card in state.board.red_cards if card.revealed])
    blue_score = len([card for card in state.board.blue_cards if card.revealed])
    log.info("")
    log.info(f"Score: 🟦 {blue_score} - {red_score} 🟥")
    log.info(f"Winner: {state.winner}")
