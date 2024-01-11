import logging
from typing import Optional

from codenames.game.move import GuessMove, HintMove, Move, PassMove
from codenames.game.player import Player, PlayerRole
from codenames.game.runner import GameRunner
from codenames.game.state import GameState

log = logging.getLogger(__name__)


def print_results(game_runner: Optional[GameRunner]):
    if game_runner is None or game_runner.state is None:
        return
    state = game_runner.state
    _print_board(state)
    _print_moves(game_runner)
    _print_result(state)


def _print_board(state: GameState):
    log.info("")
    log.info(f"{state.board}")


def _print_moves(game_runner: GameRunner):
    log.info("")
    log.info("Game moves:")
    hint_count = 0
    state = game_runner.state
    for move in state.moves:
        player = _get_player(game_runner, move)
        if isinstance(move, HintMove):
            hint = state.raw_hints[hint_count]
            hint_count += 1
            log.info(f"{player} said '{hint.word}' for words: {hint.for_words}")
        elif isinstance(move, GuessMove):
            given_guess = move.given_guess
            log.info(f"   {player} said: {given_guess}")
        elif isinstance(move, PassMove):
            log.info(f"   {player} passed the turn.")


def _get_player(game_runner: GameRunner, move: Move) -> Player:
    role = PlayerRole.HINTER if isinstance(move, HintMove) else PlayerRole.GUESSER
    return game_runner.players.get_player(team_color=move.team_color, role=role)


def _print_result(state: GameState):
    red_score = len([card for card in state.board.red_cards if card.revealed])
    blue_score = len([card for card in state.board.blue_cards if card.revealed])
    log.info("")
    log.info(f"Score: 🟦 {blue_score} - {red_score} 🟥")
    log.info(f"Winner: {state.winner}")
