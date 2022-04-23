import pytest

from codenames.game import (
    Board,
    Guess,
    Hint,
    InvalidGuess,
    InvalidMove,
    PlayerRole,
    TeamColor,
    Winner,
    WinningReason,
    build_game_state,
)
from tests import constants


@pytest.fixture()
def board_10() -> Board:
    return constants.board_10()


def test_game_state_flow(board_10: Board):
    game_state = build_game_state(language="en", board=board_10)
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.HINTER

    # Round 1 - blue team
    game_state.process_hint(Hint(word="A", card_amount=2))
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 2

    game_state.process_guess(Guess(card_index=0))  # Blue - Correct
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 1
    assert game_state.bonus_given is False

    game_state.process_guess(Guess(card_index=1))  # Blue - Correct
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 1
    assert game_state.bonus_given is True

    with pytest.raises(InvalidGuess):
        game_state.process_guess(Guess(card_index=1))

    game_state.process_guess(Guess(card_index=7))  # Gray - Wrong
    assert game_state.current_team_color == TeamColor.RED
    assert game_state.current_player_role == PlayerRole.HINTER
    assert game_state.left_guesses == 0

    # Round 2 - red team
    game_state.process_hint(Hint(word="B", card_amount=1))
    assert game_state.current_team_color == TeamColor.RED
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 1

    game_state.process_guess(Guess(card_index=4))  # Red - Correct
    assert game_state.current_team_color == TeamColor.RED
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 1
    assert game_state.bonus_given is True

    game_state.process_guess(Guess(card_index=5))  # Red - Correct
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.HINTER

    # Round 3 - blue team
    game_state.process_hint(Hint(word="C", card_amount=2))
    assert game_state.current_team_color == TeamColor.BLUE
    assert game_state.current_player_role == PlayerRole.GUESSER
    assert game_state.left_guesses == 2

    game_state.process_guess(Guess(card_index=9))  # Black - Game over
    assert game_state.winner == Winner(team_color=TeamColor.RED, reason=WinningReason.OPPONENT_HIT_BLACK)

    with pytest.raises(InvalidMove):
        game_state.process_guess(Guess(card_index=9))
