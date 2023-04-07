from abc import ABC
from typing import List, Optional

from codenames.game import Board, CardColor, GivenHint, Player, TeamColor

from solvers.gpt.instructions import load_instructions

INSTRUCTIONS = load_instructions()
FULL_INSTRUCTIONS = INSTRUCTIONS["full_instructions"]
SHORT_INSTRUCTIONS = INSTRUCTIONS["short_instructions"]
TURN_COMMAND = INSTRUCTIONS["turn_command"]


class GPTPlayer(Player, ABC):
    def __init__(
        self, name: str, api_key: str, team_color: Optional[TeamColor] = None, model_name: str = "gpt-3.5-turbo"
    ):
        super().__init__(name=name, team_color=team_color)
        self.api_key = api_key
        self.model_name = model_name

    @classmethod
    def build_board_repr(cls, board: Board) -> str:
        words = [f"{card.word}-{card.color}" for card in board.cards]
        joined = ", ".join(words)
        return f"Board cards: {joined}."

    @classmethod
    def build_score_repr(cls, board: Board) -> str:
        total_red, total_blue_cards = len(board.red_cards), len(board.blue_cards)  # type: ignore
        unrevealed_red = len(board.unrevealed_cards_for_color(CardColor.RED))
        unrevealed_blue = len(board.unrevealed_cards_for_color(CardColor.BLUE))
        return (
            f"The current score status is: "
            f"Red: {total_red - unrevealed_red}/{total_red}, "
            f"Blue: {total_blue_cards - unrevealed_blue}/{total_blue_cards}."
        )

    @classmethod
    def build_disallowed_hints_repr(
        cls, board: Board, hints: List[GivenHint], extra: Optional[List[str]] = None
    ) -> str:
        extra = extra or []
        words = [card.word for card in board.cards] + [hint.word for hint in hints] + extra
        if not words:
            return ""
        return f"The following expressions are NOT legal hints: {', '.join(words)}."

    @classmethod
    def build_cards_to_avoid_repr(cls, board: Board, team_color: TeamColor) -> str:
        filter_card_color = team_color.as_card_color
        words = [card.word for card in board.unrevealed_cards if card.color != filter_card_color]
        return f"Avoid giving a hint that is related to any of these words: {', '.join(words)}."

    @classmethod
    def build_assassin_repr(cls, board: Board) -> str:
        words = [card.word for card in board.unrevealed_cards if card.color == CardColor.BLACK]
        return f"The assassin word is: {', '.join(words)}, avoid hints suggesting this word as much as possible."

    @classmethod
    def build_hinted_words(cls, board: Board, team_color: TeamColor) -> str:
        words = [card.word for card in board.unrevealed_cards_for_color(team_color.as_card_color)]
        return f"You are looking for a hint for this word group: {', '.join(words)}."
