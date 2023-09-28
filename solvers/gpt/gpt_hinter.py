import logging
import random
from typing import List, Optional

from codenames.boards.english import ENGLISH_WORDS
from codenames.game.board import Board
from codenames.game.color import CardColor, TeamColor
from codenames.game.move import GivenHint, Hint
from codenames.game.player import Hinter
from codenames.game.state import HinterGameState

from solvers.gpt.gpt_player import (
    HINTER_TURN_COMMAND,
    GPTPlayer,
    extract_data_from_response,
)

log = logging.getLogger(__name__)


class GPTHinter(GPTPlayer, Hinter):
    def pick_hint(self, game_state: HinterGameState) -> Hint:
        board_repr = self.build_board_repr(board=game_state.board)
        team = self.build_team_repr()
        moves = self.build_moves_repr(state=game_state)
        score_status = self.build_score_repr(score=game_state.score)
        words_to_hint = self.build_hinted_words(board=game_state.board, team_color=self.team_color)
        words_to_avoid = self.build_cards_to_avoid_repr(board=game_state.board, team_color=self.team_color)
        disallowed_hints = self.build_disallowed_hints_repr(
            board=game_state.board, hints=game_state.given_hints, extra=[]
        )
        assassin = self.build_assassin_repr(board=game_state.board)
        # single_command_prompt = (
        #     f"{score_status} {team} {hinted_words} {avoid_words} {assassin} {disallowed_hints} {TURN_COMMAND}"
        # )
        # pylint: disable=R0801
        infos = [
            # SHORT_INSTRUCTIONS,
            board_repr,
            team,
            moves,
            score_status,
            words_to_hint,
            words_to_avoid,
            disallowed_hints,
            assassin,
            HINTER_TURN_COMMAND,
            words_to_hint,
            assassin,
        ]
        messages = [
            # {"role": "user", "content": single_command_prompt},
            {"role": "system", "content": info}
            for info in infos
            if info is not None
        ]
        try:
            result = self.generate_completion(messages=messages)
            hint = self.parse_hint(completion_result=result)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Error while generating hint", exc_info=e)
            return Hint(word=_random_english_word(), card_amount=1)
        if hint.word in game_state.illegal_words:
            log.warning(f"Generated a hint that is not allowed: {hint}")
            return Hint(word=_random_english_word(), card_amount=1)
        self._verify_hint(hint=hint, game_state=game_state)
        return hint

    def _verify_hint(self, hint: Hint, game_state: HinterGameState):
        for word in hint.for_words:
            try:
                card = game_state.board[word]
            except KeyError:
                log.warning(f"Hint {hint} is referring to a word that is not on the board: {word}")
                continue
            if card.revealed:
                log.warning(f"Hint {hint} is referring to a word that is already revealed: {word}")
                continue
            if card.color == CardColor.BLACK:
                log.warning(f"Hint {hint} is referring to the assassin word: {word}")
                continue

    @classmethod
    def build_board_repr(cls, board: Board) -> str:
        words = [f"'{card.word}' ({card.color})" for card in board.cards]
        joined = ", ".join(words)
        return f"Board cards: {joined}."

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

    @classmethod
    def parse_hint(cls, completion_result: dict) -> Hint:
        data = extract_data_from_response(completion_result=completion_result)
        extra = data.get("extra")
        word_raw: str = data["word"]
        word = cls._parse_word(word_raw).lower()
        referred_cards_raw: List[str] = data["referred_cards"] or []
        referred_cards = [word.lower() for word in referred_cards_raw]
        hint = Hint(word=word, card_amount=len(referred_cards), for_words=tuple(referred_cards))
        log.info(f"Parsed hint: {hint}. Extra: {extra}")
        return hint

    @classmethod
    def _parse_word(cls, word: str) -> str:
        parts = word.split()
        if len(parts) == 1:
            return word
        if len(parts) > 2:
            raise ValueError(f"Hint word must be a single word or a two-word phrase, got: {word}")
        second = parts[1]
        if not second.isnumeric():
            return word
        log.debug("Got a number as the second word of the hint, assuming it's a number of cards")
        return parts[0]


def _random_english_word() -> str:
    return random.choice(ENGLISH_WORDS)
