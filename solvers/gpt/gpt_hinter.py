import json
import logging
from typing import List, Optional

from codenames.game import (
    Board,
    CardColor,
    GivenHint,
    Hint,
    Hinter,
    HinterGameState,
    TeamColor,
)
from openai import ChatCompletion

from solvers.gpt.gpt_player import HINTER_TURN_COMMAND, GPTPlayer, find_json_in_string

log = logging.getLogger(__name__)


class GPTHinter(GPTPlayer, Hinter):
    def pick_hint(self, game_state: HinterGameState) -> Hint:
        board_repr = self.build_board_repr(board=game_state.board)
        score_status = self.build_score_repr(score=game_state.score)
        team = self.build_team_repr()
        hinted_words = self.build_hinted_words(board=game_state.board, team_color=self.team_color)
        avoid_words = self.build_cards_to_avoid_repr(board=game_state.board, team_color=self.team_color)
        assassin = self.build_assassin_repr(board=game_state.board)
        disallowed_hints = self.build_disallowed_hints_repr(
            board=game_state.board, hints=game_state.given_hints, extra=[]
        )
        # single_command_prompt = (
        #     f"{score_status} {team} {hinted_words} {avoid_words} {assassin} {disallowed_hints} {TURN_COMMAND}"
        # )
        infos = [
            # SHORT_INSTRUCTIONS,
            board_repr,
            score_status,
            team,
            hinted_words,
            avoid_words,
            assassin,
            disallowed_hints,
        ]
        messages = [
            # {"role": "user", "content": single_command_prompt},
            {"role": "system", "content": info}
            for info in infos
        ]
        messages += [{"role": "user", "content": HINTER_TURN_COMMAND}]
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        result = ChatCompletion.create(model=self.model_name, messages=messages, api_key=self.api_key, temperature=0)
        log.debug("Got completion result", extra={"result": result})
        return self.parse_hint(completion_result=result)

    @classmethod
    def build_board_repr(cls, board: Board) -> str:
        words = [f"{card.word}-{card.color}" for card in board.cards]
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
        response_content = completion_result["choices"][0]["message"]["content"]
        data_raw = find_json_in_string(response_content)
        log.debug(f"Parsing hint from: '{data_raw}'")
        data = json.loads(data_raw)
        extra = data.get("extra")
        word_raw: str = data["word"]
        word = cls._parse_word(word_raw).lower()
        referred_cards_raw: List[str] = data["referred_cards"] or []
        referred_cards = [word.lower() for word in referred_cards_raw]
        hint = Hint(word=word, card_amount=len(referred_cards), for_words=referred_cards)
        log.info(f"Parsed hint: {hint}. Extra: {extra}")
        return hint

    @classmethod
    def _parse_word(cls, word: str) -> str:
        log.debug(f"Parsing hint word: '{word}'")
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
