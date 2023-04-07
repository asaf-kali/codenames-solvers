import json
import logging
from typing import List

from codenames.game import Hint, Hinter, HinterGameState
from openai import ChatCompletion

from solvers.gpt.gpt_player import SHORT_INSTRUCTIONS, TURN_COMMAND, GPTPlayer

log = logging.getLogger(__name__)


class GPTHinter(GPTPlayer, Hinter):
    def pick_hint(self, game_state: HinterGameState) -> Hint:
        score_status = self.build_score_repr(board=game_state.board)
        team = f"You are the {self.team_color} team {self.role}."
        board_repr = self.build_board_repr(board=game_state.board)
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
            SHORT_INSTRUCTIONS,
            board_repr,
            score_status,
            team,
            hinted_words,
            avoid_words,
            assassin,
            disallowed_hints,
            TURN_COMMAND,
        ]
        messages = [
            # {"role": "user", "content": single_command_prompt},
            {"role": "system", "content": info}
            for info in infos
        ]
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        result = ChatCompletion.create(model=self.model_name, messages=messages, api_key=self.api_key)
        log.debug("Got completion result", extra={"result": result})
        return self.parse_hint(completion_result=result)

    @classmethod
    def parse_hint(cls, completion_result: dict) -> Hint:
        data_raw = completion_result["choices"][0]["message"]["content"]
        data = json.loads(data_raw)
        extra = data.get("extra")
        word_raw: str = data["word"]
        word = cls._parse_word(word_raw).lower()
        referred_cards_raw: List[str] = data["referred_cards"] or []
        referred_cards = [word.lower() for word in referred_cards_raw]
        hint = Hint(word=word, card_amount=len(referred_cards), for_words=referred_cards)
        log.info(f"Got hint: {hint}. Extra: {extra}")
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
