import logging
import random
from typing import List, Optional

from codenames.classic.color import ClassicColor
from codenames.generic.board import Board
from codenames.generic.move import Clue, GivenClue
from codenames.generic.player import Spymaster, Team
from codenames.generic.state import SpymasterState
from codenames.utils.vocabulary.english import ENGLISH_WORDS

from solvers.gpt.gpt_player import (
    HINTER_TURN_COMMAND,
    GPTPlayer,
    extract_data_from_response,
)

log = logging.getLogger(__name__)


class GPTSpymaster(GPTPlayer, Spymaster):
    def give_clue(self, game_state: SpymasterState) -> Clue:
        board_repr = self.build_board_repr(board=game_state.board)
        team = self.build_team_repr()
        moves = self.build_moves_repr(state=game_state)
        # score_status = self.build_score_repr(score=game_state.score)
        words_to_clue = self.build_clueed_words(board=game_state.board, team=self.team)
        words_to_avoid = self.build_cards_to_avoid_repr(board=game_state.board, team=self.team)
        disallowed_clues = self.build_disallowed_clues_repr(
            board=game_state.board, clues=game_state.given_clues, extra=[]
        )
        assassin = self.build_assassin_repr(board=game_state.board)
        # single_command_prompt = (
        #     f"{score_status} {team} {clueed_words} {avoid_words} {assassin} {disallowed_clues} {TURN_COMMAND}"
        # )
        # pylint: disable=R0801
        infos = [
            # SHORT_INSTRUCTIONS,
            board_repr,
            team,
            moves,
            # score_status,
            words_to_clue,
            words_to_avoid,
            disallowed_clues,
            assassin,
            HINTER_TURN_COMMAND,
            words_to_clue,
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
            clue = self.parse_clue(completion_result=result)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Error while generating clue", exc_info=e)
            return Clue(word=_random_english_word(), card_amount=1)
        if clue.word in game_state.illegal_clue_words:
            log.warning(f"Generated a clue that is not allowed: {clue}")
            return Clue(word=_random_english_word(), card_amount=1)
        self._verify_clue(clue=clue, game_state=game_state)
        return clue

    def _verify_clue(self, clue: Clue, game_state: SpymasterState):
        good_clue = True
        for word in clue.for_words:
            if not self._card_valid_for_clue(clue=clue, game_state=game_state, word=word):
                good_clue = False
        if good_clue:
            log.info(f"Clue {clue} is valid")

    def _card_valid_for_clue(self, clue: Clue, game_state: SpymasterState, word: str) -> bool:
        try:
            card = game_state.board[word]
        except KeyError:
            log.warning(f"Clue {clue} is referring to a word that is not on the board: {word}")
            return False
        if card.revealed:
            log.warning(f"Clue {clue} is referring to a word that is already revealed: {word}")
            return False
        if card.color != self.team.as_card_color:
            log.warning(f"Clue {clue} is referring to a word that is not of the team's color: {word}")
            return False
        return True

    @classmethod
    def build_board_repr(cls, board: Board) -> str:
        words = [f"'{card.word}' ({card.color})" for card in board.cards]
        joined = ", ".join(words)
        return f"Board cards: {joined}."

    @classmethod
    def build_disallowed_clues_repr(
        cls, board: Board, clues: List[GivenClue], extra: Optional[List[str]] = None
    ) -> str:
        extra = extra or []
        words = [card.word for card in board.cards] + [clue.word for clue in clues] + extra
        if not words:
            return ""
        return f"The following expressions are NOT legal clues: {', '.join(words)}."

    @classmethod
    def build_cards_to_avoid_repr(cls, board: Board, team: Team) -> str:
        filter_card_color = team.as_card_color
        words = [card.word for card in board.unrevealed_cards if card.color != filter_card_color]
        return f"Avoid giving a clue that is related to any of these words: {', '.join(words)}."

    @classmethod
    def build_assassin_repr(cls, board: Board) -> str:
        words = [card.word for card in board.unrevealed_cards if card.color == ClassicColor.ASSASSIN]
        return f"The assassin word is: {', '.join(words)}, avoid clues suggesting this word as much as possible."

    @classmethod
    def build_clueed_words(cls, board: Board, team: Team) -> str:
        words = [card.word for card in board.unrevealed_cards_for_color(team.as_card_color)]
        return f"You are looking for a clue for this word group: {', '.join(words)}."

    @classmethod
    def parse_clue(cls, completion_result: dict) -> Clue:
        data = extract_data_from_response(completion_result=completion_result)
        extra = data.get("extra")
        word_raw: str = data["word"]
        word = cls._parse_word(word_raw).lower()
        referred_cards_raw: List[str] = data["referred_cards"] or []
        referred_cards = [word.lower() for word in referred_cards_raw]
        clue = Clue(word=word, card_amount=len(referred_cards), for_words=tuple(referred_cards))
        log.info(f"Parsed clue: {clue}. Extra: {extra}")
        return clue

    @classmethod
    def _parse_word(cls, word: str) -> str:
        parts = word.split()
        if len(parts) == 1:
            return word
        if len(parts) > 2:
            raise ValueError(f"Clue word must be a single word or a two-word phrase, got: {word}")
        second = parts[1]
        if not second.isnumeric():
            return word
        log.debug("Got a number as the second word of the clue, assuming it's a number of cards")
        return parts[0]


def _random_english_word() -> str:
    return random.choice(ENGLISH_WORDS)
