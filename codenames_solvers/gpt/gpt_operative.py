import json
import logging
from typing import Optional

from codenames.classic.player import ClassicOperative
from codenames.classic.state import ClassicOperativeState
from codenames.generic.board import Board
from codenames.generic.card import Card
from codenames.generic.move import PASS_GUESS, GivenClue, Guess
from codenames.generic.state import OperativeState

from codenames_solvers.gpt.gpt_player import (
    GUESSER_TURN_COMMAND,
    GPTPlayer,
    extract_data_from_response,
)

log = logging.getLogger(__name__)


class GPTOperative(GPTPlayer, ClassicOperative):
    def guess(self, game_state: ClassicOperativeState) -> Guess:
        # legal_guesses = self.build_legal_guesses(board=game_state.board)
        # illegal_guesses = self.build_illegal_guesses(board=game_state.board)
        board_repr = self.build_board_repr(board=game_state.board)
        team = self.build_team_repr()
        moves = self.build_moves_repr(state=game_state)
        # score_status = self.build_score_repr(score=game_state.score)
        options = self.build_options(state=game_state)
        clue = self.build_clue_repr(clue=game_state.current_clue)
        you_already_guessed = self.build_you_already_guesses(state=game_state)
        # pylint: disable=R0801
        infos = [
            # SHORT_INSTRUCTIONS,
            # score_status,
            # legal_guesses,
            # illegal_guesses,
            board_repr,
            team,
            moves,
            # score_status,
            options,
            GUESSER_TURN_COMMAND,
            clue,
            you_already_guessed,
            GUESSER_TURN_COMMAND,
            options,
        ]
        messages = [{"role": "system", "content": info} for info in infos if info is not None]
        try:
            result = self.generate_completion(messages=messages)
            return self.parse_guess(completion_result=result, game_state=game_state)
        except Exception as e:
            log.exception(f"Error while generating guess: {e}")
            return Guess(card_index=PASS_GUESS)

    @classmethod
    def build_clue_repr(cls, clue: GivenClue) -> str:
        return f"Clue word: '{clue.formatted_word}', {clue.card_amount} cards."

    @classmethod
    def build_options(cls, state: OperativeState) -> str:
        legal_guesses = [card.formatted_word for card in state.board.cards if not card.revealed]
        illegal_guesses = [card.formatted_word for card in state.board.cards if card.revealed] + [
            clue.word for clue in state.given_clues
        ]
        assert set(legal_guesses).isdisjoint(set(illegal_guesses))
        options = {
            "legal_guesses": legal_guesses,
            "illegal_guesses": illegal_guesses,
        }
        return json.dumps(options)

    # @classmethod
    # def build_legal_guesses(cls, board: Board) -> str:
    #     words = [card.formatted_word for card in board.cards if not card.revealed]
    #     joined = ", ".join(words)
    #     return f"Guess options - the given guess word HAS TO BE EXACTLY IDENTICAL to one of the following: {joined}."

    # @classmethod
    # def build_illegal_guesses(cls, board: Board) -> Optional[str]:
    #     words = [card.formatted_word for card in board.cards if card.revealed]
    #     if not words:
    #         return None
    #     joined = ", ".join(words)
    #     return f"Illegal guesses - DO NOT provide any guess word from the following list: {joined}."

    @classmethod
    def build_you_already_guesses(cls, state: OperativeState) -> Optional[str]:
        current_turn_guesses = [guess for guess in state.given_guesses if guess.for_clue == state.current_clue]
        if not current_turn_guesses:
            return None
        words = [guess.guessed_card.word for guess in current_turn_guesses]
        joined = ", ".join(words)
        return f"You already guessed: {joined}, DO NOT repeat any of these guesses!"

    @classmethod
    def build_board_repr(cls, board: Board) -> str:
        words = [_card_repr(card) for card in board.cards]
        joined = ", ".join(words)
        return f"Board cards: {joined}."

    def parse_guess(self, completion_result: dict, game_state: OperativeState) -> Guess:
        data = extract_data_from_response(completion_result=completion_result)
        extra = data.get("extra")
        word = _parse_guess_word(raw=data["word"])
        log.info(f"Parsed guess: '{word}'. Extra: {extra}")
        card_index = game_state.board.find_card_index(word=word)
        guess = Guess(card_index=card_index)
        if game_state.board[card_index].revealed:
            raise ValueError(f"Guessed card '{word}' is already revealed!")
        return guess


def _parse_guess_word(raw: str) -> str:
    return raw.strip().lower()


def _card_repr(card: Card) -> str:
    if card.revealed:
        return f"'{card.word}' ({card.color})"
    return f"{card.word} (Unknown)"
