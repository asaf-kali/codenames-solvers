import json
import logging
from typing import Optional

from codenames.game import (
    PASS_GUESS,
    Board,
    Card,
    GivenHint,
    Guess,
    Guesser,
    GuesserGameState,
    TeamColor,
)
from openai import ChatCompletion

from solvers.gpt.gpt_player import GUESSER_TURN_COMMAND, GPTPlayer, find_json_in_string

log = logging.getLogger(__name__)


class GPTGuesser(GPTPlayer, Guesser):
    def __init__(
        self,
        name: str,
        api_key: str,
        team_color: Optional[TeamColor] = None,
        model_name: str = "gpt-3.5-turbo-0301",
    ):
        super().__init__(name=name, api_key=api_key, team_color=team_color, model_name=model_name)

    def guess(self, game_state: GuesserGameState) -> Guess:
        # board_repr = self.build_board_repr(board=game_state.board)
        # score_status = self.build_score_repr(score=game_state.score)
        # team = self.build_team_repr()
        # legal_guesses = self.build_legal_guesses(board=game_state.board)
        # illegal_guesses = self.build_illegal_guesses(board=game_state.board)
        hint = self.build_hint_repr(hint=game_state.current_hint)
        you_already_guessed = self.build_you_already_guesses(state=game_state)
        options = self.build_options(state=game_state)
        # single_command_prompt = (
        #     f"{score_status} {team} {hinted_words} {avoid_words} {assassin} {illegal_hints} {TURN_COMMAND}"
        # )
        infos = [
            # SHORT_INSTRUCTIONS,
            # board_repr,
            # score_status,
            # team,
            # legal_guesses,
            # illegal_guesses,
            hint,
            you_already_guessed,
            GUESSER_TURN_COMMAND,
            options,
        ]
        messages = [{"role": "system", "content": info} for info in infos if info is not None]
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        try:
            result = ChatCompletion.create(
                model=self.model_name, messages=messages, api_key=self.api_key, temperature=0
            )
            log.debug("Got completion result", extra={"result": result})
            return self.parse_guess(completion_result=result, game_state=game_state)
        except Exception as e:  # pylint disable=broad-except
            log.exception(f"Failed to get completion result: {e}")
            return Guess(card_index=PASS_GUESS)

    @classmethod
    def build_hint_repr(cls, hint: GivenHint) -> str:
        return f"Hint word: '{hint.formatted_word}', {hint.card_amount} cards."

    @classmethod
    def build_options(cls, state: GuesserGameState) -> str:
        legal_guesses = [card.formatted_word for card in state.board.cards if not card.revealed]
        illegal_guesses = [card.formatted_word for card in state.board.cards if card.revealed] + [
            hint.word for hint in state.given_hints
        ]
        assert set(legal_guesses).isdisjoint(set(illegal_guesses))
        options = {
            "legal_guesses": legal_guesses,
            "illegal_guesses": illegal_guesses,
        }
        return json.dumps(options)

    @classmethod
    def build_legal_guesses(cls, board: Board) -> str:
        words = [card.formatted_word for card in board.cards if not card.revealed]
        joined = ", ".join(words)
        return f"Guess options - the given guess word HAS TO BE EXACTLY IDENTICAL to one of the following: {joined}."

    @classmethod
    def build_illegal_guesses(cls, board: Board) -> Optional[str]:
        words = [card.formatted_word for card in board.cards if card.revealed]
        if not words:
            return None
        joined = ", ".join(words)
        return f"Illegal guesses - DO NOT provide any guess word from the following list: {joined}."

    @classmethod
    def build_you_already_guesses(cls, state: GuesserGameState) -> Optional[str]:
        current_turn_guesses = [guess for guess in state.given_guesses if guess.given_hint == state.current_hint]
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

    def parse_guess(self, completion_result: dict, game_state: GuesserGameState) -> Guess:
        response_content = completion_result["choices"][0]["message"]["content"]
        data_raw = find_json_in_string(response_content)
        log.debug(f"Parsing guess from: '{data_raw}'")
        data = json.loads(data_raw)
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
        return f"{card.word} ({card.color})"
    return f"{card.word} (Unknown)"
