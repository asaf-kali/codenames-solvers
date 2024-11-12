import json
import logging
import re
from abc import ABC
from typing import List, Optional

from codenames.classic.score import Score
from codenames.generic.move import GivenClue, GivenGuess
from codenames.generic.player import Player, Spymaster, Team
from codenames.generic.state import PlayerState
from openai import ChatCompletion

from solvers.gpt.instructions import load_instructions
from solvers.gpt.moves import ClueMove, GuessMove, Move, PassMove, get_moves

log = logging.getLogger(__name__)
INSTRUCTIONS = load_instructions()
FULL_INSTRUCTIONS = INSTRUCTIONS["full_instructions"]
SHORT_INSTRUCTIONS = INSTRUCTIONS["short_instructions"]
HINTER_TURN_COMMAND = INSTRUCTIONS["spymaster_turn_command"]
GUESSER_TURN_COMMAND = INSTRUCTIONS["operative_turn_command"]


class GPTPlayer(Player, ABC):
    def __init__(
        self,
        name: str,
        api_key: str,
        team: Optional[Team] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        super().__init__(name=name, team=team)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    @property
    def role(self) -> str:
        return "spymaster" if isinstance(self, Spymaster) else "operative"

    def build_team_repr(self):
        return f"You are the {self.team} team {self.role}."

    @classmethod
    def build_score_repr(cls, score: Score) -> str:
        return (
            f"The current score is: "
            f"Red: {score.red.revealed}/{score.red.total}, "
            f"Blue: {score.blue.revealed}/{score.blue.total}."
        )

    @classmethod
    def build_moves_repr(cls, state: PlayerState) -> Optional[str]:
        moves: List[Move] = get_moves(state)
        if not moves:
            return None
        moves_repr = []
        for move in moves:
            if isinstance(move, ClueMove):
                moves_repr.append(clue_repr(clue=move.given_clue))
            elif isinstance(move, GuessMove):
                moves_repr.append(guess_repr(guess=move.given_guess))
            elif isinstance(move, PassMove):
                moves_repr.append(pass_repr(move=move))
        return "\n".join(moves_repr)

    def generate_completion(self, messages: List[dict]) -> dict:
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        response = ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        usage = response.get("usage")
        log.debug(f"Got completion response, usage: {usage}", extra={"response": response})
        return response


def extract_data_from_response(completion_result: dict) -> dict:
    response_content: str = completion_result["choices"][0]["message"]["content"]
    data_raw = find_json_in_string(response_content)
    log.debug(f"Parsing content: {data_raw}")
    data = json.loads(data_raw)
    return data


def find_json_in_string(data: str) -> str:
    match = re.search(r"\{.*}", data)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in string")


def clue_repr(clue: GivenClue) -> str:
    return f"{clue.team} spymaster said: '{clue.word}', {clue.card_amount} cards."


def guess_repr(guess: GivenGuess) -> str:
    return f"{guess.team} operative said: {guess}."


def pass_repr(move: PassMove) -> str:
    return f"{move.team} team operative passed the turn."
