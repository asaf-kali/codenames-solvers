import logging
import re
from abc import ABC
from typing import List, Optional

from codenames.game import Player, Score, TeamColor
from openai import ChatCompletion

from solvers.gpt.instructions import load_instructions

log = logging.getLogger(__name__)
INSTRUCTIONS = load_instructions()
FULL_INSTRUCTIONS = INSTRUCTIONS["full_instructions"]
SHORT_INSTRUCTIONS = INSTRUCTIONS["short_instructions"]
HINTER_TURN_COMMAND = INSTRUCTIONS["hinter_turn_command"]
GUESSER_TURN_COMMAND = INSTRUCTIONS["guesser_turn_command"]


class GPTPlayer(Player, ABC):
    def __init__(
        self,
        name: str,
        api_key: str,
        team_color: Optional[TeamColor] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
    ):
        super().__init__(name=name, team_color=team_color)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def build_team_repr(self):
        return f"You are the {self.team_color} team {self.role}."

    @classmethod
    def build_score_repr(cls, score: Score) -> str:
        return (
            f"The current score is: "
            f"Red: {score.red.revealed}/{score.red.total}, "
            f"Blue: {score.blue.revealed}/{score.blue.total}."
        )

    def generate_completion(self, messages: List[dict]) -> dict:
        log.debug("Sending completion request", extra={"payload_size": len(str(messages)), "messages": messages})
        response = ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        log.debug("Got completion response", extra={"response": response})
        return response


def find_json_in_string(data: str) -> str:
    match = re.search(r"\{.*}", data)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in string")
