import json
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROMPTS_FILE_NAME = "prompts.json"


def load_instructions() -> dict:
    file_path = os.path.join(CURRENT_DIRECTORY, PROMPTS_FILE_NAME)
    with open(file_path, "r") as file:  # noqa: W1514
        data = json.load(file)
    return data
