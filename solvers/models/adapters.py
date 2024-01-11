from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set

ExistenceChecker = Callable[[str], bool]
FormatMapping = Dict[str, str]

SUFFIX_LETTER_TO_NON_SUFFIX_LETTER = {
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
}
NON_SUFFIX_LETTER_TO_SUFFIX_LETTER = {v: k for k, v in SUFFIX_LETTER_TO_NON_SUFFIX_LETTER.items()}
HEBREW_SUFFIX_LETTERS = tuple(SUFFIX_LETTER_TO_NON_SUFFIX_LETTER.keys())
HEBREW_NON_SUFFIX_LETTERS = tuple(SUFFIX_LETTER_TO_NON_SUFFIX_LETTER.values())
COMMON_SPLITTERS = {" ", "-", ".", "_"}


class ModelFormatAdapter(ABC):
    def __init__(
        self,
        existence_checker: Optional[ExistenceChecker] = None,
        board_to_model: Optional[FormatMapping] = None,
        splitters: Optional[Set[str]] = None,
    ):
        self.checker = existence_checker
        self._board_to_model: FormatMapping = {}
        self._model_to_board: FormatMapping = {}
        self._splitters = splitters or COMMON_SPLITTERS
        if board_to_model:
            for board, model in board_to_model.items():
                self.cache_match(board=board, model=model)

    def cache_match(self, board: str, model: str):
        self._board_to_model[board] = model
        self._model_to_board[model] = board

    def clear_cache(self):
        self._board_to_model = {}
        self._model_to_board = {}

    def to_model_format(self, word: str) -> str:
        if word in self._board_to_model:
            return self._board_to_model[word]
        model_format = self._to_model_format(word)
        if not self.checker or self.checker(model_format):
            self.cache_match(board=word, model=model_format)
            return model_format
        variation = self.find_existing_variation(expression=model_format)
        self.cache_match(board=word, model=variation)
        return variation

    def to_board_format(self, word: str) -> str:
        if word in self._model_to_board:
            return self._model_to_board[word]
        board_format = self._to_board_format(word)
        self.cache_match(board=board_format, model=word)
        return board_format

    def find_existing_variation(self, expression: str) -> str:
        """
        Given a string expression that does not exist in the model, tries to find a variation of the expression that
        exists in the model. If no variation is found, returns the original expression.
        """
        if not self.checker:
            return expression
        for splitter in self._splitters:
            if splitter not in expression:
                continue
            tokens = expression.split(splitter)
            without_splitters = "".join(tokens)
            if self.checker(without_splitters):
                return without_splitters
            other_splitters = self._splitters - {splitter}
            for other_splitter in other_splitters:
                joined = other_splitter.join(tokens)
                if self.checker(joined):
                    return joined
        return expression

    @abstractmethod
    def _to_model_format(self, word: str) -> str:
        ...

    @abstractmethod
    def _to_board_format(self, word: str) -> str:
        ...

    def to_model_formats(self, word: str) -> List[str]:
        return [self.to_model_format(word)]

    def to_board_formats(self, word: str) -> List[str]:
        return [self.to_board_format(word)]


class DefaultFormatAdapter(ModelFormatAdapter):
    def _to_model_format(self, word: str) -> str:
        return word.replace(". ", ".").lower()

    def _to_board_format(self, word: str) -> str:
        return word


def replace_to_non_suffix_letter(word: str) -> str:
    if not word.endswith(HEBREW_SUFFIX_LETTERS):
        return word
    suffix_letter = word[-1]
    return word[:-1] + SUFFIX_LETTER_TO_NON_SUFFIX_LETTER[suffix_letter]


def replace_to_suffix_letter(word: str) -> str:
    if not word.endswith(HEBREW_NON_SUFFIX_LETTERS):
        return word
    non_suffix_letter = word[-1]
    return word[:-1] + NON_SUFFIX_LETTER_TO_SUFFIX_LETTER[non_suffix_letter]


class HebrewSuffixAdapter(ModelFormatAdapter):
    def _to_model_format(self, word: str) -> str:
        return replace_to_non_suffix_letter(word)

    def _to_board_format(self, word: str) -> str:
        if word.endswith("סקופ"):
            return word
        return replace_to_suffix_letter(word)


DEFAULT_MODEL_ADAPTER = DefaultFormatAdapter()
HEBREW_SUFFIX_ADAPTER = HebrewSuffixAdapter()
