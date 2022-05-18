from typing import List

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


class ModelFormatAdapter:
    def to_model_format(self, word: str) -> str:
        raise NotImplementedError()

    def to_model_formats(self, word: str) -> List[str]:
        return [self.to_model_format(word)]

    def to_board_format(self, word: str) -> str:
        raise NotImplementedError()

    def to_board_formats(self, word: str) -> List[str]:
        return [self.to_board_format(word)]


class DefaultFormatAdapter(ModelFormatAdapter):
    def to_model_format(self, word: str) -> str:
        return word

    def to_board_format(self, word: str) -> str:
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
    def to_model_format(self, word: str) -> str:
        return replace_to_non_suffix_letter(word)

    def to_board_format(self, word: str) -> str:
        if word.endswith("סקופ"):
            return word
        return replace_to_suffix_letter(word)


DEFAULT_MODEL_ADAPTER = DefaultFormatAdapter()
HEBREW_SUFFIX_ADAPTER = HebrewSuffixAdapter()
