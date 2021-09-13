import logging

from gensim.models import KeyedVectors

from codenames.game.base import GuesserGameState, Guess, TeamColor, Board
from codenames.game.manager import PASS_GUESS
from codenames.game.player import Guesser
from codenames.solvers.naive.naive_hinter import format_word
from codenames.solvers.utils.model_loader import load_language

log = logging.getLogger(__name__)


class NaiveGuesser(Guesser):
    def __init__(self, name: str, team_color: TeamColor):
        super().__init__(name=name, team_color=team_color)
        self.model: KeyedVectors = None  # type: ignore

    def notify_game_starts(self, language: str, board: Board):
        self.model = load_language(language=language)

    def guess(self, game_state: GuesserGameState) -> Guess:
        if game_state.bonus_given:
            log.debug("Naive guesser does not take bonuses.")
            return Guess(PASS_GUESS)
        optional_words = [card.word for card in game_state.board.unrevealed_cards]
        current_hint_word = format_word(game_state.current_hint.word)
        guess_word = self.model.most_similar_to_given(current_hint_word, optional_words)
        log.debug(f"Naive guesser thinks '{current_hint_word}' means '{guess_word}'.")
        guess_idx = game_state.board.all_words.index(guess_word)
        guess = Guess(card_index=guess_idx)
        return guess
