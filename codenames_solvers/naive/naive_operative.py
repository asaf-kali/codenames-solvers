import logging

from codenames.generic.exceptions import CardNotFoundError
from codenames.generic.move import PASS_GUESS, Guess
from codenames.generic.player import Operative
from codenames.generic.state import OperativeState

from codenames_solvers.naive.naive_player import NaivePlayer

log = logging.getLogger(__name__)


class NaiveOperative(NaivePlayer, Operative):
    def guess(self, game_state: OperativeState) -> Guess:
        if len(game_state.turn_guesses) == game_state.current_clue.card_amount:
            log.debug("Naive operative does not take extra guesses.")
            return Guess(card_index=PASS_GUESS)
        optional_words = [self.model_format(card.word) for card in game_state.board.unrevealed_cards]
        current_clue_word = game_state.current_clue.formatted_word
        model_formatted_clue = self.model_format(current_clue_word)
        model_guess_word = self.model.most_similar_to_given(model_formatted_clue, optional_words)
        board_guess_word = self.board_format(model_guess_word)
        log.debug(f"Naive operative thinks '{current_clue_word}' means '{board_guess_word}'.")
        try:
            guess_idx = game_state.board.find_card_index(board_guess_word)
        except CardNotFoundError:
            guess_idx = game_state.board.find_card_index(model_guess_word)
        guess = Guess(card_index=guess_idx)
        return guess
