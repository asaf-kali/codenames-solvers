import logging

from codenames.game.exceptions import CardNotFoundError
from codenames.game.move import PASS_GUESS, Guess
from codenames.game.player import Guesser
from codenames.game.state import GuesserGameState

from solvers.naive.naive_player import NaivePlayer

log = logging.getLogger(__name__)


class NaiveGuesser(NaivePlayer, Guesser):
    def guess(self, game_state: GuesserGameState) -> Guess:
        if game_state.left_guesses == 1:
            log.debug("Naive guesser does not take bonuses.")
            return Guess(card_index=PASS_GUESS)
        optional_words = [self.model_format(card.word) for card in game_state.board.unrevealed_cards]
        current_hint_word = game_state.current_hint.formatted_word
        model_formatted_hint = self.model_format(current_hint_word)
        model_guess_word = self.model.most_similar_to_given(model_formatted_hint, optional_words)
        board_guess_word = self.board_format(model_guess_word)
        log.debug(f"Naive guesser thinks '{current_hint_word}' means '{board_guess_word}'.")
        try:
            guess_idx = game_state.board.find_card_index(board_guess_word)
        except CardNotFoundError:
            guess_idx = game_state.board.find_card_index(model_guess_word)
        guess = Guess(card_index=guess_idx)
        return guess
