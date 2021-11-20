import logging
from threading import Semaphore, Thread
from typing import Iterable, List, Optional

from codenames.game import GameManager, Guess, Guesser, Hint, Hinter, Player, Winner
from codenames.online import (
    IllegalOperation,
    NamecodingLanguage,
    NamecodingPlayerAdapter,
)

log = logging.getLogger(__name__)


class OnlineGameError(Exception):
    pass


class NamecodingGameManager:
    def __init__(
        self,
        blue_hinter: Hinter,
        red_hinter: Hinter,
        blue_guesser: Guesser,
        red_guesser: Guesser,
        show_host: bool = True,
    ):
        self.host: Optional[NamecodingPlayerAdapter] = None
        self.guests: List[NamecodingPlayerAdapter] = []
        self._game_manager = GameManager(
            blue_hinter=blue_hinter, red_hinter=red_hinter, blue_guesser=blue_guesser, red_guesser=red_guesser
        )
        self._show_host = show_host
        self._running_game_id: Optional[str] = None
        self._auto_start_semaphore = Semaphore()
        self._language: NamecodingLanguage = NamecodingLanguage.HEBREW
        self._game_manager.hint_given_subscribers.append(self._handle_hint_given)
        self._game_manager.guess_given_subscribers.append(self._handle_guess_given)

    @property
    def adapters(self) -> Iterable[NamecodingPlayerAdapter]:
        yield self.host  # type: ignore
        yield from self.guests

    @property
    def winner(self) -> Optional[Winner]:
        return self._game_manager.winner

    def _get_adapter_for_player(self, player: Player) -> NamecodingPlayerAdapter:
        for adapter in self.adapters:
            if adapter.player == player:
                return adapter
        raise ValueError(f"Player {player} not found in this game manager.")

    def auto_start(
        self, language: NamecodingLanguage = NamecodingLanguage.ENGLISH, clock: bool = True
    ) -> "NamecodingGameManager":
        number_of_guests = 3
        self._auto_start_semaphore = Semaphore(value=number_of_guests)
        for player in self._game_manager.players:
            if not self.host:
                self.host_game(host_player=player)
                self.configure_game(language=language, clock=clock)
            else:
                self._auto_start_semaphore.acquire()
                log.debug("Semaphore acquired.")
                self.add_to_game(guest_player=player, multithreaded=True)
        if not self._running_game_id:
            log.warning("Game not running after auto start.")
            return self
        for i in range(number_of_guests):
            self._auto_start_semaphore.acquire()
            log.debug(f"Thread {i} done.")
        log.info(f"All {number_of_guests} joined, starting.")
        self.run_game()
        return self

    def host_game(self, host_player: Player) -> "NamecodingGameManager":
        if self.host:
            raise IllegalOperation("A game is already running.")
        host = NamecodingPlayerAdapter(player=host_player, headless=not self._show_host)
        host.open().host_game().choose_role()
        self._running_game_id = host.get_game_id()
        self.host = host
        return self

    def add_to_game(self, guest_player: Player, multithreaded: bool = False) -> "NamecodingGameManager":
        if not self._running_game_id:
            raise IllegalOperation("Can't join game before hosting initiated. Call host_game() first.")
        if not multithreaded:
            guest = NamecodingPlayerAdapter(player=guest_player)
            guest.open().join_game(game_id=self._running_game_id).choose_role()
            self.guests.append(guest)
            self._auto_start_semaphore.release()
            log.debug("Semaphore release")
            return self
        t = Thread(target=self.add_to_game, args=[guest_player, False], daemon=True)
        t.start()
        return self

    def bulk_add_to_game(self, *guests: Player) -> "NamecodingGameManager":
        for guest in guests:
            self.add_to_game(guest_player=guest)
        return self

    def configure_game(
        self, language: NamecodingLanguage = NamecodingLanguage.ENGLISH, clock: bool = True
    ) -> "NamecodingGameManager":
        if not self.host:
            raise IllegalOperation("Can't configure game before hosting initiated. Call host_game() first.")
        self._language = language
        self.host.set_language(language=language)
        self.host.set_clock(clock=clock)
        return self

    def run_game(self):
        self._start_game()
        board = self.host.parse_board()
        self._game_manager.run_game(language=self._language.value, board=board)

    def _start_game(self) -> "NamecodingGameManager":
        if not self.host:
            raise IllegalOperation("Can't start game before hosting initiated. Call host_game() first.")
        self.host.ready()
        for guest in self.guests:
            guest.ready()
        self.host.start_game()
        return self

    def _handle_hint_given(self, hinter: Hinter, hint: Hint):
        adapter = self._get_adapter_for_player(player=hinter)
        adapter.transmit_hint(hint=hint)

    def _handle_guess_given(self, guesser: Guesser, guess: Guess):
        adapter = self._get_adapter_for_player(player=guesser)
        adapter.transmit_guess(guess=guess)

    def close(self):
        log.info("Closing online manager...")
        for guest in self.guests:
            guest.close()
        if self.host:
            self.host.close()
