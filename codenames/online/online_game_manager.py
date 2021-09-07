import logging
from threading import Thread, Semaphore
from typing import List, Optional, Iterable

from codenames.game.manager import GameManager, Winner
from codenames.game.player import Player, Hinter, Guesser
from codenames.online.online_adapter import NamecodingPlayerAdapter, NamecodingLanguage, IllegalOperation
from codenames.online.utils import poll_not_none

log = logging.getLogger(__name__)


class OnlineGameError(Exception):
    pass


class NamecodingGameManager:
    def __init__(self, blue_hinter: Hinter, red_hinter: Hinter, blue_guesser: Guesser, red_guesser: Guesser):
        self.host: Optional[NamecodingPlayerAdapter] = None
        self.guests: List[NamecodingPlayerAdapter] = []
        self._game_manager = GameManager(
            blue_hinter=blue_hinter, red_hinter=red_hinter, blue_guesser=blue_guesser, red_guesser=red_guesser
        )
        self._running_game_id: Optional[str] = None
        self._auto_start_semaphore = Semaphore()
        self._language: NamecodingLanguage = NamecodingLanguage.HEBREW

    @property
    def players(self) -> Iterable[NamecodingPlayerAdapter]:
        yield self.host  # type: ignore
        yield from self.guests

    @property
    def winner(self) -> Optional[Winner]:
        return self._game_manager.winner

    def auto_start(
        self, language: NamecodingLanguage = NamecodingLanguage.ENGLISH, clock: bool = True
    ) -> "NamecodingGameManager":
        number_of_guests = 3
        self._auto_start_semaphore = Semaphore(value=number_of_guests)
        for player in self._game_manager.players:
            if not self.host:
                self.host_game(host=player)
            else:
                self._auto_start_semaphore.acquire()
                log.debug("Semaphore acquired.")
                self.join_game(guest=player, multithreaded=True)
        if not self._running_game_id:
            log.warning("Game not running after auto start.")
            return self
        self.configure(language=language, clock=clock)
        for i in range(number_of_guests):
            self._auto_start_semaphore.acquire()
            log.debug(f"Thread {i} done.")
        log.info(f"All {number_of_guests} joined, starting.")
        self.start_game()
        self.run_game()
        return self

    def host_game(self, host: Player) -> "NamecodingGameManager":
        if self.host:
            raise IllegalOperation("A game is already running.")
        host_adapter = NamecodingPlayerAdapter(player=host, headless=False)
        host_adapter.open().host_game().choose_role()
        self._running_game_id = host_adapter.get_game_id()
        self.host = host_adapter
        return self

    def join_game(self, guest: Player, multithreaded: bool = False) -> "NamecodingGameManager":
        if not self._running_game_id:
            raise IllegalOperation("Can't join game before hosting initiated. Call host_game() first.")
        if not multithreaded:
            guest_adapter = NamecodingPlayerAdapter(player=guest)
            guest_adapter.open().join_game(game_id=self._running_game_id).choose_role()
            self.guests.append(guest_adapter)
            self._auto_start_semaphore.release()
            log.debug("Semaphore release")
            return self
        t = Thread(target=self.join_game, args=[guest, False], daemon=True)
        t.start()
        return self

    def bulk_join_game(self, *guests: Player) -> "NamecodingGameManager":
        for guest in guests:
            self.join_game(guest=guest)
        return self

    def configure(
        self, language: NamecodingLanguage = NamecodingLanguage.ENGLISH, clock: bool = True
    ) -> "NamecodingGameManager":
        if not self.host:
            raise IllegalOperation("Can't configure game before hosting initiated. Call host_game() first.")
        self._language = language
        self.host.set_language(language=language)
        self.host.set_clock(clock=clock)
        return self

    def start_game(self) -> "NamecodingGameManager":
        if not self.host:
            raise IllegalOperation("Can't start game before hosting initiated. Call host_game() first.")
        self.host.ready()
        for guest in self.guests:
            guest.ready()
        self.host.start_game()
        return self

    def get_current_turn_player(self) -> NamecodingPlayerAdapter:
        log.debug("get_current_turn_player called.")
        players = list(self.players)
        for player in players:
            if player.is_my_turn():
                log.debug(f"Found player turn: {player}.")
                return player
        log.warning("Not current turn found.")
        raise OnlineGameError("Couldn't find current player turn.")

    def run_game(self):
        board = self.host.parse_board()
        self._game_manager.initialize_game(language=self._language.value, board=board)
        while not self._game_manager.is_game_over:
            current_player_adapter = poll_not_none(self.get_current_turn_player)
            current_player = current_player_adapter.player
            log.info(f"It is {current_player} turn.")
            if isinstance(current_player, Hinter):
                hint = self._game_manager.get_hint_from(hinter=current_player)
                current_player_adapter.transmit_hint(hint)
            if isinstance(current_player, Guesser):
                guess = self._game_manager.get_guess_from(guesser=current_player)
                current_player_adapter.transmit_guess(guess)

    def close(self):
        log.info("Closing online manager...")
        if self.host:
            self.host.close()
        for guest in self.guests:
            guest.close()
