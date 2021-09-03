import logging
from threading import Thread, Semaphore
from typing import List, Optional

from codenames.game.player import Player
from codenames.online.online_adapter import NamecodingPlayerAdapter, NamecodingLanguage, IllegalOperation

log = logging.getLogger(__name__)


class NamecodingGameManager:
    def __init__(self, *players: Player):
        self.running_game_id: Optional[str] = None
        self.host: Optional[NamecodingPlayerAdapter] = None
        self.guests: List[NamecodingPlayerAdapter] = []
        self.auto_start_semaphore = Semaphore()
        if players is not None:
            self.auto_start(*players)

    def auto_start(
        self, *players: Player, language: NamecodingLanguage = NamecodingLanguage.ENGLISH, clock: bool = True
    ) -> "NamecodingGameManager":
        number_of_guests = len(players) - 1
        self.auto_start_semaphore = Semaphore(value=number_of_guests)
        for player in players:
            if not self.host:
                self.host_game(host=player)
            else:
                self.auto_start_semaphore.acquire()
                log.debug("Semaphore acquire")
                self.join_game(guest=player, multithreaded=True)
        if not self.running_game_id:
            log.warning("Game not running after auto start.")
            return self
        self.configure(language=language, clock=clock)
        for i in range(number_of_guests):
            self.auto_start_semaphore.acquire()
            log.debug(f"Thread {i} done")
        log.debug(f"All {number_of_guests} done, starting")
        self.start_game()
        return self

    def host_game(self, host: Player) -> "NamecodingGameManager":
        if self.host:
            raise IllegalOperation("A game is already running.")
        host_adapter = NamecodingPlayerAdapter(player=host, headless=False)
        host_adapter.open().host_game().choose_role()
        self.running_game_id = host_adapter.get_game_id()
        self.host = host_adapter
        return self

    def join_game(self, guest: Player, multithreaded: bool = False) -> "NamecodingGameManager":
        if not self.running_game_id:
            raise IllegalOperation("Can't join game before hosting initiated. Call host_game() first.")
        if not multithreaded:
            guest_adapter = NamecodingPlayerAdapter(player=guest)
            guest_adapter.open().join_game(game_id=self.running_game_id).choose_role()
            self.guests.append(guest_adapter)
            self.auto_start_semaphore.release()
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

    def close(self):
        if self.host:
            self.host.close()
        for guest in self.guests:
            guest.close()
