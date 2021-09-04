import logging
import time
from enum import Enum
from time import sleep
from typing import Callable

from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement

from codenames.game.player import PlayerRole, Player
from codenames.utils import wrap

log = logging.getLogger(__name__)

WEBDRIVER_FOLDER = "codenames/online"
WEBAPP_URL = "https://namecoding.herokuapp.com/"


class NamecodingLanguage(Enum):
    ENGLISH = "english"
    HEBREW = "hebrew"


class IllegalOperation(Exception):
    pass


class PollingTimeout(Exception):
    pass


def poll(test: Callable[[], bool], timeout_seconds: float = 5, polls_per_second: int = 3):
    sleep_time = 1 / polls_per_second
    start = time.time()
    while not test():
        now = time.time()
        passed = now - start
        if passed >= timeout_seconds:
            raise PollingTimeout()
        sleep(sleep_time)


class NamecodingPlayerAdapter:
    def __init__(self, player: Player, implicitly_wait: int = 1, headless: bool = True):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("headless")
        self.driver = webdriver.Chrome(f"{WEBDRIVER_FOLDER}/chromedriver", options=options)
        self.driver.implicitly_wait(implicitly_wait)
        self.player = player

    # Utils #

    @property
    def log_prefix(self) -> str:
        return wrap(self.player.name)

    def get_shadow_root(self, tag_name: str, parent=None) -> WebElement:
        if not parent:
            parent = self.driver
        element = parent.find_element_by_tag_name(tag_name)
        shadow_root = self.driver.execute_script("return arguments[0].shadowRoot", element)
        return shadow_root

    # Pages #

    @property
    def codenames_app(self) -> WebElement:
        return self.get_shadow_root("codenames-app")

    def get_page(self, page_name: str) -> WebElement:
        return self.get_shadow_root(page_name, parent=self.codenames_app)

    def get_login_page(self) -> WebElement:
        return self.get_page("login-page")

    def get_menu_page(self) -> WebElement:
        return self.get_page("menu-page")

    def get_lobby_page(self) -> WebElement:
        return self.get_page("lobby-page")

    # Methods #

    def open(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is logging in...")
        self.driver.get(WEBAPP_URL)
        login_page = self.get_login_page()
        username_textbox = login_page.find_element_by_id("username-input")
        login_button = login_page.find_element_by_id("login-button")
        username_textbox.send_keys(self.player.name)
        login_button.click()
        return self

    def host_game(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is hosting...")
        menu_page = self.get_menu_page()
        host_button = menu_page.find_element_by_id("host-button")
        host_button.click()
        return self

    def join_game(self, game_id: str) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is joining game {wrap(game_id)}...")
        menu_page = self.get_menu_page()
        game_id_input = menu_page.find_element_by_id("game-id-input")
        join_game_button = menu_page.find_element_by_id("join-game-button")
        game_id_input.send_keys(game_id)
        join_game_button.click()
        return self

    def get_game_id(self) -> str:
        lobby_page = self.get_lobby_page()
        game_id_container = lobby_page.find_element_by_id("game-code")
        return game_id_container.text

    def choose_role(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is picking role...")
        lobby_page = self.get_lobby_page()
        team_element_id = f"{self.player.team_color.value.lower()}-team"
        role_button_class_name = "guessers" if self.player.role == PlayerRole.GUESSER else "hinters"
        team_element = lobby_page.find_element_by_id(team_element_id)
        role_button = team_element.find_element_by_class_name(role_button_class_name)
        role_button.click()
        return self

    def set_language(self, language: NamecodingLanguage) -> "NamecodingPlayerAdapter":
        lobby_page = self.get_lobby_page()
        options_section = lobby_page.find_element_by_id("options-section")
        language_options = self.get_shadow_root("x-options", options_section)
        button_index = 0 if language == NamecodingLanguage.HEBREW else 1
        buttons = language_options.find_elements_by_tag_name("x-button")
        buttons[button_index].click()
        return self

    def set_clock(self, clock: bool) -> "NamecodingPlayerAdapter":
        lobby_page = self.get_lobby_page()
        options_section = lobby_page.find_element_by_id("options-section")
        checkbox = options_section.find_element_by_tag_name("x-checkbox")
        is_checked_now = checkbox.get_attribute("value") is not None
        if is_checked_now != clock:
            checkbox.click()
        return self

    def ready(self, ready: bool = True) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is ready!")
        lobby_page = self.get_lobby_page()
        switch = lobby_page.find_element_by_id("ready-switch")
        is_checked_now = switch.get_attribute("value") is not None
        if is_checked_now != ready:
            switch.click()
        return self

    def start_game(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is starting the game!")
        lobby_page = self.get_lobby_page()
        start_game_button = lobby_page.find_element_by_id("start-game-button")
        poll(lambda: start_game_button.get_attribute("disabled") is None)
        start_game_button.click()
        return self

    def close(self):
        self.driver.close()
