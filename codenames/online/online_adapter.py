import logging
from enum import Enum
from time import sleep

from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.remote.webelement import WebElement

from codenames.game.base import Board, Card, CardColor, Guess, Hint
from codenames.game.manager import PASS_GUESS
from codenames.game.player import Player, PlayerRole
from codenames.online.utils import poll_condition
from codenames.utils import wrap

log = logging.getLogger(__name__)

WEBDRIVER_FOLDER = "codenames/online"
WEBAPP_URL = "https://namecoding.herokuapp.com/"
CLEAR = "\b\b\b\b\b"


class NamecodingLanguage(Enum):
    ENGLISH = "english"
    HEBREW = "hebrew"


class IllegalOperation(Exception):
    pass


def fill_input(element: WebElement, value: str):
    element.send_keys(CLEAR)
    element.send_keys(CLEAR)
    sleep(0.1)
    element.send_keys(value)
    sleep(0.1)


def _parse_card(card_element: WebElement) -> Card:
    word = card_element.find_element_by_id("bottom").text.strip().lower()
    namecoding_color = card_element.find_element_by_id("right").get_attribute("team")
    card_color = parse_card_color(namecoding_color=namecoding_color)
    image_overlay = card_element.find_element_by_id("image-overlay")
    revealed = image_overlay.get_attribute("revealed") is not None
    log.debug(f"Parsed card: {word}")
    return Card(word=word, color=card_color, revealed=revealed)


class NamecodingPlayerAdapter:
    def __init__(self, player: Player, implicitly_wait: int = 1, headless: bool = True):
        options = webdriver.ChromeOptions()
        if player.is_human:
            headless = False
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

    def __str__(self) -> str:
        return f"{self.player} adapter"

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

    def get_game_page(self) -> WebElement:
        return self.get_page("codenames-game")

    def get_clue_area(self) -> WebElement:
        return self.get_shadow_root("clue-area", parent=self.get_game_page())

    # Methods #

    def open(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is logging in...")
        self.driver.get(WEBAPP_URL)
        login_page = self.get_login_page()
        username_textbox = login_page.find_element_by_id("username-input")
        login_button = login_page.find_element_by_id("login-button")
        fill_input(username_textbox, self.player.name)
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
        fill_input(game_id_input, game_id)
        join_game_button.click()
        return self

    def get_game_id(self) -> str:
        lobby_page = self.get_lobby_page()
        game_id_container = lobby_page.find_element_by_id("game-code")
        return game_id_container.text

    def choose_role(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is picking role...")
        lobby_page = self.get_lobby_page()
        team_element_id = f"{self.player.team_color.value.lower()}-team"  # type: ignore
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
        sleep(0.1)
        return self

    def set_clock(self, clock: bool) -> "NamecodingPlayerAdapter":
        lobby_page = self.get_lobby_page()
        options_section = lobby_page.find_element_by_id("options-section")
        checkbox = options_section.find_element_by_tag_name("x-checkbox")
        is_checked_now = checkbox.get_attribute("value") is not None
        if is_checked_now != clock:
            checkbox.click()
            sleep(0.1)
        return self

    def ready(self, ready: bool = True) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is ready!")
        lobby_page = self.get_lobby_page()
        switch = lobby_page.find_element_by_id("ready-switch")
        is_checked_now = switch.get_attribute("value") is not None
        if is_checked_now != ready:
            switch.click()
            sleep(0.1)
        return self

    def start_game(self) -> "NamecodingPlayerAdapter":
        log.info(f"{self.log_prefix} is starting the game!")
        lobby_page = self.get_lobby_page()
        start_game_button = lobby_page.find_element_by_id("start-game-button")
        poll_condition(lambda: start_game_button.get_attribute("disabled") is None)
        start_game_button.click()
        return self

    def is_my_turn(self) -> bool:
        clue_area = self.get_clue_area()
        if self.player.role == PlayerRole.HINTER and clue_area.find_elements_by_id("submit-clue-button") != []:
            return True
        if self.player.role == PlayerRole.GUESSER and clue_area.find_elements_by_id("finish-turn-button") != []:
            return True
        return False

    def parse_board(self) -> Board:
        log.debug("Parsing board...")
        game_page = self.get_game_page()
        card_containers = game_page.find_elements_by_id("card-padding-container")
        card_elements = [self.get_shadow_root("card-element", card_container) for card_container in card_containers]
        cards = [_parse_card(card_element) for card_element in card_elements]
        log.debug("Parse board done")
        return Board(cards)

    def transmit_hint(self, hint: Hint) -> "NamecodingPlayerAdapter":
        log.debug(f"Sending hint: {hint}")
        clue_area = self.get_clue_area()
        clue_input = clue_area.find_element_by_id("clue-input")
        cards_input = clue_area.find_element_by_id("cards-input")
        submit_clue_button = clue_area.find_element_by_id("submit-clue-button")
        fill_input(clue_input, hint.word.title())
        fill_input(cards_input, str(hint.card_amount))
        submit_clue_button.click()
        sleep(0.2)
        self.approve_alert()
        sleep(0.5)
        return self

    def approve_alert(self, max_tries: int = 20, interval_seconds: float = 0.5):
        log.debug("Approve alert called.")
        tries = 0
        while True:
            tries += 1
            try:
                self.driver.switch_to.alert.accept()
                log.debug("Alert found.")
                return
            except NoAlertPresentException as e:
                if tries >= max_tries:
                    log.warning(f"Alert not found after {max_tries} tries, quitting.")
                    raise e
                log.debug(f"Alert not found, sleeping {interval_seconds} seconds.")
                sleep(interval_seconds)

    def transmit_guess(self, guess: Guess) -> "NamecodingPlayerAdapter":
        log.debug(f"Sending guess: {guess}")
        game_page = self.get_game_page()
        if guess.card_index == PASS_GUESS:
            clue_area = self.get_clue_area()
            finish_turn_button = clue_area.find_element_by_id("finish-turn-button")
            finish_turn_button.click()
            sleep(0.2)
            self.approve_alert()
        else:
            card_containers = game_page.find_elements_by_id("card-padding-container")
            selected_card = card_containers[guess.card_index]
            selected_card.click()
        sleep(0.5)
        return self

    def close(self):
        self.driver.close()


def parse_card_color(namecoding_color: str) -> CardColor:
    namecoding_color = namecoding_color.strip().upper()
    if namecoding_color == "GREEN":
        namecoding_color = "GRAY"
    return CardColor[namecoding_color]
