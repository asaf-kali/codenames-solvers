from abc import ABC
from typing import List, Optional

from codenames.game import Board, CardColor, GivenHint, Player, TeamColor

FULL_INSTRUCTIONS = """You are a chatbot playing the Codenames board game. Players are split into two teams: blue and red. One player in each team is selected as the team's spymaster (the Hinter); the rest become field operatives (Guessers).
Twenty-five codename cards, each bearing a word, are laid out in a 5×5 grid in random order. A number of these words represent blue agents, a number represent red agents, one represents an assassin, and the rest represent innocent bystanders. The teams' spymasters are given a randomly-dealt key card showing a 5×5 grid of 25 squares of various colors, each corresponding to one of the codename cards on the table.
On each turn, the appropriate spymaster gives a verbal hint about the words on the respective cards. Each hint may only consist of one single word and a number. The clue has to be related to as many of the words on the team's own agents' cards as possible, but not to any others – lest the words accidentally lead them to choose a card representing an innocent bystander, an opposing agent, or the assassin. The clue word can be chosen freely, as long as it is not (and does not contain, nor is contained in) any of the words on the codename cards still visible at the time. Codename cards are covered as guesses, correct or otherwise, are made.
The number in the hint tells the field operatives how many words in the grid are related to the clue word. It also determines the maximum number of guesses the field operatives may make on that turn, which is the stated number plus one. The field operatives of a team are required to make at least one guess per turn, risking a wrong guess and its consequences. If their first guess is right, the field operatives may continue to make guesses until they reach the guess limit or make a wrong guess, or they can instead choose to end their turn voluntarily.
After a spymaster gives a clue with its word and number, their field operatives make guesses about which codename cards bear words related to the clue and point them out, one at a time. When a codename card is pointed out, the spymaster covers that card with an appropriate identity card – a blue agent, a red agent, an innocent bystander, or the assassin – as indicated on the spymasters' map of the grid. Revealing an opposing agent ends the team's turn, as does revealing an innocent bystander, though in the former case, the opposing team also gets a small advantage before the start of their turn as a result. If the assassin is revealed, the game ends immediately with a loss for the team who identified him.
Besides the aforementioned assassin, the game ends when all agents belonging to one team are identified, winning the game for that team."""
SHORT_INSTRUCTIONS = """Codenames is a board game where two teams, red and blue, compete against each other. Each team has a spymaster who gives hints to their field operatives about which codename cards on the table belong to their team's agents. The goal is to identify all the agents of their respective team before the other team does so. The 25 codename cards are laid out, each card bearing a word representing a blue agent, a red agent, an innocent bystander, or the assassin.
The spymaster gives a verbal hint consisting of one word and a number that indicates the number of codename cards related to the hint. The field operatives must then guess which cards belong to their team's agents and point them out one at a time. The spymaster covers each guessed card with an appropriate identity card. If the operatives guess correctly, they may continue to guess until they reach the guess limit or make a wrong guess. If they guess incorrectly, the turn ends, and play moves to the other team.
The game ends when all agents belonging to one team are identified, winning the game for that team. However, revealing the assassin immediately ends the game with a loss for the team who identified him. If an opposing agent or innocent bystander is identified, the turn ends, and the other team gains a small advantage before starting their turn. The spymaster's hints cannot contain any of the words on the codename cards still visible, and the game continues until one team wins or the assassin is revealed."""


class GPTPlayer(Player, ABC):
    def __init__(self, name: str, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(name)
        self.api_key = api_key
        self.model_name = model_name

    @classmethod
    def build_board_prompt_repr(cls, board: Board) -> str:
        words = [f"{card.word}-{card.color}" for card in board.cards]
        joined = ", ".join(words)
        return f"Board cards: {joined}."

    @classmethod
    def build_score_repr(cls, board: Board) -> str:
        total_red, total_blue_cards = len(board.red_cards), len(board.blue_cards)  # type: ignore
        unrevealed_red = len(board.unrevealed_cards_for_color(CardColor.RED))
        unrevealed_blue = len(board.unrevealed_cards_for_color(CardColor.BLUE))
        return f"The current score status is: Red: {total_red - unrevealed_red}/{total_red}, Blue: {total_blue_cards - unrevealed_blue}/{total_blue_cards}."

    @classmethod
    def build_disallowed_hints_repr(
        cls, board: Board, hints: List[GivenHint], extra: Optional[List[str]] = None
    ) -> str:
        extra = extra or []
        words = [card.word for card in board.cards] + [hint.word for hint in hints] + extra
        if not words:
            return ""
        return f"The following expressions are NOT legal hints: {', '.join(words)}."

    @classmethod
    def build_cards_to_avoid_repr(cls, board: Board, team_color: TeamColor) -> str:
        filter_card_color = team_color.as_card_color
        words = [card.word for card in board.unrevealed_cards if card.color != filter_card_color]
        return f"Avoid giving a hint that is related to any of these words: {', '.join(words)}."

    @classmethod
    def build_assassin_repr(cls, board: Board) -> str:
        words = [card.word for card in board.unrevealed_cards if card.color == CardColor.BLACK]
        return f"The assassin is: {', '.join(words)}, avoid this word at all costs."

    @classmethod
    def build_hinted_words(cls, board: Board, team_color: TeamColor) -> str:
        words = [card.word for card in board.unrevealed_cards_for_color(team_color.as_card_color)]
        return f"These are the words you are looking for hints to: {', '.join(words)}."
