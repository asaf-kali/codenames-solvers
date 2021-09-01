from src.game import Team, Game


class AbstractHinter:
    def __init__(self, team: Team, game: Game):
        self.team = team
        self.game = game

    def pick_hint(self) -> str:
        raise NotImplementedError()
