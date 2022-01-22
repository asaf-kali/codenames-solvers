from codenames.game.base import Board, Card, CardColor


def board_10() -> Board:
    return Board(
        [
            Card("Card 0", color=CardColor.BLUE),  # 0
            Card("Card 1", color=CardColor.BLUE),  # 1
            Card("Card 2", color=CardColor.BLUE),  # 2
            Card("Card 3", color=CardColor.BLUE),  # 3
            Card("Card 4", color=CardColor.RED),  # 4
            Card("Card 5", color=CardColor.RED),  # 5
            Card("Card 6", color=CardColor.RED),  # 6
            Card("Card 7", color=CardColor.GRAY),  # 7
            Card("Card 8", color=CardColor.GRAY),  # 8
            Card("Card 9", color=CardColor.BLACK),  # 9
        ]
    )


def board_25() -> Board:
    return Board(
        [
            Card("Card 0", color=CardColor.BLUE),
            Card("Card 1", color=CardColor.BLUE),
            Card("Card 2", color=CardColor.BLUE),
            Card("Card 3", color=CardColor.BLUE),
            Card("Card 4", color=CardColor.BLUE),
            Card("Card 5", color=CardColor.BLUE),
            Card("Card 6", color=CardColor.BLUE),
            Card("Card 7", color=CardColor.BLUE),
            Card("Card 8", color=CardColor.BLUE),
            Card("Card 9", color=CardColor.RED),
            Card("Card 10", color=CardColor.RED),
            Card("Card 11", color=CardColor.RED),
            Card("Card 12", color=CardColor.RED),
            Card("Card 13", color=CardColor.RED),
            Card("Card 14", color=CardColor.RED),
            Card("Card 15", color=CardColor.RED),
            Card("Card 16", color=CardColor.RED),
            Card("Card 17", color=CardColor.GRAY),
            Card("Card 18", color=CardColor.GRAY),
            Card("Card 19", color=CardColor.GRAY),
            Card("Card 20", color=CardColor.GRAY),
            Card("Card 21", color=CardColor.GRAY),
            Card("Card 22", color=CardColor.GRAY),
            Card("Card 23", color=CardColor.GRAY),
            Card("Card 24", color=CardColor.BLACK),
        ]
    )


def hebrew_board() -> Board:
    return Board(
        [
            Card("חיים", color=CardColor.GRAY),
            Card("ערך", color=CardColor.BLACK),
            Card("מסוק", color=CardColor.BLUE),
            Card("שבוע", color=CardColor.GRAY),
            Card("רובוט", color=CardColor.RED),
            Card("פוטר", color=CardColor.GRAY),
            Card("אסור", color=CardColor.BLUE),
            Card("דינוזאור", color=CardColor.BLUE),
            Card("מחשב", color=CardColor.RED),
            Card("מעמד", color=CardColor.GRAY),
            Card("בעל", color=CardColor.RED),
            Card("פנים", color=CardColor.RED),
            Card("פרק", color=CardColor.RED),
            Card("גפילטע", color=CardColor.BLUE),
            Card("שונה", color=CardColor.RED),
            Card("שכר", color=CardColor.RED),
            Card("קפיץ", color=CardColor.BLUE),
            Card("תרסיס", color=CardColor.GRAY),
            Card("דגל", color=CardColor.GRAY),
            Card("חופשה", color=CardColor.BLUE),
            Card("מועדון", color=CardColor.RED),
            Card("ציון", color=CardColor.BLUE),
            Card("שק", color=CardColor.GRAY),
            Card("אקורדיון", color=CardColor.RED),
            Card("ילד", color=CardColor.BLUE),
        ]
    )
