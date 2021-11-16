from codenames.game.base import Card, CardColor


def test_censored_when_revealed_is_same_as_card():
    card = Card("Card 3", CardColor.BLUE, True)
    assert card.censored == card


def test_censored_when_not_revealed_does_not_have_color():
    card = Card("Card 3", CardColor.BLUE, False)
    assert card.censored == Card("Card 3", None, False)


def test_cards_can_be_members_of_a_set():
    card1 = Card("Card 1", None, False)
    card2 = Card("Card 1", CardColor.BLUE, False)
    card3 = Card("Card 1", CardColor.BLUE, True)
    card_set = {card1, card2, card3}
    assert len(card_set) == 3
