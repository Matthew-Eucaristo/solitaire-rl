"""Deal correctness and card primitives."""



from solitaire_rl.cards import Card, new_deck
from solitaire_rl.state import NUM_TABLEAU, deal


def test_deck_size_and_unique():
    deck = new_deck()
    assert len(deck) == 52
    assert len({c.id for c in deck}) == 52


def test_card_roundtrip():
    for c in new_deck():
        assert Card.from_id(c.id) == c


def test_card_color():
    assert Card(5, 0).is_red is False  # clubs
    assert Card(5, 1).is_red  # diamonds
    assert Card(5, 2).is_red  # hearts
    assert Card(5, 3).is_red is False  # spades


def test_deal_layout():
    st = deal(0)
    assert len(st.tableau) == NUM_TABLEAU
    total = 0
    for i, col in enumerate(st.tableau):
        assert len(col.down) == i
        assert len(col.up) == 1
        total += i + 1
    assert total == 28
    assert len(st.stock) == 24
    assert st.waste == []
    assert all(f == [] for f in st.foundations)


def test_deal_no_duplicate_cards():
    st = deal(7)
    seen = set()
    for col in st.tableau:
        for c in col.down + col.up:
            assert c.id not in seen
            seen.add(c.id)
    for c in st.stock:
        assert c.id not in seen
        seen.add(c.id)
    assert len(seen) == 52


def test_deal_determinism_same_seed():
    a, b = deal(123), deal(123)
    assert a.signature() == b.signature()


def test_deal_differs_across_seeds():
    assert deal(1).signature() != deal(2).signature()


def test_deal_variants_share_layout():
    a, b = deal(5, "draw1"), deal(5, "draw3")
    assert a.draw_size == 1 and b.draw_size == 3
    assert a.stock == b.stock
    assert [c.up for c in a.tableau] == [c.up for c in b.tableau]
