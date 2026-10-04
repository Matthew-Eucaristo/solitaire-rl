"""Move legality rules — hand-built states covering the legality matrix."""


from solitaire_rl.cards import Card
from solitaire_rl.moves import Move, MoveKind, apply_move, legal_moves
from solitaire_rl.state import GameState, TableauColumn


def mk_state(tableau, stock=None, waste=None, foundations=None, draw_size=1, max_redeals=None):
    return GameState(
        tableau=[TableauColumn(list(d), list(u)) for d, u in tableau],
        stock=list(stock or []),
        waste=list(waste or []),
        foundations=[list(f) for f in (foundations or [[], [], [], []])],
        draw_size=draw_size,
        max_redeals=max_redeals,
    )


C = Card


def test_empty_column_accepts_only_king():
    st = mk_state(
        [
            ([], [C(13, 3)]),  # K spade
            ([], [C(5, 0)]),  # 5 club
            ([], []),
            ([], []),
            ([], []),
            ([], []),
            ([], []),
        ]
    )
    moves = legal_moves(st)
    t2t = [m for m in moves if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
    # K can go to any empty column; 5 cannot.
    assert any(m.src == 0 and m.dst in (2, 3, 4, 5, 6) for m in t2t)
    assert not any(m.src == 1 and m.dst in (2, 3, 4, 5, 6) for m in t2t)


def test_tableau_placement_opposite_color_descending():
    st = mk_state(
        [
            ([], [C(7, 0)]),  # 7♣ (black)
            ([], [C(6, 2)]),  # 6♥ (red) -> can go on 7♣
            ([], [C(6, 3)]),  # 6♠ (black) -> cannot (same color)
            ([], [C(8, 1)]),  # 8♦ -> 6♥ cannot (wrong rank)
            ([], []),
            ([], []),
            ([], []),
        ]
    )
    t2t = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
    assert any(m.src == 1 and m.dst == 0 for m in t2t)
    assert not any(m.src == 2 and m.dst == 0 for m in t2t)
    assert not any(m.src == 3 and m.dst == 0 for m in t2t)


def test_foundation_must_start_at_ace():
    st = mk_state(
        [
            ([], [C(1, 0)]),  # A♣
            ([], [C(2, 0)]),  # 2♣ — cannot start foundation
            ([], []), ([], []), ([], []), ([], []), ([], []),
        ]
    )
    t2f = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_FOUNDATION]
    assert {m.src for m in t2f} == {0}


def test_foundation_ascends_in_suit():
    st = mk_state(
        [
            ([], [C(3, 2)]),  # 3♥ — foundation has A♥2♥ -> legal
            ([], [C(4, 2)]),  # 4♥ — needs 3♥ first -> illegal
            ([], []), ([], []), ([], []), ([], []), ([], []),
        ],
        foundations=[[], [], [C(1, 2), C(2, 2)], []],
    )
    t2f = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_FOUNDATION]
    assert {m.src for m in t2f} == {0}


def test_run_moves_and_partial_runs():
    # col0: K♠ Q♥ ; col1 empty ; col2: A♣ 2♦ 3♠
    st = mk_state(
        [
            ([], [C(13, 3), C(12, 2)]),
            ([], []),
            ([], [C(1, 0), C(2, 1), C(3, 3)]),
            ([], []), ([], []), ([], []), ([], []),
        ]
    )
    t2t = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
    # Whole K-led run to empty col1
    assert any(m.src == 0 and m.dst == 1 and m.run_start == 0 for m in t2t)
    # Q♥ (run_start 1) is NOT a King -> cannot move to empty col1
    assert not any(m.src == 0 and m.dst == 1 and m.run_start == 1 for m in t2t)
    # 2♦ (red) can move onto 3♠? No — same column; 2♦ needs black 3: col2's own
    # 3♠ can't be a destination for a card in the same column. 3♠ (black) needs
    # a red 4 — none. So no other t2t involving col2.
    assert not any(m.src == 2 for m in t2t)


def test_partial_run_move_legality():
    # col0: 9♠ 8♥ 7♠ ; col1: 8♣ 7♦
    st = mk_state(
        [
            ([], [C(9, 3), C(8, 2), C(7, 3)]),
            ([], [C(8, 0), C(7, 1)]),
            ([], []), ([], []), ([], []), ([], []), ([], []),
        ]
    )
    t2t = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
    # 7♦ (red, col1 top) onto 8♥? illegal (same column + wrong rank).
    # 7♠ (black, col0 top) onto 8♥? illegal: 8 is not 7+1... wait 7♠ needs a red 8:
    # col1 top is 7♦ — not an 8. col0's own 8♥ can't be dst. -> no t2t at all?
    # 8♥7♠ suffix needs a black 9 — none. 9♠ needs a red 10 — none.
    # 8♣ needs a red 9 — none. So expect zero tableau moves here.
    assert t2t == []


def test_suffix_move_frees_card_below():
    # col0: 6♥ 5♣ ; col1: 6♠ — suffix [5♣]? no. Check 5♣ onto 6♠? same color no.
    # Use: col0: 7♦ 6♣ ; col1: 7♠ ; suffix [6♣] onto 7♦? same col. 6♣ needs red 7:
    # col1 top 7♠ is black -> illegal. Instead move whole [7♦6♣] onto col1's 7♠? no.
    # col0: 8♣ 7♦ ; col1: 9♦ — whole col0 run onto col1? 8♣ onto 9♦ red-> needs black9? 8 black onto 9 red: 9 is 8+1 ✓ and 9♦ red vs 8♣ black diff color ✓ legal.
    st = mk_state(
        [
            ([], [C(8, 0), C(7, 1)]),
            ([], [C(9, 1)]),
            ([], []), ([], []), ([], []), ([], []), ([], []),
        ]
    )
    t2t = [m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_TABLEAU]
    assert any(m.src == 0 and m.dst == 1 and m.run_start == 0 for m in t2t)
    # 7♦ alone onto 8♣? needs black 8 — 8♣ is black! Wait 7♦ onto 8♣ same column illegal anyway.
    assert not any(m.src == 0 and m.dst == 0 for m in t2t)


def test_draw_moves_cards_stock_to_waste():
    st = mk_state([([], [])] * 7, stock=[C(5, 0), C(6, 1), C(7, 2)])
    # stock[-1]=7♥ is the top card
    draw = [m for m in legal_moves(st) if m.kind == MoveKind.DRAW]
    assert len(draw) == 1
    apply_move(st, draw[0])
    assert len(st.stock) == 2
    assert st.waste == [C(7, 2)]


def test_draw3_moves_three():
    st = mk_state(
        [([], [])] * 7,
        stock=[C(1, 0), C(2, 1), C(3, 2), C(4, 3)],
        draw_size=3,
    )
    draw = next(m for m in legal_moves(st) if m.kind == MoveKind.DRAW)
    apply_move(st, draw)
    assert st.waste == [C(4, 3), C(3, 2), C(2, 1)]  # top = last
    assert st.stock == [C(1, 0)]


def test_recycle_restores_stock_order():
    cards = [C(i % 13 + 1, i // 13) for i in range(24)]  # 24 distinct cards
    st = mk_state([([], [])] * 7, stock=list(cards), draw_size=1)
    original_order = list(st.stock)
    while st.stock:
        apply_move(st, Move(MoveKind.DRAW))
    assert st.waste  # all drawn
    apply_move(st, Move(MoveKind.DRAW))  # recycle
    assert st.stock == original_order
    assert st.redeals == 1


def test_no_redeal_when_stock_nonempty():
    st = mk_state([([], [])] * 7, stock=[C(5, 0)], waste=[C(6, 1)])
    apply_move(st, Move(MoveKind.DRAW))
    assert st.waste == [C(6, 1), C(5, 0)]
    assert st.stock == []


def test_max_redeals_blocks_recycle():
    st = mk_state([([], [])] * 7, stock=[], waste=[C(6, 1)], max_redeals=0)
    draws = [m for m in legal_moves(st) if m.kind == MoveKind.DRAW]
    assert draws == []


def test_win_detection():
    found = [[C(r, s) for r in range(1, 14)] for s in range(4)]
    st = mk_state([([], [])] * 7, foundations=found)
    assert st.is_win()


def test_facedown_flip_on_empty_up():
    st = mk_state(
        [
            ([C(10, 0)], [C(1, 1)]),  # down card + A♦ up
            ([], []), ([], []), ([], []), ([], []), ([], []), ([], []),
        ]
    )
    m = next(m for m in legal_moves(st) if m.kind == MoveKind.TABLEAU_TO_FOUNDATION)
    out = apply_move(st, m)
    assert out.revealed
    assert st.tableau[0].up == [C(10, 0)]
    assert st.tableau[0].down == []
