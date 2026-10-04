"""Text rendering for Klondike, plus PIL frame rendering for recordings."""

from __future__ import annotations

from solitaire_rl.cards import SUIT_SYMBOLS, Card
from solitaire_rl.state import GameState

FACE_DOWN = "▒▒"
EMPTY = "  "
RED_SUITS = {1, 2}


def _fmt(card: Card | None) -> str:
    return str(card) if card is not None else EMPTY


def render_text(state: GameState, footer: str = "") -> str:
    """Render the board as aligned monospace text."""
    lines: list[str] = []

    # Stock / waste / foundations row.
    stock_str = f"[{len(state.stock):2d}]" if state.stock else "[--]"
    waste_top = _fmt(state.waste[-1] if state.waste else None)
    waste_str = f"{waste_top}({len(state.waste):2d})"
    fdn = " ".join(
        f"{_fmt(f[-1] if f else None)}{SUIT_SYMBOLS[s]}" for s, f in enumerate(state.foundations)
    )
    lines.append(f"  STK {stock_str}  WST {waste_str}        FDN {fdn}")
    lines.append("")

    # Tableau: print row by row over the tallest column.
    height = max(len(c.down) + len(c.up) for c in state.tableau)
    grid: list[list[str]] = []
    for col in state.tableau:
        cells = [FACE_DOWN] * len(col.down) + [str(c) for c in col.up]
        grid.append(cells)
    header = "   " + "   ".join(f" {i} " for i in range(7))
    lines.append(header)
    for row in range(height):
        cells = (grid[c][row] if row < len(grid[c]) else "  " for c in range(7))
        lines.append("   " + "   ".join(f"{cell:2s}" for cell in cells))
    lines.append(f"   downs: {' '.join(str(len(c.down)) for c in state.tableau)}")

    meta = f"moves={state.moves} redeals={state.redeals} draw={state.draw_size}"
    if footer:
        meta += f" | {footer}"
    lines.append(meta)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# PIL frame rendering (for GIF/MP4 episode recordings)
# --------------------------------------------------------------------------

def _load_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size, index=0)
        except Exception:
            continue
    return ImageFont.load_default()


def render_frame(state: GameState, footer: str = "", cell_w: int = 46, cell_h: int = 30):
    """Render the board to a PIL Image (dark theme, red/black cards)."""
    from PIL import Image, ImageDraw

    left_pad, top_pad = 12, 10
    height = max(len(c.down) + len(c.up) for c in state.tableau)
    rows = 3 + height + 2
    w = left_pad * 2 + 7 * cell_w + 120
    h = top_pad * 2 + rows * cell_h
    img = Image.new("RGB", (w, h), (13, 17, 23))
    dr = ImageDraw.Draw(img)
    font = _load_font(17)
    font_small = _load_font(14)

    x0 = left_pad
    # Stock / waste / foundations.
    stock_txt = f"STK[{len(state.stock):2d}]" if state.stock else "STK[--]"
    dr.text((x0, top_pad), stock_txt, font=font_small, fill=(180, 180, 180))
    if state.waste:
        wc = state.waste[-1]
        color = (255, 120, 120) if wc.is_red else (230, 230, 230)
        dr.text(
            (x0 + 90, top_pad),
            f"WST {wc} ({len(state.waste):2d})",
            font=font_small,
            fill=color,
        )
    else:
        dr.text((x0 + 90, top_pad), f"WST -- ({0:2d})", font=font_small, fill=(120, 120, 120))
    fx = x0 + 260
    for s, f in enumerate(state.foundations):
        txt = str(f[-1]) if f else "--"
        color = (255, 120, 120) if s in RED_SUITS else (200, 200, 255)
        dr.text((fx, top_pad), f"{txt}{SUIT_SYMBOLS[s]}", font=font_small, fill=color)
        fx += 55

    # Tableau.
    ty = top_pad + 2 * cell_h
    for c, col in enumerate(state.tableau):
        x = x0 + c * cell_w
        cells = [None] * len(col.down) + list(col.up)
        for r, card in enumerate(cells):
            y = ty + r * cell_h
            if card is None:
                dr.rectangle(
                    [x, y, x + cell_w - 6, y + cell_h - 4], fill=(40, 46, 60)
                )
            else:
                color = (255, 110, 110) if card.is_red else (235, 235, 235)
                dr.text((x + 2, y + 4), str(card), font=font, fill=color)

    # Column face-down counts + footer.
    fy = ty + height * cell_h + 6
    dr.text(
        (x0, fy),
        "downs: " + " ".join(f"{len(c.down):2d}" for c in state.tableau),
        font=font_small,
        fill=(140, 140, 140),
    )
    meta = f"moves={state.moves} redeals={state.redeals} draw={state.draw_size}"
    if footer:
        meta += " | " + footer
    dr.text((x0, fy + cell_h), meta, font=font_small, fill=(160, 180, 200))
    return img
