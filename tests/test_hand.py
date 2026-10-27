"""Unit tests for tile encoding, shanten, win detection and waits.

Run with::

    python tests/test_hand.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sichuan_mahjong.hand import (  # noqa: E402
    best_discards,
    is_winning,
    shanten,
    shanten_after_discard,
    shanten_seven_pairs,
    shanten_standard,
    ukeire,
    winning_tiles,
)
from sichuan_mahjong.tiles import (  # noqa: E402
    counts_from_tiles,
    make_tile,
    parse_tiles,
    tile_to_str,
)

_FAILS: list[str] = []
_PASSES = 0


def check(name: str, got, want) -> None:
    global _PASSES
    if got == want:
        _PASSES += 1
    else:
        message = f"{name}: got {got!r}, want {want!r}"
        _FAILS.append(message)
        # fail loudly if a pytest-based runner (e.g. PyCharm's "Run Tests") is used
        if "pytest" in sys.modules:
            raise AssertionError(message)


def C(text: str):
    return counts_from_tiles(parse_tiles(text))


# ---------------------------------------------------------------------------
# tiles
# ---------------------------------------------------------------------------


def test_tiles() -> None:
    check("parse 1m", parse_tiles("1m"), [0])
    check("parse 9s", parse_tiles("9s"), [26])
    check("tile_to_str 1m", tile_to_str(0), "1万")
    check("tile_to_str 9s", tile_to_str(26), "9条")
    check("make_tile", make_tile(1, 5), 13)
    # 万=0..8, 筒=9..17, 条=18..26
    check("parse mixed", parse_tiles("123m 456p 789s"),
          [0, 1, 2, 12, 13, 14, 24, 25, 26])
    check("parse compact", parse_tiles("123m456p"), [0, 1, 2, 12, 13, 14])


# ---------------------------------------------------------------------------
# winning hands
# ---------------------------------------------------------------------------


def test_wins() -> None:
    win = C("123m 456m 789m 123p 55p")
    check("standard win", is_winning(win), True)
    check("standard win shanten", shanten(win), -1)
    check("standard win shanten_standard", shanten_standard(win), -1)

    check("seven pairs win", is_winning(C("11m 22m 33m 44m 55m 66m 77m")), True)
    check("seven pairs shanten", shanten(C("11m 22m 33m 44m 55m 66m 77m")), -1)

    # 龙七对: a quad counts as two pairs
    check("long seven pairs", is_winning(C("1111m 22m 33m 44m 55m 66m")), True)
    check("long seven pairs shanten", shanten(C("1111m 22m 33m 44m 55m 66m")), -1)

    check("not a win", is_winning(C("123m 456m 789m 123p 56p")), False)
    check("not a win shanten", shanten(C("123m 456m 789m 123p 56p")) >= 0, True)


def test_exposed_melds() -> None:
    # one exposed peng: 3 concealed sets + pair = 11 tiles
    hand = C("123m 456m 789m 55p")
    check("win with 1 exposed meld", is_winning(hand, exposed_melds=1), True)
    check("win with 1 exposed meld shanten", shanten(hand, exposed_melds=1), -1)

    # three sets + a single tile: waiting for the pair
    tenpai = C("123m 456m 789m 5p")
    check("tenpai with 1 exposed meld", shanten(tenpai, exposed_melds=1), 0)


def test_missing_suit() -> None:
    # 缺的是条 -> the hand below contains no 条 and therefore may win
    win = C("123m 456m 789m 123p 55p")
    check("缺条 wins", is_winning(win, missing_suit=2), True)
    # 缺的是筒 but the hand contains 筒 -> cannot win
    check("缺筒 cannot win", is_winning(win, missing_suit=1), False)
    # 缺的是万 but the hand contains 万 -> cannot win
    check("缺万 cannot win", is_winning(win, missing_suit=0), False)


# ---------------------------------------------------------------------------
# shanten
# ---------------------------------------------------------------------------


def test_shanten() -> None:
    tenpai = C("123m 456m 789m 123p 5p")
    check("tenpai shanten", shanten(tenpai), 0)
    check("tenpai shanten_standard", shanten_standard(tenpai), 0)

    one_away = C("123m 456m 789m 12p 4p 7s")
    check("one-away shanten", shanten(one_away), 1)

    # 1111m is a kong-like block; still usable as a meld + a lone tile
    check("quad hand shanten", shanten(C("1111m 234p 567p 89s 9s")), 1)


def test_waits() -> None:
    tenpai = C("123m 456m 789m 123p 5p")
    check("tenpai waits", winning_tiles(tenpai), [13])  # 5p

    # waiting on 3p / 6p with a 45p shape
    hand = C("123m 456m 789m 45p 77p")
    waits = winning_tiles(hand)
    check("two-sided waits", sorted(waits), [11, 14])  # 3p=11, 6p=14


def test_discards() -> None:
    # 14 tiles, discard 7s (index 24) to reach tenpai
    hand = C("123m 456m 789m 123p 5p 7s")
    check("best discard keeps tenpai", shanten_after_discard(hand), 0)
    check("7s is a best discard", 24 in best_discards(hand), True)


def test_ukeire() -> None:
    tenpai = C("123m 456m 789m 123p 5p")
    eff = ukeire(tenpai)
    check("tenpai has effective tiles", 13 in eff, True)
    # 5p: 4 total, 1 already in hand -> 3 remaining
    check("ukeire counts remaining copies", eff.get(13), 3)


def main() -> int:
    test_tiles()
    test_wins()
    test_exposed_melds()
    test_missing_suit()
    test_shanten()
    test_waits()
    test_discards()
    test_ukeire()

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for f in _FAILS:
            print("  -", f)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
