"""Unit tests for fan scoring.

Run with::

    python tests/test_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sichuan_mahjong.scoring import (  # noqa: E402
    FanConfig,
    Meld,
    MeldKind,
    WinContext,
    count_gen,
    score_win,
)
from sichuan_mahjong.tiles import counts_from_tiles, parse_tiles  # noqa: E402

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


def C(text: str) -> list[int]:
    return counts_from_tiles(parse_tiles(text))


def peng(tile: int) -> Meld:
    return Meld(MeldKind.PENG, tile)


def gang(tile: int) -> Meld:
    return Meld(MeldKind.GANG, tile)


def ctx(text: str, melds=None, win_tile=0, missing_suit=None, **kw) -> WinContext:
    return WinContext(
        counts=C(text),
        melds=melds or [],
        win_tile=win_tile,
        missing_suit=missing_suit,
        **kw,
    )


def test_shapes() -> None:
    plain = ctx("123m 456m 789m 123p 55p", missing_suit=2)
    r = score_win(plain)
    check("平胡 fan", (r.items, r.fan, r.points), (["平胡"], 0, 1))

    flush = ctx("123m 456m 789m 123m 55m", missing_suit=2)
    r = score_win(flush)
    check("清一色 fan", (r.items[0], r.fan, r.points), ("清一色", 2, 4))

    triplets = ctx("111m 222m 333m 444p 55p", missing_suit=2)
    r = score_win(triplets)
    check("对对胡 fan", (r.items[0], r.fan, r.points), ("对对胡", 1, 2))

    pure_triplets = ctx("111m 222m 333m 444m 55m", missing_suit=2)
    r = score_win(pure_triplets)
    check("清对 fan", (r.items[0], r.fan, r.points), ("清对", 3, 8))

    jiang_dui = ctx("222m 555m 888m 222p 55p", missing_suit=2)
    r = score_win(jiang_dui)
    check("将对 fan", (r.items[0], r.fan, r.points), ("将对", 3, 8))

    seven = ctx("11m 22m 33p 44p 55s 66s 77m", missing_suit=None)
    r = score_win(seven)
    check("七对 fan", (r.items[0], r.fan, r.points), ("七对", 2, 4))

    long_seven = ctx("1111m 22m 33p 44p 55s 66s", missing_suit=None)
    r = score_win(long_seven)
    check("龙七对 fan", (r.items[0], r.fan, r.points), ("龙七对", 4, 16))
    check("龙七对 root counts quad", count_gen(C("1111m 22m 33p 44p 55s 66s"), []), 1)


def test_exposed() -> None:
    melds = [peng(0), peng(1), peng(2), peng(12)]
    r = score_win(ctx("55p", melds=melds, missing_suit=2))
    check("金钩钓 items", r.items, ["对对胡", "金钩钓"])
    check("金钩钓 fan", r.fan, 2)

    # a kong contributes one 根
    melds = [peng(0), peng(1), gang(2)]
    r = score_win(ctx("456p 77p", melds=melds, missing_suit=2))
    check("kong root fan", (r.items, r.fan), (["平胡", "根x1"], 1))
    check("count_gen kong", count_gen(C("456p 77p"), melds), 1)


def test_flags() -> None:
    hand = ctx("123m 456m 789m 123p 55p", missing_suit=2, flags={"gangshang_hua"})
    r = score_win(hand)
    check("杠上花 fan", r.fan, 1)
    check("杠上花 points", r.points, 2)

    hand = ctx("123m 456m 789m 123p 55p", missing_suit=2, flags={"tian_hu"})
    r = score_win(hand)
    check("天胡 fan", r.fan, 6)

    capped = FanConfig(cap_fan=2)
    r = score_win(ctx("111m 222m 333m 444m 55m", missing_suit=2), capped)
    check("cap fan applies", r.fan, 2)


def test_invalid() -> None:
    try:
        score_win(ctx("123m 456m 789m 123p 56p", missing_suit=2))
    except ValueError:
        check("non-winning hand rejected", True, True)
    else:
        check("non-winning hand rejected", False, True)

    try:
        # contains 筒 while 缺筒
        score_win(ctx("123m 456m 789m 123p 55p", missing_suit=1))
    except ValueError:
        check("missing-suit violation rejected", True, True)
    else:
        check("missing-suit violation rejected", False, True)


def main() -> int:
    test_shapes()
    test_exposed()
    test_flags()
    test_invalid()

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for f in _FAILS:
            print("  -", f)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
