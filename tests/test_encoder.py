"""Tests for the state encoder (v2).

Run with::

    python tests/test_encoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sichuan_mahjong.encoder import (  # noqa: E402
    CALL_DIM,
    STATE_DIM,
    call_mask,
    discard_mask,
    encode,
    encode_state,
    legal_discard_tiles,
)
from sichuan_mahjong.game import ActKind, Action, Game, Phase  # noqa: E402
from sichuan_mahjong.tiles import NUM_KINDS, counts_from_tiles, parse_tiles  # noqa: E402

_FAILS: list[str] = []
_PASSES = 0

# feature offsets
OFF_LAST_DRAW = 318
OFF_REACTION = 345
OFF_PENDING = 346
OFF_DISCARDER = 373
OFF_REMAINING = 377


def check(name: str, got, want) -> None:
    global _PASSES
    if got == want:
        _PASSES += 1
    else:
        message = f"{name}: got {got!r}, want {want!r}"
        _FAILS.append(message)
        if "pytest" in sys.modules:
            raise AssertionError(message)


def C(text: str):
    return counts_from_tiles(parse_tiles(text))


FILLER = "123m 456m 789m 123p 4p"


def rig(hands, missing, turn=0):
    g = Game(seed=7)
    for p in g.players:
        p.hand = [0] * NUM_KINDS
        p.melds = []
        p.discards = []
        p.won = False
        p.score = 0
        p.last_draw = None
    for seat, text in hands.items():
        g.players[seat].hand = C(text)
    for seat, suit in missing.items():
        g.players[seat].missing_suit = suit
    g.phase = Phase.TURN
    g.turn = turn
    g._reaction = None
    g._must_discard_only = False
    g.wall = [18, 19, 20, 21, 22, 23, 24, 25]
    return g


def test_dims() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    feats, dmask, cmask = encode(g, 0)
    check("feature length", len(feats), STATE_DIM)
    check("discard mask length", len(dmask), NUM_KINDS)
    check("call mask length", len(cmask), CALL_DIM)
    check("all floats", all(isinstance(x, float) for x in feats), True)


def test_mask_matches_legal() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    mask = discard_mask(g)
    engine_tiles = sorted(
        a.tile for a in g.legal_actions() if a.kind is ActKind.DISCARD
    )
    check("mask matches engine discards", legal_discard_tiles(mask), engine_tiles)
    check("mask size equals distinct hand kinds", sum(mask), 13)


def test_missing_restriction() -> None:
    g = rig({0: "123m 456p 789p 123s 4s", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 0, 1: 2, 2: 2, 3: 2})
    tiles = legal_discard_tiles(discard_mask(g))
    check("only missing-suit tiles masked", all(t // 9 == 0 for t in tiles), True)
    check("three 万 tiles", len(tiles), 3)


def test_no_opponent_hand_leakage() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    before, _, _ = encode(g, 0)

    g.players[1].hand = C("1111m 2222p 3333s 4m")
    g.players[2].hand = C("999m 888p 777s 6m")

    after, _, _ = encode(g, 0)
    check("opponent hands do not leak", before, after)


def test_last_draw() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    g.players[0].last_draw = 4  # 5m
    feats = encode_state(g, 0)
    check("last draw one-hot", feats[OFF_LAST_DRAW + 4], 1.0)
    check(
        "last draw sum",
        sum(feats[OFF_LAST_DRAW:OFF_LAST_DRAW + NUM_KINDS]),
        1.0,
    )


def test_reaction_encoding() -> None:
    g = rig(
        {
            0: "123m 456m 789m 123p 5p",
            1: "123m 456m 789m 123p 5p",
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
    )
    g.step(Action(ActKind.DISCARD, tile=13))  # discard 5p
    check("reaction phase", g.phase, Phase.REACTION)

    feats, dmask, cmask = encode(g, 1)
    check("reaction flag set", feats[OFF_REACTION], 1.0)
    check("pending tile one-hot", feats[OFF_PENDING + 13], 1.0)
    # discarder is seat 0, relative to seat 1 -> (0 - 1) % 4 = 3
    check("discarder relative seat", feats[OFF_DISCARDER + 3], 1.0)
    check("no discard actions while reacting", sum(dmask), 0)
    check("call mask allows hu", cmask[0], True)
    check("call mask allows pass", cmask[3], True)


def test_remaining_counts() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    feats = encode_state(g, 0)
    # 5p: one copy in own hand, three unseen
    check("remaining 5p", feats[OFF_REMAINING + 13], 3 / 4)
    # every remaining count must be within [0, 1]
    check(
        "remaining bounded",
        all(0.0 <= v <= 1.0 for v in feats[OFF_REMAINING:OFF_REMAINING + NUM_KINDS]),
        True,
    )


def test_deterministic() -> None:
    g = rig({0: "123m 456m 789m 123p 5p", 1: FILLER, 2: FILLER, 3: FILLER},
            {0: 2, 1: 2, 2: 2, 3: 2})
    check("encoder deterministic", encode(g, 0), encode(g, 0))


def main() -> int:
    test_dims()
    test_mask_matches_legal()
    test_missing_restriction()
    test_no_opponent_hand_leakage()
    test_last_draw()
    test_reaction_encoding()
    test_remaining_counts()
    test_deterministic()

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for failure in _FAILS:
            print("  -", failure)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
