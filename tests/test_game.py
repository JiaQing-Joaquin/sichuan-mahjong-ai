"""Tests for the 血战到底 game engine.

Run with::

    python tests/test_game.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.game import (  # noqa: E402
    ActKind,
    Action,
    Game,
    GameConfig,
    Phase,
)
from sichuan_mahjong.scoring import Meld, MeldKind  # noqa: E402
from sichuan_mahjong.tiles import NUM_KINDS, counts_from_tiles, make_tile, parse_tiles  # noqa: E402

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


FILLER = "123m 456m 789m 123p 4p"


def rig(
    hands: dict[int, str],
    missing: dict[int, int],
    turn: int = 0,
    wall: list[int] | None = None,
) -> Game:
    g = Game(seed=1)
    for p in g.players:
        p.hand = [0] * NUM_KINDS
        p.melds = []
        p.discards = []
        p.won = False
        p.score = 0
        p.just_gang_draw = False
        p.win_item = p.win_fan = p.win_points = None
    for seat, text in hands.items():
        g.players[seat].hand = C(text)
    for seat, suit in missing.items():
        g.players[seat].missing_suit = suit
    g.phase = Phase.TURN
    g.turn = turn
    g._reaction = None
    g._must_discard_only = False
    g._discard_count = 1
    g.wall = list(wall) if wall is not None else [18, 19, 20, 21, 22, 23, 24, 25]
    return g


# ---------------------------------------------------------------------------
# individual mechanics
# ---------------------------------------------------------------------------


def test_self_draw() -> None:
    g = rig(
        {
            0: "123m 456m 789m 123p 55p",
            1: FILLER,
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
    )
    acts = g.legal_actions()
    check("self-draw offers HU_SELF", any(a.kind is ActKind.HU_SELF for a in acts), True)
    g.step(Action(ActKind.HU_SELF))
    check("winner marked", g.players[0].won, True)
    check("winner scored positive", g.players[0].score > 0, True)
    check("zero-sum after self-draw", sum(g.scores()), 0)
    check("play continues with 3 left", g.is_over, False)


def test_discard_hu() -> None:
    g = rig(
        {
            0: "123m 456m 789m 123p 5p",
            1: "123m 456m 789m 123p 5p",
            2: "1111m 2222m 3333m 4m",
            3: "1111m 2222m 3333m 4m",
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
    )
    g.step(Action(ActKind.DISCARD, tile=13))  # discard 5p
    check("reaction phase starts", g.phase, Phase.REACTION)
    check("next responder is seat 1", g.current_player, 1)
    acts = g.legal_actions()
    check("can hu on discard", any(a.kind is ActKind.HU_DISCARD for a in acts), True)
    g.step(Action(ActKind.HU_DISCARD))
    check("responder moves to seat 2", g.current_player, 2)
    g.step(Action(ActKind.PASS))
    g.step(Action(ActKind.PASS))
    check("discard-hu winner", g.players[1].won, True)
    check("discarder paid", g.players[0].score < 0, True)
    check("zero-sum after discard-hu", sum(g.scores()), 0)


def test_peng() -> None:
    g = rig(
        {
            0: "123m 456m 789m 123p 5p",
            1: "5p 5p 123m 456m 789m 1m 1m",
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
    )
    g.step(Action(ActKind.DISCARD, tile=13))  # 5p
    check("seat 1 may peng", any(a.kind is ActKind.PENG for a in g.legal_actions()), True)
    g.step(Action(ActKind.PENG, tile=13))
    g.step(Action(ActKind.PASS))
    g.step(Action(ActKind.PASS))
    melds = g.players[1].melds
    check("peng creates a meld", len(melds), 1)
    check("peng tile", melds[0].tile, 13)
    check("peng player must discard next", (g.phase, g.turn), (Phase.TURN, 1))


def test_minggang() -> None:
    g = rig(
        {
            0: "123m 456m 789m 123p 5p",
            1: "5p 5p 5p 123m 456m 789m 1m",
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
        wall=[make_tile(0, 9), make_tile(0, 8), make_tile(0, 7)],
    )
    g.step(Action(ActKind.DISCARD, tile=13))
    check("seat 1 may minggang", any(a.kind is ActKind.MINGGANG for a in g.legal_actions()), True)
    g.step(Action(ActKind.MINGGANG, tile=13))
    g.step(Action(ActKind.PASS))
    g.step(Action(ActKind.PASS))
    melds = g.players[1].melds
    check("minggang creates a kong", (len(melds), melds[0].kind.value), (1, "gang"))
    check("konger draws and acts", (g.phase, g.turn), (Phase.TURN, 1))
    check("wall shrank by replacement", len(g.wall), 2)


def test_missing_restriction() -> None:
    # 缺万: only 万 tiles may be discarded while any remain
    g = rig(
        {
            0: "123m 456p 789p 123s 4s",
            1: FILLER,
            2: FILLER,
            3: FILLER,
        },
        {0: 0, 1: 2, 2: 2, 3: 2},
    )
    discards = [a for a in g.legal_actions() if a.kind is ActKind.DISCARD]
    check("only missing-suit discards", all(a.tile // 9 == 0 for a in discards), True)
    check("all three 万 available", len(discards), 3)  # 1万,2万,3万


# ---------------------------------------------------------------------------
# full games
# ---------------------------------------------------------------------------


def test_random_games() -> None:
    for seed in range(120):
        g = Game(seed=seed)
        steps = 0
        while not g.is_over:
            acts = g.legal_actions()
            if not acts:
                break
            g.step(g.rng.choice(acts))
            steps += 1
            if steps > 5000:
                raise AssertionError(f"seed {seed} did not terminate")
        check(f"random game {seed} over", g.is_over, True)
        check(f"random game {seed} zero-sum", sum(g.scores()), 0)


def test_baseline_games() -> None:
    for seed in range(40):
        g = Game(seed=seed, config=GameConfig(gang_payment={"ming": 0, "an": 0, "bu": 0}))
        ai = BaselineAI(seed=seed)
        steps = 0
        while not g.is_over:
            g.step(ai.act(g))
            steps += 1
            if steps > 5000:
                raise AssertionError(f"baseline seed {seed} did not terminate")
        check(f"baseline game {seed} over", g.is_over, True)
        check(f"baseline game {seed} zero-sum", sum(g.scores()), 0)


def test_qiang_gang() -> None:
    """A player robs another player's added kong (抢杠胡)."""
    g = rig(
        {
            0: "1m2m3m4m5m6m7m8m9m 1p 5p",  # holds the 4th 5p
            1: "123m 456m 789m 123p 5p",  # waiting on 5p for the pair
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
        wall=[make_tile(0, 9), make_tile(0, 8), make_tile(0, 7), make_tile(0, 6)],
    )
    g.players[0].melds = [Meld(MeldKind.PENG, 13)]  # exposed 5p peng
    acts = g.legal_actions()
    check(
        "bukang offered",
        any(a.kind is ActKind.BUKANG and a.tile == 13 for a in acts),
        True,
    )
    g.step(Action(ActKind.BUKANG, tile=13))
    check("qiang-gang opens a reaction", g.phase, Phase.REACTION)
    check("kong not applied yet", g.players[0].melds[0].kind, MeldKind.PENG)
    check(
        "robber may hu",
        any(a.kind is ActKind.HU_DISCARD for a in g.legal_actions()),
        True,
    )
    g.step(Action(ActKind.HU_DISCARD))
    g.step(Action(ActKind.PASS))
    g.step(Action(ActKind.PASS))
    check("robber wins", g.players[1].won, True)
    check("qiang-gang recorded", "抢杠胡" in (g.players[1].win_item or ""), True)
    check("fourth tile taken from konger", g.players[0].hand[13], 0)
    check("peng preserved", g.players[0].melds[0].kind, MeldKind.PENG)


def test_bukang_without_robber() -> None:
    """If nobody can rob, the added kong goes through normally."""
    g = rig(
        {
            0: "1m2m3m4m5m6m7m8m9m 1p 5p",
            1: FILLER,  # waits on 4p, not 5p
            2: FILLER,
            3: FILLER,
        },
        {0: 2, 1: 2, 2: 2, 3: 2},
        wall=[make_tile(0, 9), make_tile(0, 8), make_tile(0, 7)],
    )
    g.players[0].melds = [Meld(MeldKind.PENG, 13)]
    before_wall = len(g.wall)
    g.step(Action(ActKind.BUKANG, tile=13))
    check("kong applied", g.players[0].melds[0].kind, MeldKind.GANG)
    check("fourth tile consumed", g.players[0].hand[13], 0)
    check("konger draws replacement", len(g.wall), before_wall - 1)
    check("konger acts again", (g.phase, g.turn), (Phase.TURN, 0))


def test_swap_three() -> None:
    """Optional opening phase: everyone passes three same-suit tiles."""
    g = Game(seed=5, config=GameConfig(swap_three=True))
    check("swap phase first", g.phase, Phase.SWAP)
    check(
        "108 tiles before swap",
        sum(sum(p.hand) for p in g.players) + len(g.wall),
        108,
    )

    before = [p.hand[:] for p in g.players]
    selections: dict[int, list[int]] = {}
    for _ in range(4):
        seat = g.current_player
        swaps = [a for a in g.legal_actions() if a.kind is ActKind.SWAP_THREE]
        check(f"seat {seat} has swap options", len(swaps) > 0, True)
        action = swaps[0]
        check(f"seat {seat} passes one suit", len({t // 9 for t in action.tiles}), 1)
        selections[seat] = list(action.tiles)
        g.step(action)

    check("swap -> missing", g.phase, Phase.MISSING)
    check(
        "108 tiles after swap",
        sum(sum(p.hand) for p in g.players) + len(g.wall),
        108,
    )

    direction = g.cfg.swap_direction % 4
    for seat in range(4):
        expected = before[seat][:]
        for t in selections[seat]:
            expected[t] -= 1
        for t in selections[(seat - direction) % 4]:
            expected[t] += 1
        check(f"seat {seat} hand after swap", g.players[seat].hand, expected)


def main() -> int:
    test_self_draw()
    test_discard_hu()
    test_peng()
    test_minggang()
    test_missing_restriction()
    test_qiang_gang()
    test_bukang_without_robber()
    test_swap_three()
    test_random_games()
    test_baseline_games()

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for f in _FAILS:
            print("  -", f)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
