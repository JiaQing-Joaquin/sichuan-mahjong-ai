"""Fan (番) calculation for Sichuan Mahjong "血战到底".

Regional rule sets differ a lot, so every fan value lives in :class:`FanConfig`
and the default table below is only a reference implementation.  A commercial
product should calibrate these numbers against the exact rule set of the
target platform before going live.

The public entry point is :func:`score_win`, which takes a fully validated
winning hand and returns a :class:`ScoreResult` with an itemised fan list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .hand import is_seven_pairs, is_winning
from .tiles import suit_of


class MeldKind(str, Enum):
    PENG = "peng"  # 碰, 3 tiles
    GANG = "gang"  # 杠, 4 tiles


@dataclass(frozen=True)
class Meld:
    kind: MeldKind
    tile: int

    @property
    def size(self) -> int:
        return 4 if self.kind is MeldKind.GANG else 3

    @property
    def is_triplet(self) -> bool:
        # both a peng and a kong are triplet-shaped sets
        return True


#: Reference fan table (成都/血战到底 common convention).  Override per product.
DEFAULT_FAN: dict[str, int] = {
    "平胡": 0,
    "对对胡": 1,
    "清对": 3,
    "将对": 3,
    "七对": 2,
    "龙七对": 3,
    "清七对": 4,
    "清龙七对": 5,
    "清一色": 2,
    "金钩钓": 1,
    # 门清 is optional: many 血战到底 rule sets do not award it.  Enable by
    # setting a non-zero value in a custom FanConfig.
    "门清": 0,
    "自摸": 0,
    "根": 1,
    "杠上花": 1,
    "杠上炮": 1,
    "抢杠胡": 1,
    "海底捞月": 1,
    "天胡": 6,
    "地胡": 4,
}


@dataclass(frozen=True)
class FanConfig:
    fan: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FAN))
    base_score: int = 1
    cap_fan: int | None = None

    def get(self, name: str) -> int:
        return self.fan.get(name, 0)


@dataclass
class WinContext:
    """Everything needed to score a validated winning hand."""

    counts: list[int]  # concealed tiles, including the winning tile
    melds: list[Meld]
    win_tile: int
    self_draw: bool = False
    flags: set[str] = field(default_factory=set)  # e.g. {"gangshang_hua"}
    missing_suit: int | None = None
    is_dealer: bool = False
    first_turn: bool = False  # 天胡/地胡 detection is handled by the caller


@dataclass
class ScoreResult:
    fan: int
    items: list[str]
    points: int

    def __str__(self) -> str:
        return f"{'+'.join(self.items) or '平胡'} = {self.fan}番 = {self.points}分"


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------


def _concealed_counts(ctx: WinContext) -> list[int]:
    return list(ctx.counts)


def _all_tiles(counts: Sequence[int], melds: Iterable[Meld]) -> list[int]:
    tiles = [t for t, c in enumerate(counts) for _ in range(c)]
    for m in melds:
        tiles.extend([m.tile] * m.size)
    return tiles


def _one_suit(counts: Sequence[int], melds: Iterable[Meld]) -> bool:
    suits = {suit_of(t) for t in _all_tiles(counts, melds)}
    return len(suits) == 1


def _is_all_triplets(counts: Sequence[int]) -> bool:
    """Concealed part is triplets plus one pair (no runs)."""
    pair_seen = False
    for c in counts:
        if c == 0:
            continue
        if c == 3:
            continue
        if c == 2 and not pair_seen:
            pair_seen = True
            continue
        return False
    return pair_seen


def _all_258(counts: Sequence[int], melds: Iterable[Meld]) -> bool:
    ranks = {t % 9 + 1 for t in _all_tiles(counts, melds)}
    return ranks <= {2, 5, 8}


def count_gen(counts: Sequence[int], melds: Iterable[Meld]) -> int:
    """Number of 根: one per kong, plus one for each concealed quad (龙七对)."""
    gen = sum(1 for m in melds if m.kind is MeldKind.GANG)
    gen += sum(1 for c in counts if c >= 4)
    return gen


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_win(ctx: WinContext, config: FanConfig | None = None) -> ScoreResult:
    cfg = config or FanConfig()
    counts = _concealed_counts(ctx)
    melds = list(ctx.melds)
    exposed_sets = len(melds)

    if not is_winning(counts, exposed_melds=exposed_sets, missing_suit=ctx.missing_suit):
        raise ValueError("score_win called on a non-winning hand")

    items: list[str] = []

    seven = exposed_sets == 0 and is_seven_pairs(counts)
    quad = any(c >= 4 for c in counts)
    triplets = (not seven) and _is_all_triplets(counts) and all(m.is_triplet for m in melds)
    one_suit = _one_suit(counts, melds)

    # -- shape fan (mutually exclusive) -----------------------------------
    if seven:
        shape = "龙七对" if quad else "七对"
        if one_suit:
            shape = "清龙七对" if quad else "清七对"
    elif triplets:
        if one_suit:
            shape = "清对"
        elif _all_258(counts, melds):
            shape = "将对"
        else:
            shape = "对对胡"
    else:
        shape = "清一色" if one_suit else "平胡"
    items.append(shape)

    total = cfg.get(shape)

    # -- all-exposed / concealed bonuses ----------------------------------
    if len(melds) == 4 and cfg.get("金钩钓"):
        items.append("金钩钓")
        total += cfg.get("金钩钓")
    elif exposed_sets == 0 and not seven and cfg.get("门清"):
        items.append("门清")
        total += cfg.get("门清")

    # -- 根 ----------------------------------------------------------------
    gen = count_gen(counts, melds)
    if gen:
        items.append(f"根x{gen}")
        total += cfg.get("根") * gen

    # -- situational flags -------------------------------------------------
    if ctx.self_draw and cfg.get("自摸"):
        items.append("自摸")
        total += cfg.get("自摸")

    flag_names = {
        "gangshang_hua": "杠上花",
        "gangshang_pao": "杠上炮",
        "qiang_gang": "抢杠胡",
        "haidi": "海底捞月",
        "tian_hu": "天胡",
        "di_hu": "地胡",
    }
    for key, name in flag_names.items():
        if key in ctx.flags and cfg.get(name):
            items.append(name)
            total += cfg.get(name)

    if cfg.cap_fan is not None:
        total = min(total, cfg.cap_fan)

    points = cfg.base_score * (2 ** total)
    return ScoreResult(fan=total, items=items, points=points)
