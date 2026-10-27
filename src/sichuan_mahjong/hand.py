"""Hand analysis for Sichuan Mahjong: shanten, win detection, wait tiles.

Terminology
-----------
shanten  : number of tile draws still needed to reach a winning hand.
           ``-1`` means the hand already wins, ``0`` means tenpai (听牌).
melds    : exposed sets (碰 / 杠). A kong counts as one meld even though it
           is made of four tiles.

Sichuan specifics handled here
------------------------------
* ``缺一门``: a player may only win after discarding every tile of the suit
  they declared missing, so no winning hand may contain that suit.
* ``龙七对``: a seven-pairs hand where one pair is a quad is still a win, so a
  quad contributes two pairs when counting seven-pairs shanten.
* No honor tiles, therefore 27 kinds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .tiles import COPIES, NUM_KINDS, suit_of, validate_counts

# ---------------------------------------------------------------------------
# standard form (4 sets + 1 pair)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1 << 18)
def _dfs(counts: tuple[int, ...], i: int, m: int, t: int, eyes: int) -> int:
    """Best remaining shanten by decomposing ``counts`` from index ``i``.

    ``m`` completed melds, ``t`` partial sets, ``eyes`` whether a pair has
    been reserved as the eyes.  Returns the smallest achievable shanten.
    """
    counts_l = list(counts)
    while i < NUM_KINDS and counts_l[i] == 0:
        i += 1
    if i >= NUM_KINDS:
        return 8 - 2 * m - min(t, 4 - m) - eyes

    best = 99

    def consider(nm: int, nt: int, ne: int) -> None:
        nonlocal best
        value = _dfs(tuple(counts_l), i, nm, nt, ne)
        if value < best:
            best = value

    # complete melds
    if m < 4:
        if counts_l[i] >= 3:  # triplet
            counts_l[i] -= 3
            consider(m + 1, t, eyes)
            counts_l[i] += 3
        if i % 9 <= 6 and counts_l[i + 1] and counts_l[i + 2]:  # run
            counts_l[i] -= 1
            counts_l[i + 1] -= 1
            counts_l[i + 2] -= 1
            consider(m + 1, t, eyes)
            counts_l[i] += 1
            counts_l[i + 1] += 1
            counts_l[i + 2] += 1

    # eyes (only a single pair may serve as eyes)
    if not eyes and counts_l[i] >= 2:
        counts_l[i] -= 2
        consider(m, t, 1)
        counts_l[i] += 2

    # partial sets
    if m + t < 4:
        if counts_l[i] >= 2:  # pair used as a partial
            counts_l[i] -= 2
            consider(m, t + 1, eyes)
            counts_l[i] += 2
        if i % 9 <= 7 and counts_l[i + 1]:  # two-sided / edge
            counts_l[i] -= 1
            counts_l[i + 1] -= 1
            consider(m, t + 1, eyes)
            counts_l[i] += 1
            counts_l[i + 1] += 1
        if i % 9 <= 6 and counts_l[i + 2]:  # gapped
            counts_l[i] -= 1
            counts_l[i + 2] -= 1
            consider(m, t + 1, eyes)
            counts_l[i] += 1
            counts_l[i + 2] += 1

    # leave this tile isolated (it will be discarded)
    counts_l[i] -= 1
    consider(m, t, eyes)
    counts_l[i] += 1

    return best


def shanten_standard(counts: Sequence[int], exposed_melds: int = 0) -> int:
    """Shanten using the 4-sets-plus-pair form."""
    return _dfs(tuple(counts), 0, exposed_melds, 0, 0)


def shanten_seven_pairs(counts: Sequence[int]) -> int:
    """Shanten for 七对 / 龙七对 (only valid with a fully concealed hand)."""
    pairs = sum(c // 2 for c in counts)
    if pairs > 7:
        pairs = 7
    return 6 - pairs


def shanten(counts: Sequence[int], exposed_melds: int = 0) -> int:
    """Minimum shanten over all legal hand shapes.

    ``exposed_melds`` counts 碰/杠 already on the table.  Seven pairs is only
    considered when nothing is exposed.
    """
    validate_counts(counts)
    best = shanten_standard(counts, exposed_melds)
    if exposed_melds == 0:
        best = min(best, shanten_seven_pairs(counts))
    return best


# ---------------------------------------------------------------------------
# win detection
# ---------------------------------------------------------------------------


def _all_melds(counts: list[int], needed: int) -> bool:
    if needed == 0:
        return all(c == 0 for c in counts)
    i = 0
    while i < NUM_KINDS and counts[i] == 0:
        i += 1
    if i >= NUM_KINDS:
        return False
    if counts[i] >= 3:
        counts[i] -= 3
        if _all_melds(counts, needed - 1):
            counts[i] += 3
            return True
        counts[i] += 3
    if i % 9 <= 6 and counts[i + 1] and counts[i + 2]:
        counts[i] -= 1
        counts[i + 1] -= 1
        counts[i + 2] -= 1
        if _all_melds(counts, needed - 1):
            counts[i] += 1
            counts[i + 1] += 1
            counts[i + 2] += 1
            return True
        counts[i] += 1
        counts[i + 1] += 1
        counts[i + 2] += 1
    return False


def is_seven_pairs(counts: Sequence[int]) -> bool:
    return sum(counts) == 14 and sum(c // 2 for c in counts) == 7


def is_standard_win(counts: Sequence[int], exposed_melds: int = 0) -> bool:
    """True if the concealed tiles form (4 - exposed) sets plus one pair."""
    counts_l = list(counts)
    needed = 4 - exposed_melds
    if needed < 0:
        return False
    if sum(counts_l) != needed * 3 + 2:
        return False
    for i in range(NUM_KINDS):
        if counts_l[i] >= 2:
            counts_l[i] -= 2
            if _all_melds(counts_l, needed):
                counts_l[i] += 2
                return True
            counts_l[i] += 2
    return False


def is_winning(
    counts: Sequence[int],
    exposed_melds: int = 0,
    missing_suit: int | None = None,
) -> bool:
    """Full win test, including the Sichuan ``缺一门`` constraint."""
    if missing_suit is not None and any(
        counts[i] for i in range(missing_suit * 9, missing_suit * 9 + 9)
    ):
        return False
    if exposed_melds == 0 and is_seven_pairs(counts):
        return True
    return is_standard_win(counts, exposed_melds)


# ---------------------------------------------------------------------------
# waits
# ---------------------------------------------------------------------------


def winning_tiles(
    counts: Sequence[int],
    exposed_melds: int = 0,
    missing_suit: int | None = None,
    seen: Sequence[int] | None = None,
) -> list[int]:
    """Tiles that would complete the hand.

    ``seen`` is the count vector of every visible tile (all discards, exposed
    melds and the player's own hand).  When supplied, a kind with all four
    copies already visible is not reported as a wait.
    """
    counts_l = list(counts)
    out: list[int] = []
    for t in range(NUM_KINDS):
        if counts_l[t] >= COPIES:
            continue
        if seen is not None and seen[t] >= COPIES:
            continue
        if missing_suit is not None and suit_of(t) == missing_suit:
            continue
        counts_l[t] += 1
        if is_winning(counts_l, exposed_melds, missing_suit):
            out.append(t)
        counts_l[t] -= 1
    return out


def shanten_after_discard(
    counts: Sequence[int], exposed_melds: int = 0, discard: int | None = None
) -> int:
    """Shanten of a 14-tile hand after discarding ``discard``.

    ``discard=None`` returns the best value over every possible discard.
    """
    counts_l = list(counts)
    if discard is not None:
        if counts_l[discard] <= 0:
            raise ValueError(f"cannot discard {discard}: not in hand")
        counts_l[discard] -= 1
        return shanten(counts_l, exposed_melds)
    best = 99
    for t in range(NUM_KINDS):
        if counts_l[t]:
            counts_l[t] -= 1
            value = shanten(counts_l, exposed_melds)
            counts_l[t] += 1
            if value < best:
                best = value
    return best


def best_discards(counts: Sequence[int], exposed_melds: int = 0) -> list[int]:
    """Discards that minimise shanten (useful for a rule-based baseline)."""
    counts_l = list(counts)
    best = 99
    result: list[int] = []
    for t in range(NUM_KINDS):
        if not counts_l[t]:
            continue
        counts_l[t] -= 1
        value = shanten(counts_l, exposed_melds)
        counts_l[t] += 1
        if value < best:
            best = value
            result = [t]
        elif value == best:
            result.append(t)
    return result


def ukeire(
    counts: Sequence[int],
    exposed_melds: int = 0,
    seen: Sequence[int] | None = None,
    missing_suit: int | None = None,
) -> dict[int, int]:
    """Effective draw tiles of a 13-tile hand and how many copies remain.

    A tile is effective when drawing it either wins outright or, after
    discarding the worst tile, improves the hand's shanten.  ``seen`` is the
    count vector of all visible tiles; without it only the player's own hand
    is considered.
    """
    base = shanten(counts, exposed_melds)
    counts_l = list(counts)
    out: dict[int, int] = {}
    for t in range(NUM_KINDS):
        if counts_l[t] >= COPIES:
            continue
        if missing_suit is not None and suit_of(t) == missing_suit:
            continue
        left = COPIES - (seen[t] if seen is not None else counts_l[t])
        if left <= 0:
            continue
        counts_l[t] += 1
        if is_winning(counts_l, exposed_melds, missing_suit):
            out[t] = left
        elif shanten_after_discard(counts_l, exposed_melds) < base:
            out[t] = left
        counts_l[t] -= 1
    return out
