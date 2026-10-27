"""Tile encoding and utilities for Sichuan Mahjong (血战到底).

The game has 108 tiles: three suits, ranks 1-9, four copies of each.
There are no honor tiles (no 风牌/字牌), hence 27 distinct tile kinds.

Encoded tile ids run 0..26::

    0..8   : 万 (characters) 1..9
    9..17  : 筒 (dots) 1..9
    18..26 : 条 (bamboo) 1..9

A "hand" is normally represented as a length-27 count vector.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

NUM_KINDS = 27
COPIES = 4

SUIT_NAMES = ("万", "筒", "条")
# m = 万 (wan), p = 筒 (pin/tong), s = 条 (sou/tiao) -- riichi-like notation
SUIT_KEYS = ("m", "p", "s")


def suit_of(tile: int) -> int:
    """0=万, 1=筒, 2=条."""
    return tile // 9


def rank_of(tile: int) -> int:
    """1..9"""
    return tile % 9 + 1


def make_tile(suit: int, rank: int) -> int:
    if not (0 <= suit < 3):
        raise ValueError(f"suit out of range: {suit}")
    if not (1 <= rank <= 9):
        raise ValueError(f"rank out of range: {rank}")
    return suit * 9 + (rank - 1)


def tile_to_str(tile: int) -> str:
    return f"{rank_of(tile)}{SUIT_NAMES[suit_of(tile)]}"


def tiles_to_str(tiles: Iterable[int]) -> str:
    return " ".join(tile_to_str(t) for t in sorted(tiles))


_TOKEN_RE = re.compile(r"(\d+)\s*([mps万筒条])")


def parse_tiles(text: str) -> list[int]:
    """Parse a compact tile string into a list of tile ids.

    Accepted forms (all equivalent)::

        "123m 456p 789s"
        "1m2m3m4p5p6p7s8s9s"
        "123万456筒789条"
        "1万 2万 3万"

    The result is sorted and NOT deduplicated -- callers decide whether
    duplicates are legal (a hand never has more than 4 of a kind).
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    out: list[int] = []
    for match in _TOKEN_RE.finditer(text):
        digits, key = match.group(1), match.group(2)
        suit = SUIT_NAMES.index(key) if key in SUIT_NAMES else SUIT_KEYS.index(key)
        for ch in digits:
            rank = int(ch)
            if not 1 <= rank <= 9:
                raise ValueError(f"invalid rank {ch!r} in {text!r}")
            out.append(make_tile(suit, rank))
    if not out:
        raise ValueError(f"could not parse any tiles from {text!r}")
    return sorted(out)


def counts_from_tiles(tiles: Iterable[int]) -> list[int]:
    """Build a length-27 count vector from an iterable of tile ids."""
    counts = [0] * NUM_KINDS
    for t in tiles:
        if not 0 <= t < NUM_KINDS:
            raise ValueError(f"tile out of range: {t}")
        counts[t] += 1
        if counts[t] > COPIES:
            raise ValueError(f"too many copies of {tile_to_str(t)}")
    return counts


def tiles_from_counts(counts: Sequence[int]) -> list[int]:
    out: list[int] = []
    for t, c in enumerate(counts):
        out.extend([t] * c)
    return out


def hand_str_from_counts(counts: Sequence[int]) -> str:
    return tiles_to_str(tiles_from_counts(counts))


def total_tiles(counts: Sequence[int]) -> int:
    return sum(counts)


def validate_counts(counts: Sequence[int]) -> None:
    if len(counts) != NUM_KINDS:
        raise ValueError(f"expected {NUM_KINDS} counts, got {len(counts)}")
    for t, c in enumerate(counts):
        if not 0 <= c <= COPIES:
            raise ValueError(f"bad count {c} for {tile_to_str(t)}")


def remaining_counts(seen: Sequence[int]) -> list[int]:
    """How many copies of each kind are still unseen."""
    return [COPIES - c for c in seen]
