"""Micro-benchmark for shanten calculation.

Usage::

    python scripts/bench.py [hands]
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sichuan_mahjong.hand import shanten  # noqa: E402
from sichuan_mahjong.tiles import COPIES, NUM_KINDS  # noqa: E402


def random_hand(rng: random.Random, n: int) -> list[int]:
    bag = [t for t in range(NUM_KINDS) for _ in range(COPIES)]
    rng.shuffle(bag)
    counts = [0] * NUM_KINDS
    for t in bag[:n]:
        counts[t] += 1
    return counts


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = random.Random(0)
    hands_13 = [random_hand(rng, 13) for _ in range(n)]
    hands_14 = [random_hand(rng, 14) for _ in range(n)]

    for label, hands in (("13-tile", hands_13), ("14-tile", hands_14)):
        t0 = time.perf_counter()
        values = [shanten(h) for h in hands]
        dt = time.perf_counter() - t0
        avg = sum(values) / len(values)
        print(
            f"{label}: {len(hands)} hands in {dt:.3f}s "
            f"({len(hands) / dt:,.0f} hands/s), avg shanten {avg:.2f}"
        )


if __name__ == "__main__":
    main()
