"""A/B evaluation: baseline vs random, 2 seats each, rotating positions.

Usage::

    python scripts/eval.py [games]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.game import Game  # noqa: E402


class RandomAI:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, game: Game):
        return self.rng.choice(game.legal_actions())


def play(seed: int, baseline_seats: set[int]) -> Game:
    g = Game(seed=seed)
    agents = {
        i: (BaselineAI(seed * 7 + i) if i in baseline_seats else RandomAI(seed * 7 + i))
        for i in range(4)
    }
    steps = 0
    while not g.is_over:
        g.step(agents[g.current_player].act(g))
        steps += 1
        if steps > 5000:
            break
    return g


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    base_score = 0
    rand_score = 0
    base_wins = 0
    rand_wins = 0
    base_hu_games = 0
    rand_hu_games = 0

    for k in range(games):
        s0 = k % 4
        base_seats = {s0, (s0 + 1) % 4}
        g = play(seed=k, baseline_seats=base_seats)
        for p in g.players:
            if p.seat in base_seats:
                base_score += p.score
                if p.won:
                    base_wins += 1
            else:
                rand_score += p.score
                if p.won:
                    rand_wins += 1
        if any(g.players[s].won for s in base_seats):
            base_hu_games += 1
        if any(g.players[s].won for s in range(4) if s not in base_seats):
            rand_hu_games += 1

    print(f"games={games}")
    print(
        f"baseline seats : avg score {base_score / (games * 2):+.3f}, "
        f"hu count {base_wins}, games with a baseline winner {base_hu_games}"
    )
    print(
        f"random   seats : avg score {rand_score / (games * 2):+.3f}, "
        f"hu count {rand_wins}, games with a random winner {rand_hu_games}"
    )
    print(
        f"score differential (baseline - random, per seat): "
        f"{(base_score - rand_score) / (games * 2):+.3f}"
    )


if __name__ == "__main__":
    main()
