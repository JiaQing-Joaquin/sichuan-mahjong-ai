"""Self-play / evaluation harness for the 血战到底 engine.

Usage::

    python scripts/simulate.py [games] [seed]

Runs one baseline seat against three random seats and reports per-seat
statistics.  This is the "evaluation floor" for every learned policy.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.game import Game, GameConfig  # noqa: E402


class RandomAI:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, game: Game):
        return self.rng.choice(game.legal_actions())


def play(seed: int, ai_seats: set[int]) -> Game:
    g = Game(seed=seed, config=GameConfig(gang_payment={"ming": 1, "an": 2, "bu": 1}))
    agents = {
        i: (BaselineAI(seed * 10 + i) if i in ai_seats else RandomAI(seed * 10 + i))
        for i in range(4)
    }
    steps = 0
    while not g.is_over:
        seat = g.current_player
        if seat is None:
            break
        g.step(agents[seat].act(g))
        steps += 1
        if steps > 5000:
            break
    return g


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    wins = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    fan_counter: Counter[str] = Counter()
    draws = 0
    hu_games = 0

    for k in range(games):
        seed = seed0 + k
        ai_seat = k % 4
        g = play(seed, {ai_seat})
        won = [p for p in g.players if p.won]
        if not won:
            draws += 1
        else:
            hu_games += 1
        for p in g.players:
            total[p.seat] += p.score
            if p.won:
                wins[p.seat] += 1
                if p.win_item:
                    fan_counter[p.win_item.split("+")[0]] += 1

    print(f"games={games}  hands with a winner={hu_games}  draws={draws}")
    print("seat   wins   total_score   avg_score")
    for s in range(4):
        print(f"{s:>4} {wins[s]:>6} {total[s]:>13} {total[s] / games:>+.2f}")
    print("\nshape distribution (winning hands):")
    for name, cnt in fan_counter.most_common():
        print(f"  {name:<8} {cnt}")


if __name__ == "__main__":
    main()
