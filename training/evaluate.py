"""Evaluate a trained model against the rule-based baseline.

Two model seats vs two baseline seats, rotating positions to cancel seat bias.

Usage::

    python training/evaluate.py --checkpoint checkpoints/sl_policy.pt --games 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from policy_ai import ModelAI  # noqa: E402
from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.game import Game  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "sl_policy.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_ai = ModelAI(args.checkpoint, device=args.device)

    model_score = 0
    base_score = 0
    model_wins = 0
    base_wins = 0
    model_winner_games = 0
    base_winner_games = 0

    for k in range(args.games):
        s0 = k % 4
        model_seats = {s0, (s0 + 1) % 4}

        game = Game(seed=args.seed + k)
        agents = {}
        for i in range(4):
            agents[i] = model_ai if i in model_seats else BaselineAI(args.seed * 7 + i)

        steps = 0
        while not game.is_over:
            game.step(agents[game.current_player].act(game))
            steps += 1
            if steps > 5000:
                break

        for p in game.players:
            if p.seat in model_seats:
                model_score += p.score
                if p.won:
                    model_wins += 1
            else:
                base_score += p.score
                if p.won:
                    base_wins += 1

        if any(game.players[s].won for s in model_seats):
            model_winner_games += 1
        if any(game.players[s].won for s in range(4) if s not in model_seats):
            base_winner_games += 1

    seat_games = args.games * 2
    print(f"games={args.games}  (2 model seats vs 2 baseline seats, rotating)")
    print(
        f"model    : avg score {model_score / seat_games:+.3f}  "
        f"hu {model_wins}  games with a model winner {model_winner_games}"
    )
    print(
        f"baseline : avg score {base_score / seat_games:+.3f}  "
        f"hu {base_wins}  games with a baseline winner {base_winner_games}"
    )
    print(f"score differential (model - baseline, per seat): "
          f"{(model_score - base_score) / seat_games:+.3f}")


if __name__ == "__main__":
    main()
