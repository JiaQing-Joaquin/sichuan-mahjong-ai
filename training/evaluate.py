"""Evaluate a trained model against the rule-based baseline.

Two model seats vs two baseline seats, rotating positions to cancel seat bias.

Usage::

    python training/evaluate.py --checkpoint checkpoints/sl_policy.pt --games 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from policy_ai import ModelAI  # noqa: E402
from experiment import git_state, sha256_file  # noqa: E402
from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.game import Game, GameConfig  # noqa: E402
from sichuan_mahjong.replay import make_record  # noqa: E402
from sichuan_mahjong.scoring import FanConfig  # noqa: E402


def _bootstrap_interval(values: list[float], seed: int = 0) -> tuple[float, float]:
    """Deterministic percentile interval over paired per-game score gaps."""
    if len(values) < 2:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(2000)
    )
    return means[49], means[1949]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "sl_policy.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="",
                        help="comma-separated starting seeds; --games is per seed")
    parser.add_argument("--log", type=str, default="",
                        help="new JSONL path for per-game actions and scores")
    parser.add_argument("--game-config", type=Path,
                        help="JSON file with GameConfig fields")
    parser.add_argument("--fan-config", type=Path,
                        help="JSON file with FanConfig fields")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    all_game_seeds = [seed + k for seed in seeds for k in range(args.games)]
    if len(all_game_seeds) != len(set(all_game_seeds)):
        parser.error("seed ranges overlap; use non-overlapping --seeds groups")
    game_config = GameConfig(**json.loads(args.game_config.read_text(encoding="utf-8"))) if args.game_config else GameConfig()
    fan_config = FanConfig(**json.loads(args.fan_config.read_text(encoding="utf-8"))) if args.fan_config else FanConfig()

    model_ai = ModelAI(args.checkpoint, device=args.device)
    checkpoint_hash = sha256_file(args.checkpoint)
    git_commit, git_dirty = git_state(ROOT)

    model_score = 0
    base_score = 0
    model_wins = 0
    base_wins = 0
    model_winner_games = 0
    base_winner_games = 0

    per_game_gaps: list[float] = []
    log_path = Path(args.log) if args.log else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") if log_path else nullcontext(None) as log_file:
        for group_seed in seeds:
            for k in range(args.games):
                game_seed = group_seed + k
                s0 = k % 4
                model_seats = {s0, (s0 + 1) % 4}
                game = Game(seed=game_seed, config=game_config, fan_config=fan_config)
                # Reset fallback randomness per game: one seed never depends on
                # which games were evaluated before it.
                model_ai.baseline = BaselineAI(game_seed * 7 + 97)
                agents = {
                    i: model_ai if i in model_seats else BaselineAI(game_seed * 7 + i)
                    for i in range(4)
                }
                steps = 0
                while not game.is_over:
                    game.step(agents[game.current_player].act(game))
                    steps += 1
                    if steps > 5000:
                        raise RuntimeError(f"game seed {game_seed} exceeded 5000 actions")
                scores = game.scores()
                if sum(scores) != 0:
                    raise RuntimeError(f"game seed {game_seed} is not zero-sum: {scores}")
                model_total = sum(scores[s] for s in model_seats)
                base_total = sum(scores[s] for s in range(4) if s not in model_seats)
                per_game_gaps.append((model_total - base_total) / 2)
                model_score += model_total
                base_score += base_total
                model_wins += sum(game.players[s].won for s in model_seats)
                base_wins += sum(game.players[s].won for s in range(4) if s not in model_seats)
                model_winner_games += any(game.players[s].won for s in model_seats)
                base_winner_games += any(
                    game.players[s].won for s in range(4) if s not in model_seats
                )
                if log_file is not None:
                    record = make_record(
                        game, game_seed, group_seed=group_seed, game_index=k,
                        roles=["model" if i in model_seats else "baseline" for i in range(4)],
                        checkpoint=str(Path(args.checkpoint).resolve()),
                        checkpoint_sha256=checkpoint_hash,
                        git_commit=git_commit, git_dirty=git_dirty,
                    )
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_games = len(per_game_gaps)
    seat_games = total_games * 2
    lower, upper = _bootstrap_interval(per_game_gaps)
    print(f"games={total_games}  (2 model seats vs 2 baseline seats, rotating)")
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
    print(f"paired game gap: {statistics.mean(per_game_gaps):+.3f}  "
          f"bootstrap 95% interval [{lower:+.3f}, {upper:+.3f}]")
    if log_path is not None:
        print(f"per-game log: {log_path}")


if __name__ == "__main__":
    main()
