"""Collect supervised-learning samples from baseline self-play (v2).

Two kinds of decisions are recorded::

    T=0  discard : label = tile id (0..26), mask = legal discs (27)
    T=1  call    : label = 0..3 (胡/碰/杠/过), mask = legal calls (4)

Only reaction points with a real choice (>= 2 legal calls) are kept, which
removes the huge amount of trivial "pass" samples.

Features are accumulated per game as float16 arrays, so memory stays bounded
even for thousands of games.

Usage::

    python training/collect.py --games 5000 --out data/sl.npz
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.encoder import (  # noqa: E402
    CALL_DIM,
    CALL_KINDS,
    DISCARD_DIM,
    STATE_DIM,
    call_index,
    encode_state,
)
from sichuan_mahjong.game import ActKind, Game, Phase  # noqa: E402

SCORE_SCALE = 50.0
EMPTY_DISCARD_MASK = [False] * DISCARD_DIM
EMPTY_CALL_MASK = [False] * CALL_DIM


def collect(games: int, seed0: int = 0, epsilon: float = 0.0) -> dict:
    rng = random.Random(seed0)
    xs: list = []
    disc_masks: list = []
    call_masks: list = []
    labels: list = []
    types: list = []
    rewards: list = []
    game_ids: list = []
    seats: list = []
    draws = 0
    max_steps = 5000

    for k in range(games):
        game = Game(seed=seed0 + k)
        agents = [BaselineAI(seed0 * 1000 + k * 4 + i) for i in range(4)]
        local: list = []
        steps = 0

        while not game.is_over:
            seat = game.current_player
            if seat is None:
                break
            actions = game.legal_actions()
            action = agents[seat].choose(game, seat, actions)

            if action.kind is ActKind.DISCARD:
                if epsilon > 0.0 and rng.random() < epsilon:
                    action = rng.choice(
                        [a for a in actions if a.kind is ActKind.DISCARD]
                    )
                feats = encode_state(game, seat)
                dmask = EMPTY_DISCARD_MASK.copy()
                for a in actions:
                    if a.kind is ActKind.DISCARD and a.tile is not None:
                        dmask[a.tile] = True
                local.append(
                    (0, feats, dmask, EMPTY_CALL_MASK, int(action.tile), seat)
                )

            elif game.phase is Phase.REACTION and action.kind in CALL_KINDS:
                present = {a.kind for a in actions}
                cmask = [kind in present for kind in CALL_KINDS]
                if sum(cmask) >= 2:  # only keep real decisions
                    if epsilon > 0.0 and rng.random() < epsilon:
                        action = rng.choice(
                            [a for a in actions if a.kind in CALL_KINDS]
                        )
                    feats = encode_state(game, seat)
                    local.append(
                        (
                            1,
                            feats,
                            EMPTY_DISCARD_MASK,
                            cmask,
                            call_index(action.kind),
                            seat,
                        )
                    )

            game.step(action)
            steps += 1
            if steps > max_steps:
                break

        if not any(p.won for p in game.players):
            draws += 1

        scores = game.scores()
        if local:
            xs.append(np.asarray([r[1] for r in local], dtype=np.float16))
            disc_masks.append(np.asarray([r[2] for r in local], dtype=bool))
            call_masks.append(np.asarray([r[3] for r in local], dtype=bool))
            labels.extend(r[4] for r in local)
            types.extend(r[0] for r in local)
            rewards.extend(scores[r[5]] / SCORE_SCALE for r in local)
            game_ids.extend([k] * len(local))
            seats.extend(r[5] for r in local)

    data = {
        "X": np.concatenate(xs, axis=0) if xs else np.zeros((0, STATE_DIM), np.float16),
        "D": np.concatenate(disc_masks, axis=0) if disc_masks else np.zeros((0, DISCARD_DIM), bool),
        "C": np.concatenate(call_masks, axis=0) if call_masks else np.zeros((0, CALL_DIM), bool),
        "Y": np.asarray(labels, dtype=np.int64),
        "T": np.asarray(types, dtype=np.int8),
        "R": np.asarray(rewards, dtype=np.float32),
        "G": np.asarray(game_ids, dtype=np.int32),
        "S": np.asarray(seats, dtype=np.int32),
    }
    data["meta"] = {
        "games": games,
        "draws": draws,
        "samples": len(labels),
        "discard_samples": int((data["T"] == 0).sum()),
        "call_samples": int((data["T"] == 1).sum()),
        "state_dim": STATE_DIM,
        "discard_dim": DISCARD_DIM,
        "call_dim": CALL_DIM,
        "score_scale": SCORE_SCALE,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "sl.npz"))
    args = parser.parse_args()

    started = time.perf_counter()
    data = collect(args.games, args.seed, args.epsilon)
    elapsed = time.perf_counter() - started

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=data["X"],
        D=data["D"],
        C=data["C"],
        Y=data["Y"],
        T=data["T"],
        R=data["R"],
        G=data["G"],
        S=data["S"],
    )

    meta = data["meta"]
    print(f"games            : {meta['games']}")
    print(f"draws            : {meta['draws']}")
    print(f"samples          : {meta['samples']}")
    print(f"  discards       : {meta['discard_samples']}")
    print(f"  calls          : {meta['call_samples']}")
    print(f"elapsed          : {elapsed:.1f}s")
    print(f"saved            : {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
