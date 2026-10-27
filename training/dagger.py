"""DAgger-style data aggregation.

The current policy plays; the **expert** (rule AI) labels every state the
policy actually visits.  Training on these states fixes the distribution
shift that pure behaviour cloning suffers from ("compounding error").

Usage::

    python training/dagger.py --init checkpoints/sl_policy.pt \
        --games 600 --out data/dagger.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from policy_ai import ModelAI  # noqa: E402
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


def _discard_mask(actions) -> list[bool]:
    mask = [False] * DISCARD_DIM
    for a in actions:
        if a.kind is ActKind.DISCARD and a.tile is not None:
            mask[a.tile] = True
    return mask


def _call_mask(actions) -> list[bool]:
    present = {a.kind for a in actions}
    return [kind in present for kind in CALL_KINDS]


def collect_dagger(checkpoint: str, games: int, seed0: int = 0) -> dict:
    model_ai = ModelAI(checkpoint)
    xs: list = []
    disc_masks: list = []
    call_masks: list = []
    labels: list = []
    types: list = []
    rewards: list = []
    game_ids: list = []
    seats: list = []
    draws = 0

    for k in range(games):
        game = Game(seed=seed0 + k)
        expert = BaselineAI(seed=seed0 * 100 + k)
        policy_seats = {k % 4, (k + 1) % 4}
        local: list = []
        steps = 0

        while not game.is_over:
            seat = game.current_player
            if seat is None:
                break
            actions = game.legal_actions()

            if seat in policy_seats:
                action = model_ai.act(game)
                expert_action = expert.choose(game, seat, actions)

                if action.kind is ActKind.DISCARD:
                    feats = encode_state(game, seat)
                    local.append(
                        (
                            0,
                            feats,
                            _discard_mask(actions),
                            [False] * CALL_DIM,
                            int(expert_action.tile),
                            seat,
                        )
                    )
                elif game.phase is Phase.REACTION and action.kind in CALL_KINDS:
                    cmask = _call_mask(actions)
                    if sum(cmask) >= 2:
                        feats = encode_state(game, seat)
                        local.append(
                            (
                                1,
                                feats,
                                [False] * DISCARD_DIM,
                                cmask,
                                call_index(expert_action.kind),
                                seat,
                            )
                        )
            else:
                action = expert.choose(game, seat, actions)

            game.step(action)
            steps += 1
            if steps > 5000:
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
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=str(ROOT / "checkpoints" / "sl_policy.pt"))
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "dagger.npz"))
    args = parser.parse_args()

    started = time.perf_counter()
    data = collect_dagger(args.init, args.games, args.seed)
    elapsed = time.perf_counter() - started

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=data["X"], D=data["D"], C=data["C"], Y=data["Y"],
        T=data["T"], R=data["R"], G=data["G"], S=data["S"],
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
