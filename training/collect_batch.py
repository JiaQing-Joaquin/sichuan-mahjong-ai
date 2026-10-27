"""Collect a large dataset in shards, each in a fresh process, then merge.

Running thousands of games in one process degrades (memory growth / GC), so
this driver runs ``collect.py`` several times and concatenates the shards.

Usage::

    python training/collect_batch.py --games 5000 --shard-size 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("X", "D", "C", "Y", "T", "R", "G", "S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "sl.npz"))
    args = parser.parse_args()

    shard_dir = ROOT / "data" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    n_shards = (args.games + args.shard_size - 1) // args.shard_size
    started = time.perf_counter()

    for i in range(n_shards):
        seed = args.seed + i * args.shard_size
        out = shard_dir / f"shard_{i:03d}.npz"
        cmd = [
            sys.executable,
            str(ROOT / "training" / "collect.py"),
            "--games", str(args.shard_size),
            "--seed", str(seed),
            "--out", str(out),
        ]
        print(f"[shard {i + 1}/{n_shards}] seed={seed}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    parts = [np.load(shard_dir / f"shard_{i:03d}.npz") for i in range(n_shards)]
    merged = {}
    for key in KEYS:
        merged[key] = np.concatenate([part[key] for part in parts], axis=0)
    # remap game ids so they stay unique across shards
    merged["G"] = np.concatenate(
        [parts[i]["G"] + i * args.shard_size for i in range(n_shards)]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)

    total = len(merged["Y"])
    discards = int((merged["T"] == 0).sum())
    calls = int((merged["T"] == 1).sum())
    print(f"samples          : {total}")
    print(f"  discards       : {discards}")
    print(f"  calls          : {calls}")
    print(f"elapsed          : {time.perf_counter() - started:.1f}s")
    print(f"saved            : {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
