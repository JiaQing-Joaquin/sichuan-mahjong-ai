"""Supervised learning on collected discard + call samples.

Usage::

    python training/train_sl.py --data data/sl.npz --epochs 15
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from model import PolicyValueNet  # noqa: E402
from experiment import git_state, sha256_file  # noqa: E402


def load_datasets(paths: list[str]) -> dict:
    """Load and concatenate one or more npz datasets (``--data a,b``)."""
    keys = ("X", "D", "C", "Y", "T", "R", "G", "S")
    merged: dict = {k: [] for k in keys}
    offset = 0
    for path in paths:
        d = np.load(path)
        g = d["G"].astype(np.int32)
        if len(g):
            g = g + offset
            offset = int(g.max()) + 1
        merged["G"].append(g)
        for key in keys:
            if key != "G":
                merged[key].append(d[key])
    return {k: np.concatenate(v, axis=0) for k, v in merged.items()}


def split_by_game(game_ids: np.ndarray, val_frac: float, seed: int):
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val].tolist())
    is_val = np.array([g in val_games for g in game_ids], dtype=bool)
    return np.where(~is_val)[0], np.where(is_val)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "sl.npz"))
    parser.add_argument("--out", type=str, default=str(ROOT / "checkpoints" / "sl_policy.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0:
        parser.error("--epochs and --batch must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_paths = [p.strip() for p in args.data.split(",") if p.strip()]
    if not data_paths:
        parser.error("--data must contain at least one dataset")
    raw = load_datasets(data_paths)
    state_dim = raw["X"].shape[1]
    discard_dim = raw["D"].shape[1]
    call_dim = raw["C"].shape[1]
    print(f"datasets: {len(args.data.split(','))}  samples: {len(raw['Y'])}")

    tr_idx, va_idx = split_by_game(raw["G"], args.val_frac, args.seed)
    if not len(tr_idx) or not len(va_idx):
        parser.error("dataset needs at least two games for a nonempty train/validation split")

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(raw["X"][idx]),
            torch.from_numpy(raw["D"][idx]),
            torch.from_numpy(raw["C"][idx]),
            torch.from_numpy(raw["Y"][idx]),
            torch.from_numpy(raw["T"][idx]),
            torch.from_numpy(raw["R"][idx]),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    tr_loader = make_loader(tr_idx, True)
    va_loader = make_loader(va_idx, False)

    model = PolicyValueNet(
        state_dim=state_dim,
        discard_dim=discard_dim,
        call_dim=call_dim,
        hidden=args.hidden,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total = 0.0
        seen = 0
        d_correct = d_total = 0
        c_correct = c_total = 0
        for xb, db, cb, yb, tb, rb in loader:
            xb = xb.float().to(args.device)
            db = db.to(args.device)
            cb = cb.to(args.device)
            yb = yb.to(args.device)
            tb = tb.to(args.device)
            rb = rb.to(args.device)

            dlogits, clogits, value = model(xb, db, cb)
            is_discard = tb == 0

            loss = torch.zeros((), device=args.device)
            if bool(is_discard.any()):
                loss = loss + F.cross_entropy(dlogits[is_discard], yb[is_discard])
            if bool((~is_discard).any()):
                loss = loss + F.cross_entropy(clogits[~is_discard], yb[~is_discard])
            loss = loss + args.value_coef * F.mse_loss(value, rb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * len(yb)
            seen += len(yb)
            with torch.no_grad():
                if bool(is_discard.any()):
                    d_correct += int(
                        (dlogits[is_discard].argmax(1) == yb[is_discard]).sum()
                    )
                    d_total += int(is_discard.sum())
                if bool((~is_discard).any()):
                    c_correct += int(
                        (clogits[~is_discard].argmax(1) == yb[~is_discard]).sum()
                    )
                    c_total += int((~is_discard).sum())
        return {
            "loss": total / max(seen, 1),
            "discard_acc": d_correct / max(d_total, 1),
            "call_acc": c_correct / max(c_total, 1),
            "n": seen,
        }

    print(f"train samples: {len(tr_idx)}  val samples: {len(va_idx)}")
    best_val = float("inf")
    best_state = None
    best_epoch = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(tr_loader, True)
        va = run_epoch(va_loader, False)
        print(
            f"epoch {epoch:>2}  train loss {tr['loss']:.4f} "
            f"disc {tr['discard_acc']:.3f} call {tr['call_acc']:.3f}  |  "
            f"val loss {va['loss']:.4f} disc {va['discard_acc']:.3f} "
            f"call {va['call_acc']:.3f}"
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = git_state(ROOT)
    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": state_dim,
            "discard_dim": discard_dim,
            "call_dim": call_dim,
            "hidden": args.hidden,
            "score_scale": 50.0,
            "training_meta": {
                "algorithm": "behavior_cloning",
                "seed": args.seed,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "batch": args.batch,
                "lr": args.lr,
                "value_coef": args.value_coef,
                "val_frac": args.val_frac,
                "train_samples": len(tr_idx),
                "val_samples": len(va_idx),
                "datasets": [
                    {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                    for path in data_paths
                ],
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "torch_version": torch.__version__,
            },
        },
        out_path,
    )
    print(f"best val loss {best_val:.4f}  elapsed {time.perf_counter() - started:.1f}s")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
