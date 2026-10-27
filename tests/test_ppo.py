"""Smoke test for the PPO pipeline.

Skipped automatically when torch is not installed.

Run with::

    python tests/test_ppo.py
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_FAILS: list[str] = []
_PASSES = 0


def check(name: str, got, want) -> None:
    global _PASSES
    if got == want:
        _PASSES += 1
    else:
        message = f"{name}: got {got!r}, want {want!r}"
        _FAILS.append(message)
        if "pytest" in sys.modules:
            raise AssertionError(message)


def main() -> int:
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("SKIP: torch not available")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

    from model import PolicyValueNet
    from ppo import collect_rollouts, ppo_update
    from sichuan_mahjong.encoder import CALL_DIM, DISCARD_DIM, STATE_DIM

    model = PolicyValueNet(
        state_dim=STATE_DIM, discard_dim=DISCARD_DIM, call_dim=CALL_DIM, hidden=32
    )
    data, stats = collect_rollouts(model, "cpu", games=1, seed0=0)
    check("rollout records transitions", len(data) > 0, True)
    check("all rewards finite", all(abs(d["ret"]) <= 5.0 for d in data), True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    args = Namespace(
        device="cpu",
        epochs=1,
        batch=64,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    update = ppo_update(model, optimizer, data, args)
    check("policy loss finite", bool(torch.isfinite(torch.tensor(update["pi"]))), True)
    check("value loss finite", bool(torch.isfinite(torch.tensor(update["v"]))), True)
    single = ppo_update(model, optimizer, data[:1], args)
    check("single-transition policy loss finite", bool(torch.isfinite(torch.tensor(single["pi"]))), True)
    check("single-transition value loss finite", bool(torch.isfinite(torch.tensor(single["v"]))), True)

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for failure in _FAILS:
            print("  -", failure)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
