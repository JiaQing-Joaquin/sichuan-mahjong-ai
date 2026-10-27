"""Smoke test for the DAgger collector.

Skipped automatically when torch is not installed.

Run with::

    python tests/test_dagger.py
"""

from __future__ import annotations

import sys
import tempfile
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

    training_dir = str(Path(__file__).resolve().parents[1] / "training")
    sys.path.insert(0, training_dir)

    from dagger import collect_dagger
    from model import PolicyValueNet
    from sichuan_mahjong.encoder import CALL_DIM, DISCARD_DIM, STATE_DIM

    with tempfile.TemporaryDirectory() as tmp:
        net = PolicyValueNet(
            state_dim=STATE_DIM, discard_dim=DISCARD_DIM, call_dim=CALL_DIM, hidden=32
        )
        ckpt_path = Path(tmp) / "ckpt.pt"
        torch.save(
            {
                "model": net.state_dict(),
                "state_dim": STATE_DIM,
                "discard_dim": DISCARD_DIM,
                "call_dim": CALL_DIM,
                "hidden": 32,
            },
            ckpt_path,
        )

        data = collect_dagger(str(ckpt_path), games=1, seed0=0)
        check("dagger samples positive", len(data["Y"]) > 0, True)
        check("feature dim", data["X"].shape[1], STATE_DIM)
        check("discard labels legal",
              bool(data["D"][range(len(data["Y"])), data["Y"]][data["T"] == 0].all()), True)
        check("discard samples exist", int((data["T"] == 0).sum()) > 0, True)

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for failure in _FAILS:
            print("  -", failure)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
