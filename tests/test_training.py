"""Smoke test for the training pipeline.

Skipped automatically when numpy/torch are not installed, so the core test
suite keeps working on a bare Python.

Run with::

    python tests/test_training.py
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
        import numpy as np
        import torch
    except Exception:  # noqa: BLE001
        print("SKIP: numpy/torch not available")
        return 0

    training_dir = str(Path(__file__).resolve().parents[1] / "training")
    sys.path.insert(0, training_dir)

    from collect import collect
    from model import PolicyValueNet
    from sichuan_mahjong.encoder import CALL_DIM, DISCARD_DIM, STATE_DIM

    data = collect(games=2, seed0=0)
    check("samples positive", len(data["Y"]) > 0, True)
    check("feature dim", data["X"].shape[1], STATE_DIM)
    check("discard dim", data["D"].shape[1], DISCARD_DIM)
    check("call dim", data["C"].shape[1], CALL_DIM)
    check("labels legal (discard)", bool(data["D"][np.arange(len(data["Y"])), data["Y"]][data["T"] == 0].all()), True)
    check("discard samples exist", int((data["T"] == 0).sum()) > 0, True)

    net = PolicyValueNet(
        state_dim=STATE_DIM, discard_dim=DISCARD_DIM, call_dim=CALL_DIM, hidden=64
    )
    x = torch.from_numpy(data["X"]).float()
    d = torch.from_numpy(data["D"])
    c = torch.from_numpy(data["C"])
    y = torch.from_numpy(data["Y"])
    t = torch.from_numpy(data["T"])
    r = torch.from_numpy(data["R"])

    dlogits, clogits, value = net(x, d, c)
    check("discard logits shape", tuple(dlogits.shape), (len(y), DISCARD_DIM))
    check("call logits shape", tuple(clogits.shape), (len(y), CALL_DIM))

    is_discard = t == 0
    loss = torch.nn.functional.cross_entropy(dlogits[is_discard], y[is_discard])
    if bool((~is_discard).any()):
        loss = loss + torch.nn.functional.cross_entropy(clogits[~is_discard], y[~is_discard])
    loss = loss + torch.nn.functional.mse_loss(value, r)
    check("loss finite", bool(torch.isfinite(loss)), True)
    loss.backward()
    check("gradients flow", net.discard_head.weight.grad is not None, True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        torch.save(
            {
                "model": net.state_dict(),
                "state_dim": STATE_DIM,
                "discard_dim": DISCARD_DIM,
                "call_dim": CALL_DIM,
                "hidden": 64,
            },
            path,
        )
        loaded = torch.load(path, map_location="cpu")
        check("checkpoint state_dim", loaded["state_dim"], STATE_DIM)
        check(
            "checkpoint keys",
            set(loaded["model"].keys()) == set(net.state_dict().keys()),
            True,
        )

    if _FAILS:
        print(f"FAILED {len(_FAILS)} / {_PASSES + len(_FAILS)} checks")
        for failure in _FAILS:
            print("  -", failure)
        return 1
    print(f"OK: {_PASSES} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
