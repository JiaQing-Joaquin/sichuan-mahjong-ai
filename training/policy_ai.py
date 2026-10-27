"""Inference wrapper for a trained model.

* reaction phase  -> call head (胡 / 碰 / 杠 / 过)
* pure discard    -> discard head
* everything else -> rule-based baseline (定缺 / 换三张 / 自摸 / 杠)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.encoder import CALL_KINDS, encode  # noqa: E402
from sichuan_mahjong.game import ActKind, Action, Phase  # noqa: E402
from model import PolicyValueNet  # noqa: E402


class ModelAI:
    def __init__(self, checkpoint: str, device: str = "cpu", seed: int | None = None):
        ckpt = torch.load(checkpoint, map_location=device)
        self.model = PolicyValueNet(
            state_dim=ckpt["state_dim"],
            discard_dim=ckpt["discard_dim"],
            call_dim=ckpt["call_dim"],
            hidden=ckpt["hidden"],
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.baseline = BaselineAI(seed)

    def act(self, game) -> Action:
        seat = game.current_player
        actions = game.legal_actions()
        kinds = {a.kind for a in actions}

        if game.phase is Phase.REACTION:
            action = self._call(game, seat, actions)
            if action is not None:
                return action
            return self.baseline.choose(game, seat, actions)

        if kinds == {ActKind.DISCARD}:
            return self._discard(game, seat, actions)

        # 定缺 / 换三张 / 自摸 / 杠 仍由规则处理
        return self.baseline.choose(game, seat, actions)

    def _discard(self, game, seat: int, actions) -> Action:
        feats, dmask, cmask = encode(game, seat)
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        d = torch.tensor([dmask], dtype=torch.bool, device=self.device)
        c = torch.tensor([cmask], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            dlogits, _, _ = self.model(x, d, c)
        tile = int(dlogits.argmax(dim=1).item())
        if not dmask[tile]:
            tile = next(t for t, ok in enumerate(dmask) if ok)
        return Action(ActKind.DISCARD, tile=tile)

    def _call(self, game, seat: int, actions):
        feats, dmask, cmask = encode(game, seat)
        if not any(cmask):
            return None
        x = torch.tensor([feats], dtype=torch.float32, device=self.device)
        d = torch.tensor([dmask], dtype=torch.bool, device=self.device)
        c = torch.tensor([cmask], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _, clogits, _ = self.model(x, d, c)
        index = int(clogits.argmax(dim=1).item())
        if not cmask[index]:
            index = next(i for i, ok in enumerate(cmask) if ok)

        kind = CALL_KINDS[index]
        pending = game.pending_discard
        if kind is ActKind.PENG:
            action = Action(ActKind.PENG, tile=pending)
        elif kind is ActKind.MINGGANG:
            action = Action(ActKind.MINGGANG, tile=pending)
        else:
            action = Action(kind)
        return action if action in actions else None
