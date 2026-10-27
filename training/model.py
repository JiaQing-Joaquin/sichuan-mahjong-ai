"""Policy / value network with separate discard and call heads.

* ``discard_head``: 27 logits, one per tile (which tile to play)
* ``call_head``: 4 logits (胡 / 碰 / 杠 / 过), used when reacting to a discard
* ``value_head``: scalar estimate of the final score

Both heads are masked so illegal actions get ``-inf`` logits.
"""

from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 404,
        discard_dim: int = 27,
        call_dim: int = 4,
        hidden: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.discard_dim = discard_dim
        self.call_dim = call_dim
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.discard_head = nn.Linear(hidden // 2, discard_dim)
        self.call_head = nn.Linear(hidden // 2, call_dim)
        self.value_head = nn.Linear(hidden // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        discard_mask: torch.Tensor | None = None,
        call_mask: torch.Tensor | None = None,
    ):
        h = self.trunk(x)
        discard_logits = self.discard_head(h)
        call_logits = self.call_head(h)
        if discard_mask is not None:
            discard_logits = discard_logits.masked_fill(
                ~discard_mask, torch.finfo(discard_logits.dtype).min
            )
        if call_mask is not None:
            call_logits = call_logits.masked_fill(
                ~call_mask, torch.finfo(call_logits.dtype).min
            )
        value = self.value_head(h).squeeze(-1)
        return discard_logits, call_logits, value
