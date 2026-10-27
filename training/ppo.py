"""PPO self-play fine-tuning on top of the supervised checkpoint.

The policy is shared by two seats per game while the other two seats play the
rule-based baseline (positions rotate every game).  Reward is the final score
normalised by 50, so the value head keeps the same scale as during SL.

Usage::

    python training/ppo.py --init checkpoints/sl_policy.pt \
        --out checkpoints/ppo_policy.pt --iters 15 --games 32
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from model import PolicyValueNet  # noqa: E402
from policy_ai import ModelAI  # noqa: E402
from sichuan_mahjong.baseline import BaselineAI  # noqa: E402
from sichuan_mahjong.encoder import CALL_KINDS, encode  # noqa: E402
from sichuan_mahjong.game import ActKind, Action, Game, Phase  # noqa: E402

REWARD_CLIP = 5.0


def _batch(feats, dm, cm, device):
    return (
        torch.tensor([feats], dtype=torch.float32, device=device),
        torch.tensor([dm], dtype=torch.bool, device=device),
        torch.tensor([cm], dtype=torch.bool, device=device),
    )


class RolloutAgent:
    """Model policy with rule fallback; records every model decision."""

    def __init__(self, model, device, seed=None):
        self.model = model
        self.device = device
        self.baseline = BaselineAI(seed)
        self.records: list[dict] = []

    def act(self, game: Game, seat: int) -> Action:
        actions = game.legal_actions()
        kinds = {a.kind for a in actions}
        if game.phase is Phase.REACTION:
            return self._call(game, seat, actions)
        if kinds == {ActKind.DISCARD}:
            return self._discard(game, seat, actions)
        return self.baseline.choose(game, seat, actions)

    def _discard(self, game, seat, actions) -> Action:
        feats, dm, cm = encode(game, seat)
        x, d, c = _batch(feats, dm, cm, self.device)
        with torch.no_grad():
            dlogits, _, value = self.model(x, d, c)
        dist = torch.distributions.Categorical(logits=dlogits[0])
        sample = dist.sample()
        logp = dist.log_prob(sample)
        tile = int(sample.item())
        if not dm[tile]:
            tile = next(t for t, ok in enumerate(dm) if ok)
        self.records.append(
            {
                "t": 0,
                "feats": feats,
                "dm": dm,
                "cm": cm,
                "a": tile,
                "logp": float(logp),
                "value": float(value),
                "seat": seat,
            }
        )
        return Action(ActKind.DISCARD, tile=tile)

    def _call(self, game, seat, actions) -> Action:
        feats, dm, cm = encode(game, seat)
        if not any(cm):
            return Action(ActKind.PASS)
        x, d, c = _batch(feats, dm, cm, self.device)
        with torch.no_grad():
            _, clogits, value = self.model(x, d, c)
        dist = torch.distributions.Categorical(logits=clogits[0])
        sample = dist.sample()
        logp = dist.log_prob(sample)
        index = int(sample.item())
        kind = CALL_KINDS[index]
        pending = game.pending_discard
        if kind is ActKind.PENG:
            action = Action(ActKind.PENG, tile=pending)
        elif kind is ActKind.MINGGANG:
            action = Action(ActKind.MINGGANG, tile=pending)
        else:
            action = Action(kind)
        if action not in actions:
            return self.baseline.choose(game, seat, actions)
        self.records.append(
            {
                "t": 1,
                "feats": feats,
                "dm": dm,
                "cm": cm,
                "a": index,
                "logp": float(logp),
                "value": float(value),
                "seat": seat,
            }
        )
        return action


def collect_rollouts(model, device, games: int, seed0: int, opponents=None):
    model.eval()
    data: list[dict] = []
    score_sum = 0.0
    hu_count = 0
    if opponents is None:
        opponents = [BaselineAI(seed0)]
    for k in range(games):
        game = Game(seed=seed0 + k)
        policy_seats = {k % 4, (k + 1) % 4}
        agent = RolloutAgent(model, device, seed=seed0 * 100 + k)

        steps = 0
        while not game.is_over:
            seat = game.current_player
            if seat is None:
                break
            if seat in policy_seats:
                action = agent.act(game, seat)
            else:
                action = random.choice(opponents).act(game)
            game.step(action)
            steps += 1
            if steps > 5000:
                break

        scores = game.scores()
        for rec in agent.records:
            rec["ret"] = max(-REWARD_CLIP, min(REWARD_CLIP, scores[rec["seat"]] / 50.0))
        data.extend(agent.records)
        for seat in policy_seats:
            score_sum += scores[seat]
            if game.players[seat].won:
                hu_count += 1
    return data, {"score_per_seat": score_sum / max(games * 2, 1), "hu": hu_count}


def ppo_update(model, optimizer, data, args):
    if not data:
        return {"pi": 0.0, "v": 0.0, "ent": 0.0}

    device = args.device
    x = torch.tensor([d["feats"] for d in data], dtype=torch.float32)
    dm = torch.tensor([d["dm"] for d in data], dtype=torch.bool)
    cm = torch.tensor([d["cm"] for d in data], dtype=torch.bool)
    tb = torch.tensor([d["t"] for d in data], dtype=torch.long)
    ab = torch.tensor([d["a"] for d in data], dtype=torch.long)
    old = torch.tensor([d["logp"] for d in data], dtype=torch.float32)
    val = torch.tensor([d["value"] for d in data], dtype=torch.float32)
    ret = torch.tensor([d["ret"] for d in data], dtype=torch.float32)
    adv = ret - val
    # Population std stays finite for a one-transition rollout.
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)

    n = len(data)
    indices = list(range(n))
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0}
    batches = 0

    model.train()
    for _ in range(args.epochs):
        random.shuffle(indices)
        for start in range(0, n, args.batch):
            mb = indices[start:start + args.batch]
            bx = x[mb].to(device)
            bdm = dm[mb].to(device)
            bcm = cm[mb].to(device)
            bt = tb[mb].to(device)
            ba = ab[mb].to(device)
            bo = old[mb].to(device)
            br = ret[mb].to(device)
            badv = adv[mb].to(device)

            dlogits, clogits, value = model(bx, bdm, bcm)
            logp_d = torch.log_softmax(dlogits, dim=1)
            logp_c = torch.log_softmax(clogits, dim=1)
            # clamp both gathers: the unused branch's value is discarded below
            idx_d = torch.clamp(ba, max=dlogits.size(1) - 1)
            idx_c = torch.clamp(ba, max=clogits.size(1) - 1)
            logp_d_sel = logp_d.gather(1, idx_d.unsqueeze(1)).squeeze(1)
            logp_c_sel = logp_c.gather(1, idx_c.unsqueeze(1)).squeeze(1)
            logp = torch.where(bt == 0, logp_d_sel, logp_c_sel)
            ratio = torch.exp(logp - bo)
            surr1 = ratio * badv
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * badv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, br)

            # Keep probabilities in the graph so entropy regularizes the policy.
            p_d = torch.softmax(dlogits, dim=1)
            p_c = torch.softmax(clogits, dim=1)
            ent_d = -(p_d * logp_d).sum(dim=1)
            ent_c = -(p_c * logp_c).sum(dim=1)
            entropy = torch.where(bt == 0, ent_d, ent_c).mean()

            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            stats["pi"] += float(policy_loss)
            stats["v"] += float(value_loss)
            stats["ent"] += float(entropy)
            batches += 1

    return {k: v / max(batches, 1) for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default=str(ROOT / "checkpoints" / "sl_policy.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "checkpoints" / "ppo_policy.pt"))
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--league",
        type=str,
        default="",
        help="comma-separated checkpoints to use as extra opponents",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.init, map_location=args.device)
    model = PolicyValueNet(
        state_dim=ckpt["state_dim"],
        discard_dim=ckpt["discard_dim"],
        call_dim=ckpt["call_dim"],
        hidden=ckpt["hidden"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    opponents = [BaselineAI(args.seed)]
    for path in [p.strip() for p in args.league.split(",") if p.strip()]:
        opponents.append(ModelAI(path, device=args.device))
    print(f"opponents: {len(opponents)} (1 rule AI + {len(opponents) - 1} league)")

    started = time.perf_counter()
    for it in range(1, args.iters + 1):
        data, rollout_stats = collect_rollouts(
            model, args.device, args.games, args.seed + it * 1000, opponents=opponents
        )
        update_stats = ppo_update(model, optimizer, data, args)
        print(
            f"iter {it:>2}  transitions {len(data):>5}  "
            f"rollout score/seat {rollout_stats['score_per_seat']:+.2f}  "
            f"hu {rollout_stats['hu']:>3}  |  "
            f"pi {update_stats['pi']:+.4f}  v {update_stats['v']:.4f}  "
            f"ent {update_stats['ent']:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": ckpt["state_dim"],
            "discard_dim": ckpt["discard_dim"],
            "call_dim": ckpt["call_dim"],
            "hidden": ckpt["hidden"],
            "score_scale": 50.0,
        },
        out_path,
    )
    print(f"elapsed {time.perf_counter() - started:.1f}s  saved {out_path}")


if __name__ == "__main__":
    main()
