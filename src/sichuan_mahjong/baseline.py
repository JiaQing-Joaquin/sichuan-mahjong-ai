"""A rule-based baseline agent.

This is intentionally simple and fast.  It exists to (a) be a playable
opponent on day one, (b) generate self-play data for the supervised stage and
(c) provide a lower bound for evaluating learned policies.

It only reads information that a real player could see: its own hand, all
discards, all exposed melds, everyone's declared missing suit and the wall
size.
"""

from __future__ import annotations

import random

from .game import ActKind, Action, Game
from .hand import shanten
from .tiles import COPIES, NUM_KINDS, suit_of


class BaselineAI:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def act(self, game: Game) -> Action:
        seat = game.current_player
        if seat is None:
            raise RuntimeError("no player to act")
        actions = game.legal_actions()
        return self.choose(game, seat, actions)

    def choose(self, game: Game, seat: int, actions: list[Action]) -> Action:
        kinds = {a.kind for a in actions}
        p = game.players[seat]

        if ActKind.CHOOSE_MISSING in kinds:
            return Action(ActKind.CHOOSE_MISSING, suit=self._pick_missing(p.hand))
        if ActKind.SWAP_THREE in kinds:
            return self._pick_swap_action(p.hand, actions)

        if ActKind.HU_SELF in kinds:
            return Action(ActKind.HU_SELF)
        if ActKind.HU_DISCARD in kinds:
            return Action(ActKind.HU_DISCARD)

        # kongs are almost always good for the kong holder
        for kind in (ActKind.ANKANG, ActKind.BUKANG, ActKind.MINGGANG):
            if kind in kinds:
                return next(a for a in actions if a.kind is kind)

        if ActKind.PENG in kinds:
            peng = next(a for a in actions if a.kind is ActKind.PENG)
            if self._should_peng(game, seat, peng.tile):
                return peng

        if ActKind.PASS in kinds:
            return Action(ActKind.PASS)

        discards = [a for a in actions if a.kind is ActKind.DISCARD]
        if discards:
            return self._best_discard(game, seat, discards)

        return actions[0]

    # ------------------------------------------------------------------
    # 定缺
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_missing(hand: list[int]) -> int:
        """Choose the weakest suit: few tiles, few pairs, few connections."""
        best_suit = 0
        best_score = None
        for s in range(3):
            base = s * 9
            count = sum(hand[base:base + 9])
            pairs = sum(1 for r in range(9) if hand[base + r] >= 2)
            conn = sum(1 for r in range(8) if hand[base + r] and hand[base + r + 1])
            score = count * 3 + pairs * 2 + conn
            if best_score is None or score < best_score:
                best_score = score
                best_suit = s
        return best_suit

    def _pick_swap_action(self, hand: list[int], actions: list[Action]) -> Action:
        """Give away the three least-connected tiles of one suit."""
        wanted = self._pick_swap(hand)
        for action in actions:
            if action.kind is ActKind.SWAP_THREE and action.tiles == wanted:
                return action
        return next(a for a in actions if a.kind is ActKind.SWAP_THREE)

    def _pick_swap(self, hand: list[int]) -> tuple[int, ...]:
        best_score = None
        best: tuple[int, ...] | None = None
        for suit in range(3):
            base = suit * 9
            pool: list[int] = []
            for rank in range(9):
                pool.extend([base + rank] * hand[base + rank])
            if len(pool) < 3:
                continue
            pool.sort(key=lambda t: self._connectivity(hand, t))
            chosen = tuple(sorted(pool[:3]))
            score = sum(self._connectivity(hand, t) for t in chosen)
            if best_score is None or score < best_score:
                best_score = score
                best = chosen
        assert best is not None
        return best

    # ------------------------------------------------------------------
    # discard / meld heuristics
    # ------------------------------------------------------------------

    def _should_peng(self, game: Game, seat: int, tile: int) -> bool:
        p = game.players[seat]
        current = shanten(p.hand, len(p.melds))
        hand = p.hand[:]
        hand[tile] -= 2
        best = 99
        for d in range(NUM_KINDS):
            if hand[d]:
                hand[d] -= 1
                best = min(best, shanten(hand, len(p.melds) + 1))
                hand[d] += 1
        return best <= current

    def _best_discard(
        self, game: Game, seat: int, discards: list[Action]
    ) -> Action:
        p = game.players[seat]
        best_key = None
        best_action = discards[0]
        for action in discards:
            t = action.tile
            assert t is not None
            p.hand[t] -= 1
            sh = shanten(p.hand, len(p.melds))
            p.hand[t] += 1
            conn = self._connectivity(p.hand, t)
            unseen = COPIES - self._visible_count(game, seat, t)
            key = (sh, conn, unseen, self.rng.random())
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

    @staticmethod
    def _connectivity(hand: list[int], tile: int) -> int:
        """How useful ``tile`` is inside ``hand`` (lower = more isolated)."""
        suit = suit_of(tile)
        base = suit * 9
        rank = tile - base
        score = hand[tile] * 2
        if rank >= 1:
            score += hand[base + rank - 1]
        if rank <= 7:
            score += hand[base + rank + 1]
        if rank >= 2:
            score += hand[base + rank - 2]
        if rank <= 6:
            score += hand[base + rank + 2]
        return score

    @staticmethod
    def _visible_count(game: Game, seat: int, tile: int) -> int:
        total = 0
        for p in game.players:
            total += sum(1 for d in p.discards if d == tile)
            total += sum(m.size for m in p.melds if m.tile == tile)
        total += game.players[seat].hand[tile]
        return total
