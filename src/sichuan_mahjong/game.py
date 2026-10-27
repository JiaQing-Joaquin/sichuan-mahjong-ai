"""血战到底 game engine.

The engine is deliberately *reactive*: call :meth:`Game.legal_actions` for the
player who must act, pick one, and pass it to :meth:`Game.step`.  This makes it
usable both for rule-based agents and for reinforcement-learning rollouts.

Implemented rules
-----------------
* 108 tiles, no honors; four players.
* 定缺 (choose one suit that must be discarded before winning).
* 碰 / 杠 (明杠, 暗杠, 补杠) with replacement draws and 刮风下雨 payments.
* 血战到底: a winner leaves the round, play continues until one player is left.
* 一炮多响: several players may win on the same discard.
* 抢杠胡 is implemented for the 补杠 reaction window.
* 流局 settlement is a simplified, configurable placeholder.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hand import is_winning, shanten, winning_tiles
from .scoring import FanConfig, Meld, MeldKind, WinContext, score_win
from .tiles import COPIES, NUM_KINDS, suit_of


class Phase(str, Enum):
    SWAP = "swap"  # 换三张 (optional opening phase)
    MISSING = "missing"  # players choose their missing suit
    TURN = "turn"  # current player draws/acts and discards
    REACTION = "reaction"  # others respond to a discard
    OVER = "over"


class ActKind(str, Enum):
    SWAP_THREE = "swap_three"  # 换三张
    CHOOSE_MISSING = "choose_missing"
    DISCARD = "discard"
    HU_SELF = "hu_self"  # 自摸
    HU_DISCARD = "hu_discard"  # 点炮 / 抢杠
    ANKANG = "ankang"  # 暗杠
    BUKANG = "bukang"  # 补杠
    MINGGANG = "minggang"  # 明杠 (on a discard)
    PENG = "peng"  # 碰
    PASS = "pass"


@dataclass(frozen=True)
class Action:
    kind: ActKind
    tile: int | None = None
    suit: int | None = None
    tiles: tuple[int, ...] | None = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if self.tiles is not None:
            return f"{self.kind.value}(tiles={self.tiles})"
        if self.suit is not None:
            return f"{self.kind.value}(suit={self.suit})"
        if self.tile is not None:
            return f"{self.kind.value}({self.tile})"
        return self.kind.value


@dataclass
class Player:
    seat: int
    hand: list[int] = field(default_factory=lambda: [0] * NUM_KINDS)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    missing_suit: int | None = None
    won: bool = False
    score: int = 0
    win_item: str | None = None
    win_fan: int | None = None
    win_points: int | None = None
    just_gang_draw: bool = False
    last_draw: int | None = None


@dataclass
class GameConfig:
    dealer_seat: int = 0
    #: enable the 换三张 opening phase
    swap_three: bool = False
    #: receiver offset: 1 = pass to the next seat clockwise
    swap_direction: int = 1
    #: must discard every tile of the missing suit before anything else
    must_discard_missing: bool = True
    #: points paid per gang (刮风下雨); set all to 0 to disable
    gang_payment: dict[str, int] = field(
        default_factory=lambda: {"ming": 1, "an": 2, "bu": 1}
    )
    #: simplified 流局 penalty, 0 disables settlement
    draw_penalty: int = 0


@dataclass
class _Reaction:
    discard: int
    discarder: int
    order: list[int]
    idx: int = 0
    claims: dict[int, Action] = field(default_factory=dict)
    discard_after_gang: bool = False
    is_qiang_gang: bool = False


class Game:
    def __init__(
        self,
        seed: int | None = None,
        config: GameConfig | None = None,
        fan_config: FanConfig | None = None,
    ) -> None:
        self.cfg = config or GameConfig()
        self.fan_config = fan_config or FanConfig()
        self.rng = random.Random(seed)
        self.players = [Player(i) for i in range(4)]
        self.dealer = self.cfg.dealer_seat % 4
        self.wall: list[int] = []
        self.phase: Phase = Phase.MISSING
        self.turn = self.dealer
        self.history: list[tuple[int, Action]] = []
        self._missing_queue: list[int] = []
        self._missing_idx = 0
        self._reaction: Optional[_Reaction] = None
        self._must_discard_only = False
        self._discard_count = 0
        self._reset()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        tiles = [t for t in range(NUM_KINDS) for _ in range(COPIES)]
        self.rng.shuffle(tiles)
        for p in self.players:
            p.hand = [0] * NUM_KINDS
            p.melds = []
            p.discards = []
            p.missing_suit = None
            p.won = False
            p.score = 0
            p.win_item = None
            p.win_fan = None
            p.win_points = None
            p.just_gang_draw = False
            p.last_draw = None
        for _ in range(13):
            for p in self.players:
                p.hand[tiles.pop()] += 1
        self.players[self.dealer].hand[tiles.pop()] += 1  # dealer's 14th tile
        self.wall = tiles
        self._swap_queue = [(self.dealer + k) % 4 for k in range(4)]
        self._swap_idx = 0
        self._swap_selections: dict[int, list[int]] = {}
        self._missing_queue = [(self.dealer + k) % 4 for k in range(4)]
        self._missing_idx = 0
        self.phase = Phase.SWAP if self.cfg.swap_three else Phase.MISSING
        self.turn = self.dealer
        self._reaction = None
        self._must_discard_only = False
        self._discard_count = 0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int | None:
        if self.phase is Phase.SWAP:
            return self._swap_queue[self._swap_idx]
        if self.phase is Phase.MISSING:
            return self._missing_queue[self._missing_idx]
        if self.phase is Phase.TURN:
            return self.turn
        if self.phase is Phase.REACTION and self._reaction is not None:
            r = self._reaction
            return r.order[r.idx] if r.idx < len(r.order) else None
        return None

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.OVER

    def active_players(self) -> list[Player]:
        return [p for p in self.players if not p.won]

    @property
    def pending_discard(self) -> int | None:
        """The tile currently being reacted to, if any."""
        return self._reaction.discard if self._reaction is not None else None

    @property
    def pending_discarder(self) -> int | None:
        """Seat of the player who discarded the pending tile, if any."""
        return self._reaction.discarder if self._reaction is not None else None

    def _is_missing(self, p: Player, tile: int) -> bool:
        return p.missing_suit is not None and suit_of(tile) == p.missing_suit

    # ------------------------------------------------------------------
    # legal actions
    # ------------------------------------------------------------------

    def legal_actions(self) -> list[Action]:
        if self.phase is Phase.SWAP:
            return self._swap_actions()
        if self.phase is Phase.MISSING:
            return [Action(ActKind.CHOOSE_MISSING, suit=s) for s in range(3)]
        if self.phase is Phase.TURN:
            return self._turn_actions(self.turn)
        if self.phase is Phase.REACTION:
            return self._reaction_actions()
        return []

    def _turn_actions(self, seat: int) -> list[Action]:
        p = self.players[seat]
        opts: list[Action] = []
        if not self._must_discard_only:
            if is_winning(p.hand, len(p.melds), p.missing_suit):
                opts.append(Action(ActKind.HU_SELF))
            for t in range(NUM_KINDS):
                if p.hand[t] == COPIES and not self._is_missing(p, t):
                    opts.append(Action(ActKind.ANKANG, tile=t))
            for meld in p.melds:
                if (
                    meld.kind is MeldKind.PENG
                    and p.hand[meld.tile] >= 1
                    and not self._is_missing(p, meld.tile)
                ):
                    opts.append(Action(ActKind.BUKANG, tile=meld.tile))

        missing = [t for t in range(NUM_KINDS) if p.hand[t] and self._is_missing(p, t)]
        if self.cfg.must_discard_missing and missing:
            tiles = missing
        else:
            tiles = [t for t in range(NUM_KINDS) if p.hand[t]]
        for t in tiles:
            opts.append(Action(ActKind.DISCARD, tile=t))
        return opts

    def _reaction_actions(self) -> list[Action]:
        assert self._reaction is not None
        r = self._reaction
        seat = r.order[r.idx]
        tile = r.discard
        opts: list[Action] = [Action(ActKind.PASS)]
        if self._can_hu_discard(seat, tile):
            opts.append(Action(ActKind.HU_DISCARD))
        if r.is_qiang_gang:
            # 抢杠胡: only a win (or pass) may interrupt an added kong
            return opts
        if self._can_minggang(seat, tile):
            opts.append(Action(ActKind.MINGGANG, tile=tile))
        if self._can_peng(seat, tile):
            opts.append(Action(ActKind.PENG, tile=tile))
        return opts

    # -- 换三张 --------------------------------------------------------

    def _swap_actions(self) -> list[Action]:
        seat = self._swap_queue[self._swap_idx]
        hand = self.players[seat].hand
        opts: list[Action] = []
        seen: set[tuple[int, ...]] = set()
        for suit in range(3):
            base = suit * 9
            pool: list[int] = []
            for rank in range(9):
                pool.extend([base + rank] * hand[base + rank])
            if len(pool) < 3:
                continue
            for combo in itertools.combinations(pool, 3):
                if combo in seen:
                    continue
                seen.add(combo)
                opts.append(Action(ActKind.SWAP_THREE, tiles=combo))
        return opts

    def _do_swap(self, seat: int, action: Action) -> None:
        assert action.tiles is not None
        self._swap_selections[seat] = list(action.tiles)
        self._swap_idx += 1
        if self._swap_idx >= len(self._swap_queue):
            self._perform_swap()

    def _perform_swap(self) -> None:
        direction = self.cfg.swap_direction % 4
        for seat in range(4):
            given = self._swap_selections[seat]
            giver = self.players[seat]
            for tile in given:
                giver.hand[tile] -= 1
            receiver = self.players[(seat + direction) % 4]
            for tile in given:
                receiver.hand[tile] += 1
        self.phase = Phase.MISSING
        self._missing_idx = 0

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> None:
        seat = self.current_player
        if seat is None:
            raise RuntimeError("game is over")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action} for seat {seat}")
        self.history.append((seat, action))

        if self.phase is Phase.SWAP:
            self._do_swap(seat, action)
        elif self.phase is Phase.MISSING:
            self._do_choose_missing(seat, action)
        elif self.phase is Phase.TURN:
            self._do_turn(seat, action)
        elif self.phase is Phase.REACTION:
            self._do_reaction(seat, action)

    # -- 定缺 ----------------------------------------------------------

    def _do_choose_missing(self, seat: int, action: Action) -> None:
        assert action.suit is not None
        self.players[seat].missing_suit = action.suit
        self._missing_idx += 1
        if self._missing_idx >= len(self._missing_queue):
            self.phase = Phase.TURN
            self.turn = self.dealer
            self._must_discard_only = False

    # -- own turn ------------------------------------------------------

    def _do_turn(self, seat: int, action: Action) -> None:
        p = self.players[seat]
        kind = action.kind

        if kind is ActKind.HU_SELF:
            flags = {"gangshang_hua"} if p.just_gang_draw else set()
            if self._discard_count == 0 and seat == self.dealer:
                flags.add("tian_hu")
            p.just_gang_draw = False
            self._settle_win(seat, self._last_draw(seat), self_draw=True, flags=flags)
            self._finish_after_win(seat)
            return

        if kind is ActKind.ANKANG:
            t = action.tile
            assert t is not None
            p.hand[t] -= 4
            p.melds.append(Meld(MeldKind.GANG, t))
            self._pay_gang(seat, "an")
            self._draw_replacement(seat)
            return

        if kind is ActKind.BUKANG:
            t = action.tile
            assert t is not None
            order = [(seat + k) % 4 for k in (1, 2, 3)]
            order = [s for s in order if not self.players[s].won]
            if any(self._can_hu_discard(s, t) for s in order):
                # 抢杠: give others a chance to rob the kong before it happens
                self._reaction = _Reaction(
                    discard=t,
                    discarder=seat,
                    order=order,
                    is_qiang_gang=True,
                )
                self.phase = Phase.REACTION
                return
            self._apply_bukang(seat, t)
            return

        if kind is ActKind.DISCARD:
            t = action.tile
            assert t is not None
            was_gang_draw = p.just_gang_draw
            p.hand[t] -= 1
            p.discards.append(t)
            p.just_gang_draw = False
            p.last_draw = None
            self._discard_count += 1
            self._begin_reaction(seat, t, discard_after_gang=was_gang_draw)
            return

        raise AssertionError(f"unexpected turn action {action}")

    def _last_draw(self, seat: int) -> int:
        """The winning tile for a self-draw (used for display / 番型 only)."""
        p = self.players[seat]
        candidates = [t for t in range(NUM_KINDS) if p.hand[t]]
        for t in candidates:
            p.hand[t] -= 1
            ok = shanten(p.hand, len(p.melds)) == 0
            p.hand[t] += 1
            if ok:
                return t
        return candidates[0]

    # -- reactions -----------------------------------------------------

    def _begin_reaction(self, discarder: int, tile: int, discard_after_gang: bool = False) -> None:
        order = [(discarder + k) % 4 for k in (1, 2, 3)]
        order = [s for s in order if not self.players[s].won]
        self._reaction = _Reaction(
            discard=tile,
            discarder=discarder,
            order=order,
            discard_after_gang=discard_after_gang,
        )
        if not order:
            self._advance_after_discard(discarder)
            return
        self.phase = Phase.REACTION

    def _do_reaction(self, seat: int, action: Action) -> None:
        assert self._reaction is not None
        r = self._reaction
        r.claims[seat] = action
        r.idx += 1
        if r.idx >= len(r.order):
            self._resolve_reaction()

    def _resolve_reaction(self) -> None:
        assert self._reaction is not None
        r = self._reaction
        self._reaction = None

        if r.is_qiang_gang:
            hu = [
                s
                for s in r.order
                if r.claims.get(s, Action(ActKind.PASS)).kind is ActKind.HU_DISCARD
            ]
            if hu:
                # the kong is cancelled: the konger loses the 4th tile, keeps the peng
                konger = self.players[r.discarder]
                if konger.hand[r.discard] > 0:
                    konger.hand[r.discard] -= 1
                for s in hu:
                    self._claim_for_win(s, r.discard, None)
                    self._settle_win(
                        s,
                        r.discard,
                        self_draw=False,
                        discarder=r.discarder,
                        flags={"qiang_gang"},
                    )
                self._check_over()
                if self.phase is not Phase.OVER:
                    self._advance_after_discard(r.discarder)
                return
            # nobody robbed it -> the added kong proceeds
            self._apply_bukang(r.discarder, r.discard)
            return

        hu = [s for s in r.order if r.claims.get(s, Action(ActKind.PASS)).kind is ActKind.HU_DISCARD]
        if hu:
            flags = {"gangshang_pao"} if r.discard_after_gang else set()
            for s in hu:
                self._claim_for_win(s, r.discard, r.discarder)
                self._settle_win(s, r.discard, self_draw=False, discarder=r.discarder, flags=set(flags))
            self._check_over()
            if self.phase is not Phase.OVER:
                self._advance_after_discard(r.discarder)
            return

        ming = [s for s in r.order if r.claims.get(s, Action(ActKind.PASS)).kind is ActKind.MINGGANG]
        peng = [s for s in r.order if r.claims.get(s, Action(ActKind.PASS)).kind is ActKind.PENG]

        if ming:
            s = ming[0]
            self._apply_minggang(s, r.discard, r.discarder)
            self._check_over()
            if self.phase is not Phase.OVER:
                self._draw_replacement(s)
            return
        if peng:
            s = peng[0]
            self._apply_peng(s, r.discard, r.discarder)
            self.phase = Phase.TURN
            self.turn = s
            self._must_discard_only = True
            return

        self._advance_after_discard(r.discarder)

    def _can_hu_discard(self, seat: int, tile: int) -> bool:
        p = self.players[seat]
        if p.won or self._is_missing(p, tile):
            return False
        p.hand[tile] += 1
        ok = is_winning(p.hand, len(p.melds), p.missing_suit)
        p.hand[tile] -= 1
        return ok

    def _can_peng(self, seat: int, tile: int) -> bool:
        p = self.players[seat]
        return (
            not p.won
            and not self._is_missing(p, tile)
            and p.hand[tile] >= 2
        )

    def _can_minggang(self, seat: int, tile: int) -> bool:
        p = self.players[seat]
        return (
            not p.won
            and not self._is_missing(p, tile)
            and p.hand[tile] >= 3
        )

    # -- meld application ---------------------------------------------

    def _apply_bukang(self, seat: int, tile: int) -> None:
        p = self.players[seat]
        p.hand[tile] -= 1
        for i, meld in enumerate(p.melds):
            if meld.kind is MeldKind.PENG and meld.tile == tile:
                p.melds[i] = Meld(MeldKind.GANG, tile)
                break
        self._pay_gang(seat, "bu")
        self._draw_replacement(seat)

    def _apply_peng(self, seat: int, tile: int, discarder: int) -> None:
        p = self.players[seat]
        p.hand[tile] -= 2
        p.melds.append(Meld(MeldKind.PENG, tile))
        p.just_gang_draw = False
        p.last_draw = None
        self._remove_discard(discarder, tile)

    def _apply_minggang(self, seat: int, tile: int, discarder: int) -> None:
        p = self.players[seat]
        p.hand[tile] -= 3
        p.melds.append(Meld(MeldKind.GANG, tile))
        self._remove_discard(discarder, tile)
        self._pay_gang(seat, "ming", from_seat=discarder)
        p.just_gang_draw = True

    def _remove_discard(self, seat: int, tile: int) -> None:
        discards = self.players[seat].discards
        for i in range(len(discards) - 1, -1, -1):
            if discards[i] == tile:
                del discards[i]
                return

    def _pay_gang(self, seat: int, kind: str, from_seat: int | None = None) -> None:
        amount = self.cfg.gang_payment.get(kind, 0)
        if amount <= 0:
            return
        konger = self.players[seat]
        if from_seat is not None:
            self.players[from_seat].score -= amount
            konger.score += amount
            return
        for q in self.players:
            if q.seat != seat and not q.won:
                q.score -= amount
                konger.score += amount

    def _draw_replacement(self, seat: int) -> None:
        if not self._draw(seat):
            return
        p = self.players[seat]
        p.just_gang_draw = True
        self.phase = Phase.TURN
        self.turn = seat
        self._must_discard_only = False

    # -- drawing -------------------------------------------------------

    def _draw(self, seat: int) -> bool:
        if not self.wall:
            self._settle_draw()
            return False
        tile = self.wall.pop()
        self.players[seat].hand[tile] += 1
        self.players[seat].last_draw = tile
        return True

    def _begin_turn(self, seat: int) -> None:
        self.turn = seat
        self.players[seat].just_gang_draw = False
        self._must_discard_only = False
        if not self._draw(seat):
            return
        self.phase = Phase.TURN

    def _next_active(self, seat: int) -> int | None:
        for k in range(1, 5):
            cand = (seat + k) % 4
            if not self.players[cand].won:
                return cand
        return None

    def _advance_after_discard(self, discarder: int) -> None:
        nxt = self._next_active(discarder)
        if nxt is None:
            self.phase = Phase.OVER
            return
        self._begin_turn(nxt)

    # -- winning -------------------------------------------------------

    def _claim_for_win(self, seat: int, tile: int, discarder: int | None) -> None:
        self.players[seat].hand[tile] += 1
        if discarder is not None:
            self._remove_discard(discarder, tile)

    def _settle_win(
        self,
        seat: int,
        tile: int,
        self_draw: bool,
        discarder: int | None = None,
        flags: set[str] | None = None,
    ) -> None:
        p = self.players[seat]
        ctx = WinContext(
            counts=p.hand[:],
            melds=list(p.melds),
            win_tile=tile,
            self_draw=self_draw,
            flags=set(flags or ()),
            missing_suit=p.missing_suit,
        )
        result = score_win(ctx, self.fan_config)

        if self_draw:
            payers = [q for q in self.players if q.seat != seat and not q.won]
        else:
            assert discarder is not None
            payers = [self.players[discarder]]
        for q in payers:
            q.score -= result.points
            p.score += result.points

        p.won = True
        p.win_item = "+".join(result.items)
        p.win_fan = result.fan
        p.win_points = result.points

    def _finish_after_win(self, seat: int) -> None:
        self._check_over()
        if self.phase is Phase.OVER:
            return
        # play resumes from the next active player after the winner's seat
        nxt = self._next_active(seat)
        if nxt is None:
            self.phase = Phase.OVER
            return
        self._begin_turn(nxt)

    def _check_over(self) -> None:
        if len(self.active_players()) <= 1:
            self.phase = Phase.OVER

    def _settle_draw(self) -> None:
        """Simplified 流局 settlement (查花猪 / 查大叫)."""
        self.phase = Phase.OVER
        penalty = self.cfg.draw_penalty
        if penalty <= 0:
            return
        active = self.active_players()
        huazhu: list[Player] = []
        tenpai: list[Player] = []
        for p in active:
            has_missing = (
                p.missing_suit is not None
                and any(p.hand[p.missing_suit * 9 + r] for r in range(9))
            )
            if has_missing:
                huazhu.append(p)
            elif winning_tiles(p.hand, len(p.melds), p.missing_suit):
                tenpai.append(p)
        for p in active:
            if p in huazhu:
                for q in active:
                    if q is not p:
                        p.score -= penalty * 2
                        q.score += penalty * 2
            elif p not in tenpai:
                for q in tenpai:
                    p.score -= penalty
                    q.score += penalty

    # ------------------------------------------------------------------
    # result
    # ------------------------------------------------------------------

    def result(self) -> list[tuple[int, int, bool, str | None, int | None]]:
        return [
            (p.seat, p.score, p.won, p.win_item, p.win_fan) for p in self.players
        ]

    def scores(self) -> list[int]:
        return [p.score for p in self.players]

    # ------------------------------------------------------------------
    # views for agents
    # ------------------------------------------------------------------

    def view(self, seat: int) -> dict:
        """Information visible to ``seat`` (own hand + all public state)."""
        me = self.players[seat]
        return {
            "seat": seat,
            "hand": me.hand[:],
            "melds": [(m.kind.value, m.tile) for m in me.melds],
            "discards": [p.discards[:] for p in self.players],
            "missing": [p.missing_suit for p in self.players],
            "won": [p.won for p in self.players],
            "wall_left": len(self.wall),
            "phase": self.phase.value,
        }
