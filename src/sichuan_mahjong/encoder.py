"""State encoder for the policy network (v2).

Encodes everything a real player can see:

* own hand + own melds
* every player's discards and exposed melds
* every player's declared missing suit (定缺)
* wall size, scores, whether each player has already won
* the tile just drawn
* when reacting: the pending discard tile and who discarded it

Opponent concealed hands are **never** read.

Feature layout (404 dims)::

    [  0: 27)  own hand counts / 4
    [ 27: 54)  own peng mask
    [ 54: 81)  own gang mask
    [ 81:189)  4 players' discards / 4
    [189:297)  4 players' meld tiles / 4
    [297:309)  4 players' missing suit (one-hot x3)
    [309:310)  wall remaining / 55
    [310:314)  scores / 50
    [314:318)  won flags
    [318:345)  last drawn tile one-hot
    [345:346)  reaction flag
    [346:373)  pending discard tile one-hot
    [373:377)  discarder relative seat one-hot
    [377:404)  remaining copies of each tile / 4
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .game import ActKind, Game, Phase
from .tiles import COPIES, NUM_KINDS

STATE_DIM = 404
DISCARD_DIM = NUM_KINDS
CALL_DIM = 4

#: order of the call head: 胡 / 碰 / 杠 / 过
CALL_KINDS = (
    ActKind.HU_DISCARD,
    ActKind.PENG,
    ActKind.MINGGANG,
    ActKind.PASS,
)

_WALL_NORM = 55.0
_SCORE_NORM = 50.0


def _counts_of(tiles: Sequence[int]) -> List[int]:
    out = [0] * NUM_KINDS
    for tile in tiles:
        out[tile] += 1
    return out


def encode_state(game: Game, seat: int) -> List[float]:
    me = game.players[seat]
    feats: List[float] = []

    # 0) own hand
    feats.extend(count / COPIES for count in me.hand)

    # own melds: peng mask, gang mask
    peng = [0.0] * NUM_KINDS
    gang = [0.0] * NUM_KINDS
    for meld in me.melds:
        if meld.kind.value == "gang":
            gang[meld.tile] = 1.0
        else:
            peng[meld.tile] = 1.0
    feats.extend(peng)
    feats.extend(gang)

    # 1) discards of all four players
    for player in game.players:
        feats.extend(c / COPIES for c in _counts_of(player.discards))

    # 2) exposed meld tiles of all four players (one 27-vector each)
    meld_tiles = [0] * NUM_KINDS
    for player in game.players:
        player_melds = [0] * NUM_KINDS
        for meld in player.melds:
            player_melds[meld.tile] += meld.size
            meld_tiles[meld.tile] += meld.size
        feats.extend(c / COPIES for c in player_melds)

    # 3) missing suit for all four players
    for player in game.players:
        one_hot = [0.0, 0.0, 0.0]
        if player.missing_suit is not None:
            one_hot[player.missing_suit] = 1.0
        feats.extend(one_hot)

    # 4) wall size
    feats.append(len(game.wall) / _WALL_NORM)

    # 5) scores
    for player in game.players:
        feats.append(player.score / _SCORE_NORM)

    # 6) already-won flags
    for player in game.players:
        feats.append(1.0 if player.won else 0.0)

    # 7) tile just drawn
    last_draw = [0.0] * NUM_KINDS
    if me.last_draw is not None:
        last_draw[me.last_draw] = 1.0
    feats.extend(last_draw)

    # 8) phase flag
    reacting = game.phase is Phase.REACTION
    feats.append(1.0 if reacting else 0.0)

    # 9) pending discard tile
    pending = game.pending_discard
    pending_one_hot = [0.0] * NUM_KINDS
    if pending is not None:
        pending_one_hot[pending] = 1.0
    feats.extend(pending_one_hot)

    # 10) who discarded it, relative to us
    relative = [0.0] * 4
    discarder = game.pending_discarder
    if discarder is not None:
        relative[(discarder - seat) % 4] = 1.0
    feats.extend(relative)

    # 11) remaining copies of each tile (visible = hand + discards + melds)
    visible = [0] * NUM_KINDS
    for player in game.players:
        for tile in player.discards:
            visible[tile] += 1
    for tile in range(NUM_KINDS):
        visible[tile] += meld_tiles[tile] + me.hand[tile]
    feats.extend(max(0, COPIES - visible[tile]) / COPIES for tile in range(NUM_KINDS))

    assert len(feats) == STATE_DIM, f"encoder drift: {len(feats)} != {STATE_DIM}"
    return feats


def discard_mask(game: Game) -> List[bool]:
    mask = [False] * NUM_KINDS
    for action in game.legal_actions():
        if action.kind is ActKind.DISCARD and action.tile is not None:
            mask[action.tile] = True
    return mask


def call_mask(game: Game) -> List[bool]:
    present = {action.kind for action in game.legal_actions()}
    return [kind in present for kind in CALL_KINDS]


def call_index(kind: ActKind) -> int:
    return CALL_KINDS.index(kind)


def encode(game: Game, seat: int) -> Tuple[List[float], List[bool], List[bool]]:
    """Return ``(features, discard_mask, call_mask)`` for ``seat``."""
    return encode_state(game, seat), discard_mask(game), call_mask(game)


def legal_discard_tiles(mask: Sequence[bool]) -> List[int]:
    return [tile for tile, allowed in enumerate(mask) if allowed]
