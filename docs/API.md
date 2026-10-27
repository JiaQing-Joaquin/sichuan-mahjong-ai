# API 速查

## 安装

```bash
pip install -e .            # 核心引擎（零第三方依赖）
pip install -e ".[train]"   # 额外安装 numpy + torch
```

## 1. 跑一局（规则 AI 自对弈）

```python
from sichuan_mahjong.game import Game
from sichuan_mahjong.baseline import BaselineAI

game = Game(seed=42)
agents = [BaselineAI(seed=i) for i in range(4)]

while not game.is_over:
    seat = game.current_player
    game.step(agents[seat].act(game))

print(game.result())          # [(seat, score, won, items, fan), ...]
assert sum(game.scores()) == 0
```

## 2. `Game` 接口

| 成员 | 说明 |
|---|---|
| `Game(seed=None, config=GameConfig(), fan_config=FanConfig())` | 新建对局 |
| `current_player -> int \| None` | 当前需要决策的座位 |
| `phase -> Phase` | `SWAP` / `MISSING` / `TURN` / `REACTION` / `OVER` |
| `is_over -> bool` | 是否结束 |
| `legal_actions() -> list[Action]` | 当前玩家的合法动作 |
| `step(action)` | 执行动作；非法动作抛 `ValueError` |
| `scores() / result()` | 结算结果 |
| `view(seat) -> dict` | 该座位可见的信息（不含对手暗牌） |
| `pending_discard / pending_discarder` | 响应阶段待处理的牌与出牌者 |

`GameConfig` 可配置：`swap_three`、`swap_direction`、`must_discard_missing`、
`gang_payment`（刮风下雨）、`draw_penalty`（流局）。

## 3. 动作

`Action(kind, tile=None, suit=None, tiles=None)`

| `ActKind` | 参数 |
|---|---|
| `SWAP_THREE` | `tiles=(t1,t2,t3)` |
| `CHOOSE_MISSING` | `suit=0/1/2` |
| `DISCARD` | `tile` |
| `HU_SELF` / `HU_DISCARD` / `PASS` | — |
| `ANKANG` / `BUKANG` / `MINGGANG` / `PENG` | `tile` |

## 4. 手牌分析

```python
from sichuan_mahjong.hand import shanten, winning_tiles, is_winning
from sichuan_mahjong.tiles import parse_tiles, counts_from_tiles

counts = counts_from_tiles(parse_tiles("123m 456m 789m 123p 5p"))
shanten(counts)              # 0 = 听牌
winning_tiles(counts)        # [13] 即 5筒
```

## 5. 计分

```python
from sichuan_mahjong.scoring import FanConfig, WinContext, score_win

ctx = WinContext(
    counts=counts_from_tiles(parse_tiles("123m 456m 789m 123p 55p")),
    melds=[], win_tile=13, self_draw=False, missing_suit=2,
)
result = score_win(ctx, FanConfig())
print(result.fan, result.items, result.points)
```

## 6. 编码器（模型输入）

```python
from sichuan_mahjong.encoder import encode, STATE_DIM

feats, discard_mask, call_mask = encode(game, seat)
len(feats) == STATE_DIM      # 404
```

## 7. 训练链路

| 脚本 | 作用 |
|---|---|
| `training/collect.py` | 单进程采样 |
| `training/collect_batch.py` | 分片采样 + 合并（大数据集） |
| `training/dagger.py` | DAgger：策略跑局 + 专家给策略访问到的状态打标签 |
| `training/train_sl.py` | 监督学习（`--data a,b` 可合并多份数据） |
| `training/policy_ai.py` | 模型推理 Agent |
| `training/evaluate.py` | 模型 vs 基线 |
| `training/ppo.py` | PPO 自博弈微调（`--league` 对手池） |

```python
from training.policy_ai import ModelAI

ai = ModelAI("checkpoints/ppo_policy.pt")
game = Game(seed=1)
while not game.is_over:
    game.step(ai.act(game))
```

## 8. 规则配置示例

```python
from sichuan_mahjong.game import Game, GameConfig

game = Game(
    seed=7,
    config=GameConfig(
        swap_three=True,          # 开局换三张
        swap_direction=1,         # 传给下家
        gang_payment={"ming": 1, "an": 2, "bu": 1},
        draw_penalty=1,           # 流局查叫（简化版）
    ),
)
```
