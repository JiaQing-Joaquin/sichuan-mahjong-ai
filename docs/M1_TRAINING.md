# M1：特征编码与监督学习基线

> 记录「编码 → 采样 → 训练 → 评估」的实际实现与实测结果。
> 总设计见 [DESIGN.md](DESIGN.md)，RL 阶段见 [M2_RL.md](M2_RL.md)。

## 1. 链路

```
Game (规则引擎)
   │  encode_state()            ← src/sichuan_mahjong/encoder.py
   ▼
(404 维特征, 27 维出牌 mask, 4 维响应 mask)
   │  collect()                 ← training/collect.py
   ▼
data/sl.npz  (X, D, C, Y, T, R, G, S)
   │  train_sl.py               ← training/train_sl.py
   ▼
checkpoints/sl_policy.pt
   │  ModelAI                   ← training/policy_ai.py
   ▼
evaluate.py                    ← training/evaluate.py
```

## 2. 特征（404 维）

| 段 | 维度 | 内容 |
|---|---|---|
| 手牌 | 27 | 每种牌数量 / 4 |
| 自己副露-碰 | 27 | 0/1 |
| 自己副露-杠 | 27 | 0/1 |
| 四家牌河 | 108 | 4 × 27，除以 4 |
| **四家副露** | 108 | 4 × 27，除以 4 |
| 定缺 | 12 | 4 家 × 3 one-hot |
| 牌墙 | 1 | 剩余 / 55 |
| 分数 | 4 | / 50 |
| 已胡标记 | 4 | 0/1 |
| 刚摸的牌 | 27 | one-hot |
| 阶段 | 1 | 是否响应阶段 |
| **待响应的牌** | 27 | one-hot |
| **出牌者相对座位** | 4 | one-hot |
| **每种牌剩余张数** | 27 | 可见牌之外的剩余 / 4 |
| **合计** | **404** | |

**信息隔离**：只读自己的手牌 + 公开信息；`tests/test_encoder.py` 有「换掉对手手牌后特征不变」的测试。

## 3. 动作空间

| 头 | 维度 | 用途 |
|---|---|---|
| discard head | 27 | 出牌（turn 阶段） |
| call head | 4 | 胡 / 碰 / 杠 / 过（reaction 阶段） |

- 定缺 / 换三张 / 自摸 / 暗杠 / 补杠 仍由规则 AI 决策。
- 两个头都用 mask 屏蔽非法动作。

## 4. 数据格式（`data/sl.npz`）

| 字段 | 形状 | 含义 |
|---|---|---|
| `X` | (N, 404) float16 | 特征 |
| `D` | (N, 27) bool | 出牌 mask |
| `C` | (N, 4) bool | 响应 mask |
| `Y` | (N,) int64 | 出牌牌号 / 响应动作 0–3 |
| `T` | (N,) int8 | 0=出牌，1=响应 |
| `R` | (N,) float32 | 本局最终得分 / 50 |
| `G` | (N,) int32 | 局号 |
| `S` | (N,) int32 | 座位 |

只记录「有真实选择」的响应点（合法响应 ≥ 2），避免大量无意义的 pass 样本。

## 5. 命令

```bash
# 大数据集建议分片采样（每片独立进程，避免单进程内存膨胀）
python training/collect_batch.py --games 5000 --shard-size 1000 --out data/sl.npz

python training/train_sl.py --data data/sl.npz --epochs 12
python training/evaluate.py --checkpoint checkpoints/sl_policy.pt --games 300
```

## 6. 实测结果

> 由 `collect.py` / `train_sl.py` / `evaluate.py` 输出填写。

| 指标 | 值 |
|---|---|
| 采样局数 | 5000（分 5 片，约 213s） |
| 样本数（出牌 / 响应） | 264,096 / 40,123（合计 304,219） |
| 训练 / 验证 | 273,790 / 30,429（按局切分） |
| 验证一致率（出牌） | **0.779** |
| 验证一致率（响应） | **0.962** |
| 训练耗时 | 198s（12 epoch，hidden 512，CPU） |
| 模型 vs 基线（300 局） | 模型 **-0.565/座**，基线 **+0.565/座**（差 -1.13） |

**解读**

- 出牌一致率 0.779（比 1000 局版本的 0.739 有提升），响应一致率 0.962。
- 但实战对基线仍是 **-1.13 分/座**：模仿误差会在自对弈中放大——模型自己决定碰/杠/胡后，
  状态分布偏离训练数据，后续决策进入「分布外」。
- 这是行为克隆的典型问题。第 9 节的 DAgger 实验把差距从 -1.13 缩到 **-0.69**，
  验证了这个判断。

## 7. 已知限制

1. 定缺/自摸/杠仍为规则决策，模型只学出牌与响应。
2. 训练数据来自规则 AI 自对弈，天花板受规则 AI 限制。
3. 推理为单条前向（batch=1），线上需批处理。
4. 价值头仅用于辅助训练，未接入搜索。

## 8. 下一步

见 [M2_RL.md](M2_RL.md)：PPO 自博弈、对手池、GAE。

## 9. DAgger 补充实验

纯行为克隆的误差会在自对弈中累积（模型自己碰/杠/胡后进入训练集没见过的状态）。
`training/dagger.py` 的做法是：**让当前策略跑局，由规则 AI 对策略实际访问到的状态打标签**，
把新数据并入训练集后再训练。

```bash
python training/dagger.py --init checkpoints/sl_policy.pt --games 600 --out data/dagger.npz
python training/train_sl.py --data data/sl.npz,data/dagger.npz --epochs 12
python training/evaluate.py --checkpoint checkpoints/sl_policy.pt --games 300
```

| 指标 | 值 |
|---|---|
| DAgger 局数 | 600（65.3s） |
| 新增样本（出牌 / 响应） | 18,389 / 2,496（合计 20,885） |
| 验证一致率（出牌 / 响应） | 0.774 / 0.966（val loss 0.7232） |
| 对基线（300 局） | 模型 **-0.345/座**，基线 **+0.345/座**（差 **-0.69**） |

**DAgger 效果**：相比纯 BC 的 -1.13，差缩小到 -0.69（+0.44 分/座），
说明「让专家给策略实际访问到的状态打标签」确实缓解了分布偏移。
