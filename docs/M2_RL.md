# M2：PPO 自博弈微调

> 在 M1 监督学习（[M1_TRAINING.md](M1_TRAINING.md)）基础上，用 PPO 以最终得分为奖励做自博弈微调。

## 1. 方法

```
SL checkpoint（可选先做 DAgger 聚合）
   │
   ▼
Rollout：2 个策略座 + 2 个对手座（规则 AI / --league 历史模型，每局轮换）
   │  记录 (state, mask, action, logp, value)
   ▼
按局结算：reward = 最终得分 / 50（裁剪到 ±5）
   │
   ▼
PPO update（clip=0.2，advantage 标准化，entropy 正则）
   │
   ▼
checkpoints/ppo_policy.pt
```

- 策略网络与 SL 共用（discard head + call head + value head）。
- 对手默认是规则 AI；`--league` 可叠加历史 checkpoint 组成简单对手池，
  避免策略只针对单一对手过拟合。
- 奖励为终局得分，无中间 shaping（血战计分制天然是终局信号）。

## 2. 超参（默认）

| 参数 | 值 |
|---|---|
| iterations | 15 |
| games / iter | 32 |
| opponents | 规则 AI + `sl_policy.pt`（league） |
| PPO epochs | 4 |
| minibatch | 512 |
| lr | 3e-5 |
| clip | 0.2 |
| value coef | 0.5 |
| entropy coef | 0.01 |
| reward clip | ±5（即 ±250 分） |
| grad clip | 0.5 |

## 3. 命令

```bash
python training/ppo.py --init checkpoints/sl_policy.pt \
    --out checkpoints/ppo_policy.pt --iters 15 --games 32

# 带对手池（规则 AI + 历史 checkpoint 混合）
python training/ppo.py --init checkpoints/sl_dagger.pt \
    --out checkpoints/ppo_policy.pt \
    --league checkpoints/sl_policy.pt --iters 15 --games 32

python training/evaluate.py --checkpoint checkpoints/ppo_policy.pt --games 300
```

## 4. 结果

训练规模：15 iter × 32 局 = 480 局 rollout，共 51,816 条 transition，85.5s（CPU），
对手池 = 规则 AI + `sl_policy.pt`。

| iter | 局得分/座 | 胡牌数 | policy loss | value loss | entropy |
|---|---|---|---|---|---|
| 1 | -0.20 | 23 | -0.0044 | 0.0040 | 0.165 |
| 5 | -0.67 | 26 | -0.0007 | 0.0044 | 0.154 |
| 10 | -0.95 | 16 | -0.0031 | 0.0035 | 0.154 |
| 15 | -0.14 | 23 | -0.0057 | 0.0046 | 0.150 |

- rollout 得分在 -1.5 ~ +0.4 之间波动（每 iter 32 局，噪声仍大）。
- entropy 0.165 → 0.150，策略在收敛但幅度有限。

**各模型 vs 基线（300 局，2v2 轮换）**

| 模型 | 每座得分 | 与基线差 |
|---|---|---|
| 规则 AI（锚点） | +0.565 | — |
| SL（纯 BC） | -0.565 | -1.13 |
| SL + PPO（10×24） | -0.463 | -0.927 |
| SL + DAgger | -0.345 | -0.69 |
| DAgger + PPO（15×32 + league） | -0.518 | -1.037 |

> **结论（重要）**：480 局规模的 PPO **没有带来提升，反而把 DAgger 的 -0.69 拉回到 -1.04**。
> 训练日志也出现了不稳定信号（iter 13 的 value loss 冲到 0.122）。
> 这说明当前 PPO 实现的数据量/稳定性还不足以超过行为克隆；
> 现阶段**推荐使用 `checkpoints/sl_dagger.pt`**，PPO 保留为链路与后续扩展的基础。
>
> 真正要超过规则 AI，需要「数千局/iter + GAE + KL 早停 + 学习率衰减 + 更大对手池」。

## 5. 已知局限

1. 对手只有规则 AI，没有 league，容易过拟合到单一对手。
2. 无 GAE / 折扣，直接用终局回报；长局信用分配较弱。
3. 采样为单条前向（batch=1），rollout 吞吐有限。
4. 未做 KL 早停 / 学习率衰减。

## 6. 下一步

1. 引入对手池（历史 checkpoint + 规则 AI + 随机）。
2. 改为 GAE(λ=0.95) + 折扣。
3. rollout 批量化 / 多进程。
4. 与 SL 模型做胜率矩阵，确认 RL 是否真的更强。
