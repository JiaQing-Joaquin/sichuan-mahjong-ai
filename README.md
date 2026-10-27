# 川麻血战到底 AI

四川麻将「血战到底」的规则引擎与 AI 训练工程。核心引擎零第三方依赖；
训练链路需要 PyTorch。

> 目标：作为商业对战产品的 AI 底座。
> 已完成：规则引擎 + 规则 AI + 特征编码 + 监督学习（SL）+ PPO 自博弈链路。

---

## 特性

- ✅ 完整牌型工具：牌编码、解析、显示（`1m` / `1万` / `123m` 等格式）
- ✅ 向听数（标准型 + 七对/龙七对）、胡牌判定、听牌与有效张
- ✅ 缺一门、碰/杠（明杠/暗杠/补杠）、**抢杠胡**、**换三张**、一炮多响、血战续打
- ✅ 可配置番型计分（平胡/对对胡/清一色/清对/将对/七对/龙七对/金钩钓/根/杠上花…）
- ✅ 规则 AI（向听 + 有效张 + 简单安全牌）
- ✅ 特征编码器（404 维）+ 出牌/响应双动作头
- ✅ SL 训练链路：自对弈采样 → 训练 → 对基线评估
- ✅ PPO 自博弈微调链路
- ⏳ 真实流局结算（查花猪/查大叫，当前为简化可配置版）

## 快速开始

要求：Python ≥ 3.9（引擎与测试）；训练需 PyTorch。

```bash
# 1. 运行全部测试（450 项核心与回放；装了 torch 再加 30 项训练/DAgger/PPO/接口冒烟）
python scripts/run_tests.py
#    Windows 也可以直接双击 test.bat（会暂停显示结果）

# 2. 规则 AI vs 随机 AI 的 A/B 评估
python scripts/eval.py 300

# 3. 单座位规则 AI 自博弈统计
python scripts/simulate.py 200

# 4. 向听性能基准
python scripts/bench.py 2000
```

### 训练（需要 torch）

```bash
# 用带 torch 的解释器，例如 Anaconda 环境：
#   D:\Anaconda3\envs\tensorflow\python.exe

# 1) 自对弈采样（约 3-5 分钟 / 5000 局）
python training/collect.py --games 5000 --out data/sl.npz

# 2) 监督学习
python training/train_sl.py --data data/sl.npz --epochs 12

# 3) 对基线评估
python training/evaluate.py --checkpoint checkpoints/sl_policy.pt --games 300

# 4) PPO 自博弈微调
python training/ppo.py --init checkpoints/sl_policy.pt \
    --out checkpoints/ppo_policy.pt --iters 15 --games 32

# 5) 评估 PPO 模型
python training/evaluate.py --checkpoint checkpoints/ppo_policy.pt --games 300
```

细节见 [M1 训练链路](docs/M1_TRAINING.md) 与 [M2 PPO](docs/M2_RL.md)。

## 目录结构

```
src/sichuan_mahjong/     核心库（零第三方依赖）
  tiles.py               牌编码 / 解析 / 显示
  hand.py                向听、胡牌、听牌、有效张
  scoring.py             番型计分（FanConfig 可配置）
  game.py                血战到底状态机（含抢杠胡/换三张）
  baseline.py            规则 AI
  encoder.py             状态特征 + 出牌/响应 mask
  py.typed

training/                训练链路（需要 torch）
  collect.py             基线自对弈采样
  collect_batch.py       分片采样 + 合并（大数据集）
  dagger.py              DAgger：策略跑局 + 专家打标签
  model.py               PyTorch 多任务网络
  train_sl.py            监督学习（可合并多份数据）
  policy_ai.py           模型推理（出牌/响应用模型，其余用规则）
  evaluate.py            模型 vs 基线
  ppo.py                 PPO 自博弈（支持 --league 对手池）

tests/                   单元测试（450 项核心/回放 + 30 项训练/DAgger/PPO/接口冒烟）
scripts/                 开发脚本
  run_tests.py           测试入口
  eval.py                A/B 评估
  simulate.py            自博弈统计
  bench.py               性能基准

docs/
  DESIGN.md              技术方案设计
  M1_TRAINING.md         M1 实现与实测
  M2_RL.md               PPO 设计与实测
  RESEARCH.md            开源项目与许可证调研

configs/                 规则/训练配置（预留）
.github/workflows/ci.yml CI
```

## 规则覆盖

| 规则 | 状态 |
|---|---|
| 108 张、无字牌、四人 | ✅ |
| 定缺（缺一门） | ✅ |
| 不能吃，只能碰/杠 | ✅ |
| 明杠 / 暗杠 / 补杠 + 补牌 | ✅ |
| 血战到底（胡牌者退出，续打到剩一家） | ✅ |
| 一炮多响 | ✅ |
| 抢杠胡 | ✅ |
| 换三张（可选开局阶段） | ✅ |
| 番型计分（可配置） | ✅ |
| 流局查花猪/查大叫 | ⏳ 简化版 |
| 连庄 / 多局积分 | ⏳ 未实现 |

## 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 规则引擎 + 规则 AI + 测试 | ✅ |
| M0.5 | 抢杠胡、换三张、特征编码 | ✅ |
| M1 | SL 训练链路与评估 | ✅ |
| M2 | PPO 自博弈 | ✅（首版） |
| M3 | 服务化、灰度、监控 | ⏳ |

具体指标见 [M1_TRAINING.md](docs/M1_TRAINING.md) / [M2_RL.md](docs/M2_RL.md)。

## 实验结果（对基线，300 局，2v2 轮换，每座得分差）

| 模型 | 每座得分差 |
|---|---|
| SL（纯行为克隆） | -1.13 |
| SL + PPO（240 局 rollout） | -0.93 |
| SL + DAgger（600 局） | **-0.69（当前最佳）** |
| DAgger + PPO（480 局 + 对手池） | -1.04（退化） |

> 现阶段推荐加载 `checkpoints/sl_dagger.pt`；PPO 链路已跑通，但训练量不足以超过 DAgger。

## 初始版交付边界

- 可直接交付本地规则对局、规则 AI、已保存模型推理、训练脚本和 2v2 评测入口。`test.bat` 使用带 PyTorch 的本机环境运行全套测试；`run_eval.bat` 运行规则 AI 对随机策略的 60 局冒烟评估。
- 流局赔偿仍是简化规则；不同平台的番型、杠分和结算细节需要真实牌谱对账。现有本机 HTTP 接口仅供集成验证；鉴权、持久化、房间管理和多局积分尚未实现。
- 历史模型强度数字缺少逐局日志与置信区间。初始版可用于研究验证和集成原型，不能据此承诺上线对战强度；后续评测应固定规则配置、种子和检查点，并保存逐局四座得分。

### 可复现评测与计分回放

```bash
python training/evaluate.py --checkpoint checkpoints/sl_dagger.pt --games 300 --seeds 100,1100,2100 --log runs/eval.jsonl
python scripts/replay.py runs/eval.jsonl
```

评测保存每局种子、座位角色、规则配置、检查点 SHA-256、Git 修订、动作与逐步分数，并输出按局配对的分差及 bootstrap 区间。`--log` 要求目标文件不存在，以免覆盖旧实验；可用 `--game-config`、`--fan-config` 提供 JSON 规则配置。回放逐步核对合法动作、零和分数及终局结果。字段和限制见 [评测与回放](docs/EVALUATION.md)。

### 同种子模型比较

```bash
python scripts/compare_checkpoints.py --checkpoints checkpoints/sl_policy.pt,checkpoints/sl_dagger.pt,checkpoints/ppo_policy.pt --games 100 --seeds 100,1100,2100 --out runs/compare_001
```

每个模型生成可回放日志，`summary.csv` 列出各模型分差及其相对首个模型的按局配对差。新生成的 SL/PPO 检查点会保存随机种子、输入哈希、训练参数和 Git 状态；使用方法见 [模型实验](docs/MODEL_EXPERIMENTS.md)。

### 本机推理接口原型

```bash
python training/serve.py --checkpoint checkpoints/sl_dagger.pt --host 127.0.0.1 --port 8765
```

接口支持创建对局、按座位读取视图、提交人工动作、让模型行动和释放对局。它只用于本机集成验证，不提供鉴权和持久化；请求与响应见 [本机接口](docs/SERVING.md)。

## 文档

- [技术方案设计](docs/DESIGN.md)
- [API 速查](docs/API.md)
- [M1 训练链路与实测](docs/M1_TRAINING.md)
- [M2 PPO 与实测](docs/M2_RL.md)
- [开源项目与许可证调研](docs/RESEARCH.md)

## 合规声明

本项目仅用于技术研究与产品开发。麻将类产品若涉及真实货币、兑换或变相赌博，
可能违反法律法规；**商业落地前必须完成合规评估**，本文档不构成法律意见。
