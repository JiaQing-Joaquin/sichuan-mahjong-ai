# 贡献指南

感谢参与！这个项目是四川麻将「血战到底」的规则引擎与 AI 训练工程。

## 环境

- Python ≥ 3.9（核心引擎与测试只依赖标准库）
- 训练链路需要 `numpy` 与 `torch`：`pip install -e ".[train]"`
- Windows 下可直接双击 `test.bat`，或运行 `python scripts/run_tests.py`

## 开发流程

```bash
git clone <repo>
cd MaJiangAI
python scripts/run_tests.py     # 应全部通过
```

1. 从 `main` 拉分支：`git checkout -b feat/your-topic`
2. 修改代码，**补充或更新测试**
3. 本地跑全量测试
4. 提交并推送，发起 Pull Request

## 代码结构

| 目录 | 说明 |
|---|---|
| `src/sichuan_mahjong/` | 核心库：牌/向听/番型/引擎/编码器/规则 AI |
| `training/` | 采样、SL、DAgger、PPO、评估（需要 torch） |
| `tests/` | 单元测试（441 项核心 + 训练冒烟） |
| `scripts/` | 测试入口、评估、基准 |
| `docs/` | 设计、API、M1/M2/M3、调研 |
| `paper/` | 技术报告与插图 |

## 改动建议

- **规则改动**：必须在 `tests/test_game.py` 或 `tests/test_hand.py` 增加用例；
  规则以引擎为唯一真值，不要让策略层绕过合法性校验。
- **特征/动作空间改动**：同步更新 `encoder.py` 的 `STATE_DIM`、`docs/M1_TRAINING.md`，
  并重新生成数据（旧 checkpoint 不兼容）。
- **提交信息**：建议 `feat:` / `fix:` / `docs:` / `test:` / `refactor:` 前缀。
- **不要提交**：`data/`、`checkpoints/`、`__pycache__`、`.idea/`（已在 `.gitignore`）。

## 许可证

见仓库根目录的 `LICENSE`（如未添加，请先与维护者确认）。
