# 开源麻将引擎 / 项目调研（2026-10）

> 结论先行：**目前没有可直接用于商业产品的「四川血战到底」开源引擎**。
> 建议：规则层自研（本仓库已完成核心），架构与训练范式参考 Suphx / Mortal，
> 性能目标参考 mahjax，开源代码只用于「对照验证规则」与「学习思路」。
>
> ⚠️ 本文不构成法律意见。任何代码进入商业产品前，必须由法务确认许可证。

---

## 1. 候选项目一览

| 项目 | 规则 | 语言/技术 | 许可证 | 可商用性 | 对我们的价值 |
|---|---|---|---|---|---|
| [kangsive/mahjong](https://github.com/kangsive/mahjong) | 四川麻将 + 国标 | Python，GUI，深度学习 AI | **无 LICENSE（版权保留）** | ❌ 不可直接复用 | 参考川麻规则实现、GUI 形态 |
| [Nitasurin/Mortal-Policy](https://github.com/Nitasurin/Mortal-Policy) | 立直麻将（日麻） | Rust + 深度强化学习 | **AGPL-3.0** | ⚠️ 强传染，闭源商用需谨慎 | 训练/推理架构、Suphx 范式落地 |
| [mahjax](https://github.com/nissymori/mahjax) | 立直麻将（日麻） | JAX，向量化 | **Apache-2.0** | ✅ 可商用（但规则不符） | 自博弈吞吐、向量化环境设计 |
| [xiaxin5666/majiang（四川麻将智能模型）](https://gitcode.com/xiaxin5666/majiang) | 四川麻将 | 深度学习 | 未确认 | ❓ 需核实 | 四川麻将 DL 建模参考 |
| [RLCard](https://github.com/datamllab/rlcard) | 斗地主/UNO 等 | Python | MIT | ✅ | 不完整支持川麻；RL 接口设计参考 |
| Suphx（微软） | 日麻 | 深度 RL | 未开源 | — | **方法论**：SL→RL→全局奖励 |

> 说明：`kangsive/mahjong` 的 GitHub API 返回 `"license": null`，即未声明许可证；
> 按默认版权法「保留所有权利」，**不能**拷贝其代码进商业产品。
> `Mortal-Policy` 为 AGPL-3.0：即使只作为网络服务后端，也可能触发源码开放义务。

---

## 2. 关键判断

### 2.1 为什么不能直接拿现成引擎

1. **规则不符**：主流开源实现是日麻（立直）或国标；血战到底的「定缺 / 血战续打 / 一炮多响 / 刮风下雨」差异大。
2. **许可证风险**：真正对口的川麻项目要么无许可证，要么 AGPL。
3. **质量不可控**：零散源码的规则正确性、边界处理（七对/龙七对/抢杠/流局）难以保证。
4. **商业可控性**：规则是产品核心资产，必须自研、可配置、可测试。

### 2.2 可以借鉴什么

| 来源 | 借鉴点 |
|---|---|
| Suphx 论文/博客 | SL 预训练 → RL 自博弈 → 全局奖励预测；麻将状态编码 |
| Mortal-Policy | 策略网络 + 价值网络结构；训练管线与评估方式 |
| mahjax | 向量化/JAX 环境设计；**1M+ steps/s** 的吞吐目标 |
| kangsive/mahjong | 川麻规则清单核对（缺一门、杠开、杠上炮） |

---

## 3. 许可证速查（商用前必读）

| 许可证 | 能否闭源商用 | 注意 |
|---|---|---|
| MIT / Apache-2.0 | ✅ | 保留版权声明 |
| BSD | ✅ | 保留版权声明 |
| LGPL | ⚠️ | 动态链接可闭源，修改库本身需开源 |
| GPL-2.0/3.0 | ⚠️ | 衍生作品需开源 |
| **AGPL-3.0** | ❌（通常） | 网络服务也触发开源义务 |
| 无 LICENSE | ❌ | 默认保留所有权利 |

**建议**

- 自有代码：对外 SDK 用 Apache-2.0；核心引擎可闭源。
- 参考 AGPL 项目：只读思路、不复制代码、不引入依赖。
- 引入任何第三方依赖前，跑一次许可证扫描（如 `pip-licenses` / FOSSA）。

---

## 4. 我们的选择

1. **规则引擎自研**（已完成）：`src/sichuan_mahjong/`，纯 Python、零依赖、可配置。
2. **训练范式**参考 Suphx：规则 AI → SL → RL 自博弈。
3. **性能**：Python 原型先行；自博弈吞吐不足时，参考 mahjax 做向量化，或把向听/规则核心迁 C++/Rust。
4. **不引入**上述任何代码到产品；仅用于规则对照与架构学习。

---

## 5. 参考链接

- kangsive/mahjong: <https://github.com/kangsive/mahjong>
- Nitasurin/Mortal-Policy: <https://github.com/Nitasurin/Mortal-Policy>
- mahjax: <https://github.com/nissymori/mahjax> / <https://pypi.org/project/mahjax/>
- 四川麻将智能模型: <https://gitcode.com/xiaxin5666/majiang>
- RLCard: <https://github.com/datamllab/rlcard>
- Suphx 论文: *Suphx: Mastering Mahjong with Deep Reinforcement Learning* (arXiv:2003.13590)
