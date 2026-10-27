# M3：服务化与上线

> 把训练好的策略接入线上对局服务。当前仓库提供引擎、模型与离线评估；
> 本文件描述服务化阶段的工程方案与验收标准。

## 1. 目标

| 指标 | 目标 |
|---|---|
| 单次决策延迟 | P99 < 50 ms |
| 可用性 | 99.9% |
| 降级 | 模型超时/异常时 0 感知回退规则 AI |
| 公平性 | AI 只能看到 `view(seat)` 允许的信息 |

## 2. 架构

```
Client ──WS/HTTP──> Gateway ──> Room Service（牌局编排）
                                     │
                                     ├─> Rule Engine（唯一真值，本地库）
                                     └─> Policy Service（模型推理）
                                              ├─ 正常：返回 action
                                              └─ 超时/异常：返回 fallback
```

- **Room Service** 负责发牌、合法性校验、结算；AI 只是「另一个玩家」。
- **Policy Service** 无状态：输入 `(features, masks)`，输出 `(action, confidence)`。
- **Fallback**：任何超时/非法动作直接交给 `BaselineAI`，保证牌局不中断。

## 3. 延迟预算

| 环节 | 预算 |
|---|---|
| 网络 | < 10 ms |
| 特征编码（Python） | < 2 ms |
| 模型前向（CPU，batch=1） | < 15 ms |
| 后处理 + 合法性校验 | < 2 ms |
| **P99 合计** | **< 50 ms** |

优化手段（按顺序）：

1. 模型导出 **ONNX Runtime**，CPU 推理可到毫秒级。
2. 同一桌多个 AI 决策合并为 **batch** 推理。
3. 特征编码用 numpy 向量化（当前是纯 Python 列表）。
4. 高频规则（向听）缓存/预计算。

## 4. 协议

请求：

```json
{
  "request_id": "uuid",
  "seat": 2,
  "features": [/* 404 floats */],
  "discard_mask": [/* 27 bool */],
  "call_mask": [/* 4 bool */]
}
```

响应：

```json
{
  "request_id": "uuid",
  "action": {"kind": "discard", "tile": 13},
  "confidence": 0.82,
  "fallback": false,
  "latency_ms": 3.1
}
```

生产建议用 **protobuf** 降低序列化开销。

## 5. 模型管理

| 项 | 方案 |
|---|---|
| 版本 | `policy-v{major}.{minor}`，与规则版本解耦 |
| 发布 | 按房间/用户分桶灰度（1% → 10% → 100%） |
| 回滚 | 保留上一版本，异常时一键切回 |
| 热加载 | 服务启动时加载 + 定时检查新版本 |
| 监控 | 决策分布、非法动作率、fallback 率、延迟分位 |

## 6. 监控与告警

| 指标 | 阈值 |
|---|---|
| `fallback_rate` | > 1% 告警 |
| `p99_latency_ms` | > 80 ms 告警 |
| `illegal_action_rate` | > 0（必须为 0） |
| 模型版本 | 与预期不一致告警 |

## 7. 安全与合规

1. 推理服务只接收编码后的特征，**不接收对手暗牌**。
2. 引擎侧再次校验动作合法性（双保险）。
3. 不记录未脱敏的用户身份信息。
4. 合规评估见 [DESIGN.md](DESIGN.md) §13。

## 8. 验收清单

- [ ] ONNX 推理与 PyTorch 输出一致（误差 < 1e-5）。
- [ ] 单桌 4 AI 并发压测，P99 < 50 ms。
- [ ] 断网/超时注入测试，fallback 生效。
- [ ] 灰度 1% 运行 7 天，无异常。
- [ ] 与规则 AI 的线上胜率/得分符合离线评估预期。
