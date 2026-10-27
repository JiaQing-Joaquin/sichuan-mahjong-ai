"""Generate the figures used in the paper (Chinese labels)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Chinese font ---------------------------------------------------------
for candidate in (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
):
    if os.path.exists(candidate):
        font_manager.fontManager.addfont(candidate)
        break

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DARK = "#1F3864"
MID = "#2E75B6"
LIGHT = "#DEEAF6"
GREY = "#F2F2F2"
TEXT = "#1A1A1A"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=DARK, fs=11, weight="normal", tc=TEXT):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MID, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
    ))


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Fig 1: system architecture
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(12, 100)
    ax.axis("off")

    layers = [
        (78, 16, "服务层", "对局编排 / 房间管理 / 并发推理 / 灰度发布"),
        (58, 16, "决策层", "特征编码器（404 维） → 策略网络（出牌 / 响应 / 价值）"),
        (38, 16, "规则层", "牌与手牌 · 向听 / 胡牌 / 听牌 · 番型计分 · 血战状态机"),
        (18, 12, "数据层", "对局日志 · 牌谱 · 训练数据 · 离线评估"),
    ]
    for y, h, title, detail in layers:
        box(ax, 8, y, 84, h, "", fc="white", ec=DARK)
        ax.text(11, y + h - 3.2, title, ha="left", va="center",
                fontsize=13, weight="bold", color=DARK)
        ax.text(11, y + h / 2 - 2.2, detail, ha="left", va="center",
                fontsize=10.5, color=TEXT)

    for y in (78, 58, 38):
        arrow(ax, 50, y, 50, y - 4)

    ax.text(50, 97, "川麻血战到底 AI 系统架构", ha="center", va="center",
            fontsize=15, weight="bold", color=DARK)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
# Fig 2: network
# ---------------------------------------------------------------------------
def fig_model():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(14, 100)
    ax.axis("off")

    box(ax, 2, 40, 17, 20, "输入特征\n404 维", fc=GREY, fs=11)
    box(ax, 24, 42, 16, 16, "Linear\n404 → 512\n+ ReLU", fc=LIGHT, fs=10)
    box(ax, 44, 42, 16, 16, "Linear\n512 → 512\n+ ReLU", fc=LIGHT, fs=10)
    box(ax, 64, 42, 16, 16, "Linear\n512 → 256\n+ ReLU", fc=LIGHT, fs=10)

    arrow(ax, 19, 50, 24, 50)
    arrow(ax, 40, 50, 44, 50)
    arrow(ax, 60, 50, 64, 50)

    heads = [
        (84, 74, "出牌头\n27 维", "mask 27"),
        (84, 47, "响应头\n4 维", "mask 4"),
        (84, 20, "价值头\n1 维", "—"),
    ]
    for hx, hy, label, mask in heads:
        box(ax, hx, hy, 13, 14, label, fc="#FFF2CC", fs=10)
        ax.text(hx + 6.5, hy - 3.5, mask, ha="center", va="center",
                fontsize=9, color="#7F6000")

    # arrows fan out from the trunk output to the three heads
    for _, hy, _, _ in heads:
        arrow(ax, 80, 50, 84, hy + 7)

    ax.text(50, 94, "策略网络结构（PolicyValueNet）", ha="center", va="center",
            fontsize=15, weight="bold", color=DARK)
    save(fig, "fig2_model.png")


# ---------------------------------------------------------------------------
# Fig 3: training pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(32, 96)
    ax.axis("off")

    labels = [
        "基线自对弈\n5000 局",
        "监督学习\n12 epoch",
        "DAgger\n600 局",
        "再训练\n合并数据",
        "PPO 自博弈\n480 局 + 对手池",
        "离线评估\n300 局",
    ]
    data = [
        "304,219 条样本",
        "一致率 0.779",
        "20,885 条专家标注",
        "一致率 0.774",
        "51,816 条 transition",
        "对基线 -0.69/座",
    ]
    width, gap = 14.0, 2.4
    xs = [2.0 + i * (width + gap) for i in range(len(labels))]
    ys, h = 52, 18
    for x, label in zip(xs, labels):
        box(ax, x, ys, width, h, label, fc=LIGHT, fs=10)

    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + width, ys + h / 2, xs[i + 1], ys + h / 2)

    for x, label in zip(xs, data):
        ax.text(x + width / 2, 38, label, ha="center", va="center",
                fontsize=9, color="#404040")

    ax.text(50, 90, "训练流水线", ha="center", va="center",
            fontsize=15, weight="bold", color=DARK)
    save(fig, "fig3_pipeline.png")


# ---------------------------------------------------------------------------
# Fig 4: learning curves
# ---------------------------------------------------------------------------
def fig_curves():
    epochs = list(range(1, 13))
    sl = [0.680, 0.727, 0.745, 0.757, 0.762, 0.766, 0.772, 0.774, 0.776, 0.773, 0.779, 0.779]
    dagger = [0.689, 0.729, 0.745, 0.758, 0.759, 0.763, 0.768, 0.772, 0.774, 0.780, 0.779, 0.779]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(epochs, sl, marker="o", linewidth=2, color=MID, label="纯行为克隆（5000 局）")
    ax.plot(epochs, dagger, marker="s", linewidth=2, color="#C00000", label="DAgger 聚合（+600 局）")
    ax.set_xlabel("训练轮次（epoch）", fontsize=11)
    ax.set_ylabel("验证集出牌一致率", fontsize=11)
    ax.set_xticks(epochs)
    ax.set_ylim(0.66, 0.80)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.set_title("验证集出牌一致率随训练轮次的变化", fontsize=13, color=DARK, pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig4_curves.png")


# ---------------------------------------------------------------------------
# Fig 5: results
# ---------------------------------------------------------------------------
def fig_results():
    labels = ["SL\n(BC)", "SL + PPO", "SL + DAgger", "DAgger + PPO"]
    values = [-1.13, -0.93, -0.69, -1.04]
    colors = ["#8EA9DB", "#8EA9DB", "#2E75B6", "#C00000"]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0, color="#404040", linewidth=1)
    ax.set_ylabel("对基线的每座得分差", fontsize=11)
    ax.set_ylim(-1.35, 0.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.06,
                f"{value:+.2f}", ha="center", va="top", fontsize=10, color=TEXT)
    ax.set_title("不同模型的强度对比（越低越好，0 表示与规则 AI 持平）",
                 fontsize=12, color=DARK, pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig5_results.png")


if __name__ == "__main__":
    fig_architecture()
    fig_model()
    fig_pipeline()
    fig_curves()
    fig_results()
