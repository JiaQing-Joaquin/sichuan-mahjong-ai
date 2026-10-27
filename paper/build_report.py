"""Rebuild the PDF report from the checked-in implementation and experiment notes.

The diagrams are vector drawings. Run with the bundled Codex Python runtime:
    python paper/build_report.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle, Flowable,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "阶段性技术总结.pdf"
FONT = "ReportCN"
BOLD = "ReportCNBold"
MATH = "ReportMath"
pdfmetrics.registerFont(TTFont(FONT, r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont(BOLD, r"C:\Windows\Fonts\simhei.ttf"))
pdfmetrics.registerFont(TTFont(MATH, r"C:\Windows\Fonts\cambria.ttc", subfontIndex=0))
# Diagram grammar follows the supplied multi-panel paper figure: pastel
# modules, dashed sub-system frames, black signal paths, and orange annotations.
NAVY = colors.HexColor("#202832")
BLUE = colors.HexColor("#b6deec")
TEAL = colors.HexColor("#6edac4")
OCHRE = colors.HexColor("#d5e8ad")
PALE = colors.HexColor("#d8e9f5")
MINT = colors.HexColor("#cff2e8")
AMBER = colors.HexColor("#f8e8cc")
PEACH = colors.HexColor("#f8e3d7")
ORANGE = colors.HexColor("#bf5724")
FOREST = colors.HexColor("#164e43")
GREY = colors.HexColor("#58636d")
RULE = colors.HexColor("#cbd2d8")
ARROW = colors.HexColor("#161e1f")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CNTitle", fontName=BOLD, fontSize=22, leading=31,
                          textColor=NAVY, alignment=TA_CENTER, spaceAfter=9))
styles.add(ParagraphStyle(name="CNSub", fontName=FONT, fontSize=10.5, leading=18,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name="CNH1", fontName=BOLD, fontSize=15, leading=23,
                          textColor=NAVY, spaceBefore=18, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle(name="CNH2", fontName=BOLD, fontSize=12.5, leading=20,
                          textColor=NAVY, spaceBefore=14, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle(name="CNBody", fontName=FONT, fontSize=10.5, leading=19,
                          textColor=NAVY, alignment=TA_JUSTIFY, wordWrap="CJK",
                          firstLineIndent=21,
                          spaceAfter=7))
styles.add(ParagraphStyle(name="CNNote", fontName=FONT, fontSize=9, leading=16,
                          wordWrap="CJK",
                          textColor=GREY, spaceAfter=6))
styles.add(ParagraphStyle(name="CNCap", fontName=FONT, fontSize=10.5, leading=17,
                          textColor=NAVY, alignment=TA_CENTER,
                          spaceBefore=6, spaceAfter=12))
styles.add(ParagraphStyle(name="CNTableCap", fontName=FONT, fontSize=10.5,
                          leading=17, alignment=TA_CENTER, textColor=NAVY))
styles.add(ParagraphStyle(name="CNMath", fontName=MATH, fontSize=11.5,
                          leading=19, alignment=TA_CENTER, textColor=NAVY))
styles.add(ParagraphStyle(name="CNCell", fontName=FONT, fontSize=9.2, leading=15,
                          wordWrap="CJK",
                          textColor=NAVY))
styles.add(ParagraphStyle(name="CNHeadCell", fontName=BOLD, fontSize=9.2, leading=15,
                          textColor=NAVY))

story = []


def p(text: str, style: str = "CNBody") -> None:
    story.append(Paragraph(text, styles[style]))


class Equation(Flowable):
    """Centered display equation with an independently aligned chapter number."""
    def __init__(self, markup: str, number: str):
        super().__init__()
        self.markup, self.number = markup, number
        self.width, self.height = 155 * mm, 34
        self.keepWithNext = 1

    def draw(self):
        para = Paragraph(self.markup, styles["CNMath"])
        w, h = para.wrap(self.width - 78, self.height)
        para.drawOn(self.canv, (self.width - w) / 2, (self.height - h) / 2)
        self.canv.setFont(MATH, 10.5)
        self.canv.drawRightString(self.width, (self.height - 11) / 2, f"({self.number})")


def eq(markup: str, number: str) -> None:
    story.append(Equation(markup, number))


class ThesisTable(Table):
    """Repeat caption and headers; mark the caption 续表 on later fragments."""
    def __init__(self, data, caption: str = "", **kwargs):
        kwargs.setdefault("repeatRows", 2)
        super().__init__(data, **kwargs)
        self.caption_text = caption or getattr(data[0][0], "text", "")

    def split(self, availWidth, availHeight):
        parts = super().split(availWidth, availHeight)
        for continuation in parts[1:]:
            continuation._cellvalues[0][0] = Paragraph(
                self.caption_text.replace("表 ", "续表 ", 1), styles["CNTableCap"]
            )
        return parts


def table(caption: str, headers: list[str], rows: list[list[str]],
          widths: list[int], keep: bool = False) -> None:
    span_width = 155 * mm
    scale = span_width / sum(widths)
    data = [[Paragraph(caption, styles["CNTableCap"])] + [""] * (len(headers)-1)]
    data.append([Paragraph(v, styles["CNHeadCell"]) for v in headers])
    data.extend([[Paragraph(str(v), styles["CNCell"]) for v in row] for row in rows])
    t = ThesisTable(data, caption, colWidths=[v * scale for v in widths], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("LINEABOVE", (0, 1), (-1, 1), 0.9, NAVY),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 11))


def node(d: Drawing, x: float, y: float, w: float, h: float,
         title: str, detail: str = "", fill=PALE) -> None:
    d.add(Rect(x, y, w, h, rx=7, ry=7, fillColor=fill,
               strokeColor=FOREST, strokeWidth=1.0))
    d.add(String(x + 8, y + h - 17, title, fontName=BOLD, fontSize=8.4, fillColor=NAVY))
    for i, line in enumerate(detail.split("\n")):
        d.add(String(x + 8, y + h - 30 - i * 12, line,
                     fontName=FONT, fontSize=7.2, fillColor=NAVY))


def panel(d: Drawing, x: float, y: float, w: float, h: float,
          title: str, fill) -> None:
    d.add(Rect(x, y, w, h, rx=12, ry=12, fillColor=fill,
               strokeColor=FOREST, strokeWidth=1.1, strokeDashArray=[5, 4]))
    d.add(String(x + 10, y + h - 17, title, fontName="Times-BoldItalic",
                 fontSize=11, fillColor=ORANGE))


def tensor(d: Drawing, x: float, y: float, w: float, h: float,
           label: str, color=BLUE) -> None:
    for shift in (9, 5, 1):
        d.add(Polygon([x + shift, y + shift, x + w + shift, y + shift,
                       x + w + shift + 8, y + h + shift,
                       x + shift + 8, y + h + shift],
                      fillColor=color, strokeColor=FOREST, strokeWidth=0.7))
    d.add(String(x + 8, y - 12, label, fontName=FONT, fontSize=7.2, fillColor=ORANGE))


def operator(d: Drawing, x: float, y: float, symbol: str, color=FOREST) -> None:
    d.add(Circle(x, y, 10, fillColor=colors.white,
                 strokeColor=color, strokeWidth=1.1))
    if symbol == "×":
        d.add(Line(x - 5, y - 5, x + 5, y + 5,
                   strokeColor=color, strokeWidth=1.3))
        d.add(Line(x - 5, y + 5, x + 5, y - 5,
                   strokeColor=color, strokeWidth=1.3))
    elif symbol == "−":
        d.add(Line(x - 5, y, x + 5, y,
                   strokeColor=color, strokeWidth=1.3))
    else:
        d.add(String(x - 4, y - 4, symbol, fontName="Helvetica-Bold",
                     fontSize=10, fillColor=color))


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float,
          label: str = "") -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=ARROW, strokeWidth=0.9))
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        pts = [x2, y2, x2 - sign * 5, y2 + 2.5, x2 - sign * 5, y2 - 2.5]
        lx, ly = (x1 + x2) / 2, y1 + 5
    else:
        sign = 1 if y2 > y1 else -1
        pts = [x2, y2, x2 - 2.5, y2 - sign * 5, x2 + 2.5, y2 - sign * 5]
        lx, ly = x1 + 4, (y1 + y2) / 2
    d.add(Polygon(pts, fillColor=ARROW, strokeColor=ARROW))
    if label:
        d.add(String(lx, ly, label, fontName=FONT, fontSize=6.6, fillColor=GREY))


def fig_architecture() -> Drawing:
    d = Drawing(490, 285)
    panel(d, 4, 74, 482, 209, "Game-Policy Loop", MINT)
    panel(d, 4, 1, 482, 66, "", AMBER)
    node(d, 8, 190, 125, 67, "入口与评测", "scripts/* / training/evaluate.py\n对局循环、2v2 轮换")
    node(d, 177, 190, 135, 67, "决策器", "baseline.py / policy_ai.py\n非学习动作回退规则 AI", MINT)
    node(d, 354, 190, 128, 67, "策略网络", "model.py: 404→512→512→256\n27/4/1 三个输出头", AMBER)
    node(d, 8, 83, 125, 79, "对局状态机 game.py", "legal_actions() / step()\n换牌、定缺、行动、响应\n结算与血战续打")
    node(d, 177, 83, 135, 79, "状态编码 encoder.py", "只读本人手牌与公开信息\n404 维 + 27/4 合法动作掩码", MINT)
    node(d, 354, 83, 128, 79, "规则计算", "tiles.py / hand.py\nscoring.py", AMBER)
    node(d, 8, 5, 226, 49, "离线数据与训练", "collect / DAgger / SL / PPO\ndata/*.npz → checkpoints/*.pt")
    node(d, 256, 5, 226, 49, "测试与文档", "tests/* / docs/*\n暂无服务端或房间管理实现")
    arrow(d, 133, 224, 177, 224, "选择动作")
    arrow(d, 312, 224, 354, 224, "推理")
    arrow(d, 70, 190, 70, 162, "推进")
    arrow(d, 133, 121, 177, 121, "可见状态")
    arrow(d, 245, 190, 245, 162, "编码")
    arrow(d, 70, 83, 70, 54, "采样")
    d.add(Line(70, 83, 70, 73, strokeColor=ARROW, strokeWidth=0.9))
    d.add(Line(70, 73, 419, 73, strokeColor=ARROW, strokeWidth=0.9))
    arrow(d, 419, 73, 419, 83, "牌型/计分")
    return d


def fig_training() -> Drawing:
    d = Drawing(490, 248)
    panel(d, 4, 149, 482, 96, "", PALE)
    panel(d, 4, 57, 482, 87, "", AMBER)
    d.add(String(334, 234, "Behavior Cloning", fontName="Times-BoldItalic",
                 fontSize=10, fillColor=ORANGE))
    node(d, 8, 165, 128, 64, "规则 AI 自对弈", "collect.py → sl.npz\n5000 局 / 304219 样本")
    node(d, 181, 165, 128, 64, "行为克隆 BC", "train_sl.py\n按局划分训练/验证", MINT)
    node(d, 354, 165, 128, 64, "SL checkpoint", "sl_policy.pt\n出牌/响应/价值损失", AMBER)
    node(d, 8, 70, 128, 64, "SL 策略访问状态", "dagger.py；规则专家标注\n600 局 / 20885 样本")
    node(d, 181, 70, 128, 64, "合并再训练", "sl.npz + dagger.npz\n得到 sl_dagger.pt", MINT)
    node(d, 354, 70, 128, 64, "PPO 分支", "ppo.py；终局分/50 裁剪\n可选规则 AI + 历史模型", AMBER)
    arrow(d, 136, 197, 181, 197)
    arrow(d, 309, 197, 354, 197)
    arrow(d, 418, 165, 418, 134, "可作初值")
    d.add(Line(386, 165, 386, 149, strokeColor=ARROW, strokeWidth=0.9))
    d.add(Line(386, 149, 72, 149, strokeColor=ARROW, strokeWidth=0.9))
    arrow(d, 72, 149, 72, 134, "SL 初值")
    arrow(d, 136, 102, 181, 102, "专家重标")
    arrow(d, 309, 102, 354, 102, "可作初值")
    d.add(String(8, 38, "评测：evaluate.py；每局 2 模型座 + 2 规则座，逐局轮换座位。", fontName=FONT,
                 fontSize=8, fillColor=GREY))
    d.add(String(8, 22, "PPO 使用终局回报 - 当前代码无 GAE、KL 早停和在线服务。", fontName=FONT,
                 fontSize=8, fillColor=GREY))
    return d


def fig_encoder_detail() -> Drawing:
    """Exact half-open offsets from encode_state, plus separate action masks."""
    d = Drawing(490, 345)
    d.add(String(10, 327, "Game 对当前座位可见的状态", fontName=BOLD, fontSize=9, fillColor=NAVY))
    d.add(String(363, 327, "encode_state()", fontName=BOLD, fontSize=9, fillColor=NAVY))
    d.add(Rect(4, 219, 353, 102, rx=8, ry=8, fillColor=AMBER,
               strokeColor=FOREST, strokeWidth=0.9, strokeDashArray=[4, 3]))
    d.add(Rect(4, 66, 353, 151, rx=8, ry=8, fillColor=MINT,
               strokeColor=FOREST, strokeWidth=0.9, strokeDashArray=[4, 3]))
    d.add(Rect(4, 46, 353, 19, rx=7, ry=7, fillColor=PALE,
               strokeColor=FOREST, strokeWidth=0.9, strokeDashArray=[4, 3]))
    bands = [
        ("[0:27)", "本人手牌计数 / 4", "27"),
        ("[27:54)", "本人碰牌掩码", "27"),
        ("[54:81)", "本人杠牌掩码", "27"),
        ("[81:189)", "四家弃牌计数 / 4", "108"),
        ("[189:297)", "四家副露牌张数 / 4", "108"),
        ("[297:309)", "四家定缺 one-hot", "12"),
        ("[309:310)", "牌墙剩余 / 55", "1"),
        ("[310:314)", "四家分数 / 50", "4"),
        ("[314:318)", "四家已胡标记", "4"),
        ("[318:345)", "本人最后摸牌 one-hot", "27"),
        ("[345:346)", "是否处于响应阶段", "1"),
        ("[346:373)", "待响应弃牌 one-hot", "27"),
        ("[373:377)", "弃牌者相对座位 one-hot", "4"),
        ("[377:404)", "牌种剩余可见外张数 / 4", "27"),
    ]
    for i, (offset, label, dim) in enumerate(bands):
        y = 299 - i * 19
        d.add(Rect(8, y, 345, 18, fillColor=colors.white,
                   strokeColor=RULE, strokeWidth=0.35))
        d.add(Rect(8, y, 2.5, 18,
                   fillColor=BLUE if i < 5 else TEAL if i < 10 else OCHRE,
                   strokeColor=None))
        d.add(String(15, y + 5, offset, fontName=FONT, fontSize=7.4, fillColor=ORANGE))
        d.add(String(93, y + 5, label, fontName=FONT, fontSize=7.4, fillColor=NAVY))
        d.add(String(323, y + 5, dim, fontName=FONT, fontSize=7.4, fillColor=GREY))
    node(d, 367, 222, 115, 50, "拼接向量 X", "float / 404 维", MINT)
    node(d, 367, 144, 115, 50, "出牌掩码 D", "legal_actions → 27 位", AMBER)
    node(d, 367, 66, 115, 50, "响应掩码 C", "胡/碰/明杠/过 → 4 位", AMBER)
    arrow(d, 353, 247, 367, 247)
    d.add(String(8, 8, "编码不读取对手暗牌；D 和 C 由 Game.legal_actions() 另行生成，不占 404 维。",
                 fontName=FONT, fontSize=7.4, fillColor=GREY))
    return d


def fig_network_detail() -> Drawing:
    """Exact Linear/ReLU sequence and masked heads in PolicyValueNet.forward."""
    d = Drawing(490, 265)
    panel(d, 4, 104, 360, 149, "Shared MLP", MINT)
    panel(d, 369, 17, 116, 240, "Policy / Value", AMBER)
    for x in (20, 110, 202, 294):
        tensor(d, x, 203, 24, 12, "", BLUE)
    node(d, 7, 133, 69, 58, "输入 X", "B × 404")
    node(d, 92, 133, 76, 58, "Linear 1", "404 → 512\nReLU", MINT)
    node(d, 184, 133, 76, 58, "Linear 2", "512 → 512\nReLU", MINT)
    node(d, 276, 133, 76, 58, "Linear 3", "512 → 256\nReLU", MINT)
    for x1, x2 in ((76, 92), (168, 184), (260, 276)):
        arrow(d, x1, 162, x2, 162)
    node(d, 376, 181, 107, 52, "出牌 Linear", "256 → 27 logits\nD mask → πd", AMBER)
    node(d, 376, 105, 107, 52, "响应 Linear", "256 → 4 logits\nC mask → πc", AMBER)
    node(d, 376, 29, 107, 52, "价值 Linear", "256 → 1，squeeze\nV(s)", AMBER)
    for y in (207, 131, 55):
        arrow(d, 352, 162, 376, y)
    operator(d, 365, 207, "M")
    operator(d, 365, 131, "M")
    d.add(String(10, 85, "训练：T=0 用 CE(πd,Y)，T=1 用 CE(πc,Y)，全部样本用 MSE(V,R)。",
                 fontName=FONT, fontSize=7.8, fillColor=NAVY))
    d.add(String(10, 65, "推理：选当前阶段的头，非法 logit 置 dtype 最小值；ModelAI 使用 argmax。",
                 fontName=FONT, fontSize=7.8, fillColor=NAVY))
    d.add(String(10, 45, "PPO rollout：用同一掩码头的 Categorical 分布采样，并记录 old logp 与 V。",
                 fontName=FONT, fontSize=7.8, fillColor=NAVY))
    d.add(String(10, 20, "M 表示 masked_fill 非法 logits；hidden=512，总参数 609568。",
                 fontName=FONT, fontSize=7.8, fillColor=GREY))
    return d


def fig_ppo_detail() -> Drawing:
    """Actual rollout record and clipped PPO update, without invented GAE."""
    d = Drawing(490, 347)
    panel(d, 4, 139, 482, 204, "", MINT)
    panel(d, 4, 38, 482, 97, "", PEACH)
    d.add(String(377, 332, "Self-Play Rollout", fontName="Times-BoldItalic",
                 fontSize=10, fillColor=ORANGE))
    d.add(String(363, 124, "Clipped PPO Objective", fontName="Times-BoldItalic",
                 fontSize=10, fillColor=ORANGE))
    node(d, 8, 257, 140, 66, "初始化策略", "读取 SL/DAgger checkpoint\nAdam；默认 lr=3e-5", MINT)
    node(d, 174, 257, 140, 66, "Game rollout", "每局 2 策略座 + 2 对手座\n座位逐局轮换")
    node(d, 340, 257, 140, 66, "对手池", "规则 AI；--league 可加\n历史 ModelAI", AMBER)
    arrow(d, 148, 290, 174, 290)
    arrow(d, 340, 290, 314, 290)
    node(d, 8, 152, 140, 70, "编码与掩码", "encode() → X, D, C\n非策略动作回退 BaselineAI", MINT)
    node(d, 174, 152, 140, 70, "按阶段采样", "πd/πc → Categorical\n记录 a, old logp, V")
    node(d, 340, 152, 140, 70, "transition 缓冲", "feats, dm, cm, t, a\nold logp, value, seat", AMBER)
    arrow(d, 244, 257, 78, 222)
    arrow(d, 148, 187, 174, 187)
    arrow(d, 314, 187, 340, 187)
    node(d, 8, 48, 140, 70, "终局回报 R", "每条记录按所属座位赋值\nclip(score/50, -5, 5)", MINT)
    node(d, 174, 48, 140, 70, "优势 A", "A = R - old V\n全批标准化")
    node(d, 340, 48, 140, 70, "PPO 更新", "4 epochs；minibatch 512\nclip 比率 + MSE - 熵", AMBER)
    d.add(Line(244, 323, 244, 333, strokeColor=ARROW, strokeWidth=0.9))
    d.add(Line(244, 333, 2, 333, strokeColor=ARROW, strokeWidth=0.9))
    d.add(Line(2, 333, 2, 83, strokeColor=ARROW, strokeWidth=0.9))
    arrow(d, 2, 83, 8, 83)
    arrow(d, 148, 83, 174, 83)
    operator(d, 161, 83, "−")
    arrow(d, 314, 83, 340, 83)
    arrow(d, 410, 152, 244, 118)
    arrow(d, 410, 152, 410, 118)
    d.add(String(8, 25, "ratio = exp(new logp - old logp)；ε=0.2，value_coef=0.5，entropy_coef=0.01，梯度裁剪 0.5。",
                 fontName=FONT, fontSize=7.2, fillColor=NAVY))
    d.add(String(8, 8, "无轨迹折扣、GAE、KL 早停；更新后进入下一轮 rollout，最终保存 ppo_policy.pt。",
                 fontName=FONT, fontSize=7.2, fillColor=GREY))
    return d


def add_figure(drawing: Drawing, caption: str) -> None:
    label, body = caption.split("  ", 1)
    factor = min(1.0, 155 * mm / drawing.width)
    drawing.scale(factor, factor)
    drawing.width *= factor
    drawing.height *= factor
    drawing.hAlign = "CENTER"
    story.append(KeepTogether([
        drawing,
        Paragraph(f"<b>{label}</b> {body}", styles["CNCap"]),
    ]))


p("阶段性技术总结", "CNTitle")
p("四川麻将血战到底 AI｜规则引擎 · 状态编码 · MLP · DAgger · PPO｜2026 年 10 月", "CNSub")
p("摘要", "CNH1")
p("本报告梳理仓库当前可运行的规则引擎、训练脚本、数据文件、模型检查点与测试，说明模块调用关系和训练/评测口径。规则引擎支持四人 108 张牌、定缺、可选换三张、碰杠、抢杠胡、一炮多响和血战续打；流局赔偿仍是简化实现。模型只负责出牌和响应两类决策，其余动作由规则策略处理。仓库附带 5000 局规则 AI 采样数据和 600 局 DAgger 数据。已有实验记录中，DAgger 模型相对规则基线的每座得分差为 -0.69，优于纯行为克隆的 -1.13；小规模 PPO 未带来进一步提升。")
p("1  项目文件与实现边界", "CNH1")
p("图 1 将实际代码节点展开到文件级。Game 是规则真值；决策器读取合法动作并调用编码/网络，最终由 Game.step() 推进状态。图中未把规划中的服务层画成已交付系统。")
add_figure(fig_architecture(), "图 1-1  现有实现的模块与调用关系（箭头表示调用或数据传递）")
table("表 1-1 项目主要文件及职责", ["目录 / 文件", "职责与产物"], [
    ["src/sichuan_mahjong/", "tiles.py 牌编码；hand.py 向听/胡牌；scoring.py 计分；game.py 状态机；baseline.py 规则策略；encoder.py 特征和合法动作掩码。"],
    ["training/", "collect.py 监督样本；collect_batch.py 分片合并；dagger.py 重标；train_sl.py 行为克隆；ppo.py 强化学习；model.py 网络；policy_ai.py 推理；evaluate.py 对基线评测。"],
    ["scripts/ 与 tests/", "模拟、基准、评测和测试入口；测试覆盖规则、编码、训练与 PPO 冒烟。"],
    ["data/ 与 checkpoints/", "sl.npz、dagger.npz；sl_policy.pt、sl_dagger.pt、ppo_policy.pt。二进制产物不等同于训练日志。"],
    ["docs/ 与 paper/", "设计、API、实验记录、服务规划；本报告及构建脚本。M3_SERVING.md 是方案，尚无服务端实现。"],
], [155, 325])
p("2  规则引擎与信息边界", "CNH1")
p("Game 的核心接口为 legal_actions()、step(action)、scores() 和 view(seat)。流程包含换三张、定缺、回合行动、弃牌响应与终局；合法动作由引擎生成，计分由 scoring.py 处理。玩家胡牌后退出，其他玩家继续；同一弃牌可被多家胡。流局查花猪/查大叫为可配置的简化结算，连庄与多局积分未实现。")
p("编码器只读取本人的暗牌，以及四家牌河、副露、定缺、分数、胡牌标记等公开状态；不会读取对手暗牌。它输出 404 维浮点特征，出牌掩码 27 位，响应掩码 4 位。响应动作次序固定为胡、碰、明杠、过。")
add_figure(fig_encoder_detail(), "图 2-1  状态编码的偏移、归一化和独立合法动作掩码")
p("状态编码器采用显式特征拼接。设 i 为牌种索引，s 为座位，h、q、m 分别为本人手牌计数、四家弃牌计数与明副露计数。输出由 14 个区段按固定顺序拼接，总维数为 404；合法动作掩码 D、C 不参与拼接。")
eq("<i>x</i> = [<i>h</i>/4, <i>p</i>, <i>g</i>, <i>q</i>/4, <i>m</i>/4, <i>u</i>, <i>w</i>/55, <i>f</i>/50, <i>z</i>, <i>d</i>, <i>r</i>, <i>t</i>, <i>v</i>, <i>n</i>] ∈ R<super>404</super>", "2-1")
p("式 (2-1) 中 p、g 为本人碰/杠掩码，u 为四座定缺独热编码，w 为牌墙剩余张数，f 为四座分数，z 为已胡标记，d 为刚摸牌独热向量，r 为响应阶段标记，t 为待响应牌独热向量，v 为弃牌者相对座位编码，n 为每种牌的剩余张数。")
eq("<i>n</i><sub>i</sub> = max{0, 4 - <i>h</i><sub>i</sub> - ∑<sub>s=0</sub><super>3</super> (<i>q</i><sub>s,i</sub> + <i>m</i><sub>s,i</sub>)}/4", "2-2")
p("式 (2-2) 的剩余张数只扣除本人手牌和全部可见弃牌、副露，不涉及对手暗牌。D 和 C 则由规则引擎针对当前局面独立生成：")
eq("<i>D</i><sub>i</sub> = 1[(DISCARD, i) ∈ A<sub>legal</sub>]", "2-3")
eq("<i>C</i><sub>k</sub> = 1[K<sub>k</sub> ∈ kind(A<sub>legal</sub>)]", "2-4")
p("式 (2-3) 的二元组表示出牌动作及牌种。kind(A<sub>legal</sub>) 表示合法动作的类型集合，K 按胡、碰、明杠、过排序。指示函数 1[·] 在条件成立时取 1，否则取 0。特征段的精确偏移见图 2-1，归一化汇总见表 2-1。")
table("表 2-1 状态编码特征及归一化", ["404 维特征段", "维度", "代码中的值"], [
    ["本人手牌 / 碰 / 杠", "27 + 27 + 27", "张数 / 4；两个副露布尔掩码"],
    ["四家弃牌 / 副露", "108 + 108", "按座位顺序，每家 27 维，张数 / 4"],
    ["定缺 / 牌墙 / 分数 / 已胡", "12 + 1 + 4 + 4", "one-hot；剩余 / 55；分数 / 50；布尔标记"],
    ["刚摸牌 / 响应标记 / 待响应牌 / 相对座位", "27 + 1 + 27 + 4", "one-hot 与阶段标记"],
    ["可见牌以外的剩余张数", "27", "max(0, 4 - 可见数) / 4"],
], [204, 94, 182])
p("3  网络与决策路径", "CNH1")
add_figure(fig_network_detail(), "图 3-1  共享 MLP、掩码动作头与价值头")
p("PolicyValueNet 使用 404→512→512→256 的三层 ReLU 主干，输出 27 维出牌 logits、4 维响应 logits 和 1 维价值。非法动作由掩码屏蔽。ModelAI 只在出牌或响应阶段调用动作头；定缺、换牌、自摸、暗杠和补杠交给 BaselineAI。")
p("现有网络没有独立的序列解码器。共享表示通过出牌头和响应头转为动作分数，经过合法动作掩码后完成选择。其逐层映射为：")
eq("<i>h</i><super>(0)</super> = <i>x</i>, &nbsp; <i>h</i><super>(l)</super> = ReLU(<i>W</i><super>(l)</super><i>h</i><super>(l-1)</super> + <i>b</i><super>(l)</super>), &nbsp; l = 1, 2, 3", "3-1")
eq("<i>z</i><sub>D</sub> = <i>W</i><sub>D</sub><i>h</i><super>(3)</super> + <i>b</i><sub>D</sub>, &nbsp; <i>z</i><sub>C</sub> = <i>W</i><sub>C</sub><i>h</i><super>(3)</super> + <i>b</i><sub>C</sub>, &nbsp; <i>V</i> = <i>W</i><sub>V</sub><i>h</i><super>(3)</super> + <i>b</i><sub>V</sub>", "3-2")
eq("<i>z</i><super>mask</super><sub>j</sub> = <i>M</i><sub>j</sub><i>z</i><sub>j</sub> + (1 - <i>M</i><sub>j</sub>) min(dtype), &nbsp; <i>a</i>* = argmax<sub>j</sub> <i>z</i><super>mask</super><sub>j</sub>", "3-3")
p("式 (3-3) 中 M 在出牌阶段取 D，在响应阶段取 C；min(dtype) 是浮点类型的最小有限值。推理使用 argmax，PPO 采样则从掩码后的 logits 构造 Categorical 分布。价值头不参与推理动作选择。")
table("表 3-1 网络层形状与参数量", ["层 / 输出", "形状", "参数量", "参与的决策或损失"], [
    ["共享层 1", "404 → 512 + ReLU", "207360", "全部样本"],
    ["共享层 2", "512 → 512 + ReLU", "262656", "全部样本"],
    ["共享层 3", "512 → 256 + ReLU", "131328", "全部样本"],
    ["出牌头", "256 → 27", "6939", "T=0 的合法出牌与交叉熵"],
    ["响应头", "256 → 4", "1028", "T=1 的胡/碰/明杠/过与交叉熵"],
    ["价值头", "256 → 1", "257", "终局分/50 的回归；PPO 的基线"],
], [115, 110, 70, 185])
p("3.1  为什么采用共享 MLP", "CNH2")
p("当前输入是人工定义的 404 维定长向量，已把牌种计数、公开动作、阶段和相对座位编码成数值，不包含完整出牌事件序列。三层 MLP 可以直接消费这种异质特征；共享主干让出牌、响应和价值预测复用同一局面表示。检查点中的 hidden=512，对应 609568 个参数。27/4 位掩码由规则引擎生成，模型只需在合法候选中排序；这也让监督学习和 PPO 共用同一个网络与动作接口。", "CNBody")
table("表 3-2 候选模型与当前选型依据", ["候选方法", "可能的优势", "本项目当前的取舍"], [
    ["共享 MLP（三头）", "适配定长特征；参数量较小；训练/推理接口简单。", "作为可复现基线，容易接入 BC、DAgger 和 PPO；不能自动学习长事件序列。"],
    ["CNN / 牌面卷积", "可利用同花色相邻牌的局部结构。", "现有 404 维含分数、座位和阶段等异质块；直接卷积缺少统一的局部邻接关系，需另设计输入。"],
    ["Transformer / 序列模型", "可建模弃牌和行动的先后顺序及长程依赖。", "当前编码只保留按牌种聚合的牌河计数；训练文件没有事件序列，改用序列模型需重做数据采集。"],
    ["搜索 / 蒙特卡洛规划", "可显式评估多步行动后果。", "四人不完全信息需为未知手牌建模并多次模拟；现有工程尚无信念状态或批量搜索框架。"],
], [116, 180, 184])
p("选型因此是当前数据形态、训练规模和工程复杂度下的基线选择，而不是已验证的最优架构。仓库没有 MLP 与 CNN、Transformer 或搜索模型在相同数据和种子下的对照实验，不能声称 MLP 的实战强度或延迟已优于这些方法。后续若保存逐步行动历史，可以再测试序列模型；若需更强规划能力，应先建立未知手牌采样和多模型对照评测。", "CNNote")
p("4  数据和训练路径", "CNH1")
add_figure(fig_training(), "图 4-1  行为克隆、DAgger 与 PPO 的数据依赖和检查点")
p("sl.npz 实测为 304219 条、X 为 (304219,404)；dagger.npz 为 20885 条、X 为 (20885,404)。两者均包含 D/C 合法动作掩码、Y 动作标签、T 动作类型、R 终局分数/50、G 局号和 S 座位。监督训练按局号分割训练/验证，避免同一局样本跨集合。只保留至少有两个合法响应的响应点。")
p("4.1  行为克隆与 DAgger", "CNH2")
p("collect.py 让四个规则 AI 对局，只把出牌决策和至少有两个合法响应的响应决策写入 NPZ；Y 是规则 AI 动作。train_sl.py 先按 G 局号划分训练与验证，再按 T 选择相应动作头。一个 minibatch 的损失为存在的出牌交叉熵 + 存在的响应交叉熵 + 0.5 × MSE(V,R)，其中 R 为该座位终局分/50；使用 Adam，默认 batch=512、lr=3e-4。保存验证总损失最低的权重。")
eq("<i>L</i><sub>BC</sub> = <i>L</i><sub>CE,D</sub> + <i>L</i><sub>CE,C</sub> + 0.5 <i>L</i><sub>MSE</sub>(<i>V</i>, <i>R</i>)", "4-1")
p("dagger.py 每局轮换两个模型座，另外两座由规则专家行动。在模型座的决策点，模型动作推进 Game，规则专家在同一访问状态产生监督标签 Y；因此收集的是模型分布下的专家标签。它保留与 BC 相同的 X/D/C/Y/T/R/G/S 字段，之后显式合并 sl.npz 与 dagger.npz 重新训练。当前实现是一次数据聚合，并非多轮迭代式 DAgger。")
p("4.2  PPO 采样与更新", "CNH2")
add_figure(fig_ppo_detail(), "图 4-2  PPO 的采样、transition 缓冲、优势计算和参数更新")
p("RolloutAgent 只对出牌和响应采样：从相应掩码头建立 Categorical 分布，记录 (X,D,C,T,a,old logp,V,seat)。同局另两座的对手默认是规则 AI，--league 可追加历史模型。每局结束后，把该座位最终得分 / 50 裁剪到 [-5,5]，赋给该座位的所有已记录决策点；没有逐步奖励。")
p("更新时先计算 A = R - V_old，并在整批 transition 上标准化。对每个 minibatch 重新计算 logp_new，取 r = exp(logp_new - logp_old)；策略损失为 -mean(min(rA, clip(r,1-ε,1+ε)A))，ε=0.2。总损失再加 0.5 × MSE(V_new,R)，减 0.01 × 当前动作头的熵；默认训练 4 epoch，minibatch=512，Adam lr=3e-5，梯度范数裁剪 0.5。这里的 A 是终局回报减价值基线，不是 GAE；也没有折扣回报、KL 早停或动态对手池更新。熵项概率保留在计算图中，使熵正则参与策略梯度。")
eq("<i>R</i><sub>s</sub> = clip(score<sub>s</sub>/50, -5, 5), &nbsp; <i>A</i><sub>t</sub> = <i>R</i><sub>s(t)</sub> - <i>V</i><sub>old,t</sub>", "4-2")
eq("<i>r</i><sub>t</sub> = exp(log π<sub>new</sub>(<i>a</i><sub>t</sub>|<i>x</i><sub>t</sub>) - log π<sub>old</sub>(<i>a</i><sub>t</sub>|<i>x</i><sub>t</sub>))", "4-3")
eq("<i>L</i><sub>PPO</sub> = -mean[min(<i>rA</i>, clip(<i>r</i>,0.8,1.2)<i>A</i>)] + 0.5 MSE(<i>V</i>,<i>R</i>) - 0.01 <i>H</i>", "4-4")
p("5  实验记录与可解释范围", "CNH1")
p("已有文档记录使用 300 局、每局两模型座对两规则座、按局轮换座位的评测。指标 Δ = 模型座平均分 - 基线座平均分；越接近 0 或转正表示模型越强。以下数值来自 docs/M1_TRAINING.md 与 docs/M2_RL.md 的既有实验记录，本次未重新训练或复跑 300 局评测。")
table("表 5-1 历史评测点估计", ["策略", "记录中的 Δ（分/座）", "说明"], [
    ["SL 纯行为克隆", "-1.13", "5000 局规则 AI 样本"],
    ["SL + PPO", "-0.927", "10 × 24 局，旧实验"],
    ["SL + DAgger", "-0.69", "额外 600 局专家重标"],
    ["DAgger + PPO", "-1.037", "15 × 32 局，带历史对手"],
], [192, 120, 168], keep=True)
p("在已记录的四组比较里，DAgger 模型的得分差最高；相对 BC 缩小 0.44 分/座。DAgger 后继续 PPO 的结果较差，但这些 300 局点估计没有置信区间，不能据此断言方法本身无效。验证集动作一致率只衡量对规则专家标签的拟合，不等于实战强度。旧稿的学习曲线只有脚本中硬编码的数值，仓库未留原始逐轮日志，因此本版不展示该曲线。", "CNBody")
p("6  复现、限制与后续工作", "CNH1")
p("复现入口：python scripts/run_tests.py；python training/collect_batch.py --games 5000 --shard-size 1000 --out data/sl.npz；python training/train_sl.py --data data/sl.npz --epochs 12；python training/dagger.py --init checkpoints/sl_policy.pt --games 600 --out data/dagger.npz；python training/evaluate.py --checkpoint checkpoints/sl_dagger.pt --games 300。若重训 DAgger 模型，train_sl.py 需显式指定 --data data/sl.npz,data/dagger.npz --out checkpoints/sl_dagger.pt。")
p("主要限制：规则配置尚需按目标玩法核对；PPO 采样量小且没有时序优势估计；评测未报告多种子波动或置信区间；训练脚本的检查点只存模型权重与结构参数，缺少完整实验元数据。服务化、批量推理、灰度发布和房间管理目前停留在设计文档。后续宜先完善评测记录与多种子比较，再扩大采样并调整 PPO，最后处理部署。")
p("依据：仓库 src/sichuan_mahjong/*.py、training/*.py、data/*.npz 的字段/形状，以及 docs/M1_TRAINING.md、docs/M2_RL.md 中的历史实验记录。", "CNNote")
p("后续复现应保存采样命令、随机种子、规则配置、代码版本、数据哈希、训练/验证局号与最优轮次；评测则应留存每局四座得分、座位分配和异常结束标记。当前 NPZ 中保留了 G/S 字段，但检查点缺少完整实验元数据。", "CNBody")
p("附图索引", "CNH1")
p("随正文附有七页 A3 横向矢量附图，可放大查看模块端口与内部结构：A1 总体框架及输入输出；A2 规则状态机与规则策略；A3 404 维编码和动作掩码；A4 三头 MLP 与推理路由；A5 BC 和单轮 DAgger；A6 PPO 采样与更新；A7 2v2 评测。附图的张量形状、动作映射和公式均按当前代码标注。")


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e2e8"))
    canvas.line(30 * mm, 19 * mm, 185 * mm, 19 * mm)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(30 * mm, 13 * mm, "阶段性技术总结")
    canvas.drawCentredString(105 * mm, 13 * mm, f"— {doc.page} —")
    canvas.restoreState()


if __name__ == "__main__":
    from pypdf import PdfReader, PdfWriter
    from detailed_figures import build as build_detail_figures
    tmp_dir = ROOT / "tmp" / "pdfs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    body_pdf = tmp_dir / "report_body.pdf"
    figures_pdf = tmp_dir / "report_figures.pdf"
    doc = SimpleDocTemplate(str(body_pdf), pagesize=(210 * mm, 297 * mm),
                            leftMargin=30 * mm, rightMargin=25 * mm,
                            topMargin=25 * mm, bottomMargin=25 * mm,
                            title="阶段性技术总结",
                            author="MaJiangAI Project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    build_detail_figures(figures_pdf)
    writer = PdfWriter()
    for source in (body_pdf, figures_pdf):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": "阶段性技术总结", "/Author": "MaJiangAI Project"})
    with OUT.open("wb") as stream:
        writer.write(stream)
    body_pdf.unlink()
    figures_pdf.unlink()
    print(OUT)
