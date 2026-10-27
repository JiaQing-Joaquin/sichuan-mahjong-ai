"""Build code-grounded, vector A3 detail plates for the technical report.

Each plate expands one executable subsystem and labels its input/output ports.
The drawings deliberately follow the supplied paper figure's visual grammar,
while all module names, tensor shapes and update equations come from this repo.
"""

from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT = "DetailCN"
pdfmetrics.registerFont(TTFont(FONT, r"C:\Windows\Fonts\simhei.ttf"))
W, H = landscape(A3)
INK = colors.HexColor("#202a34")
MUTED = colors.HexColor("#536373")
ORANGE = colors.HexColor("#ba5927")
BLUE = colors.HexColor("#dcecf7")
GREEN = colors.HexColor("#dcf3e9")
YELLOW = colors.HexColor("#f8efd4")
PINK = colors.HexColor("#f6e4df")
WHITE = colors.white


def txt(c, x, y, value, size=12, color=INK):
    c.setFillColor(color); c.setFont(FONT, size); c.drawString(x, y, value)


def box(c, x, y, w, h, title, lines=(), fill=BLUE, size=11):
    c.setFillColor(fill); c.setStrokeColor(INK); c.setLineWidth(.8)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    txt(c, x+10, y+h-22, title, 14, ORANGE)
    for i, line in enumerate(lines):
        txt(c, x+10, y+h-43-i*17, line, size)


def arrow(c, x1, y1, x2, y2, label="", size=10):
    from math import atan2, cos, sin, pi
    c.setStrokeColor(INK); c.setFillColor(INK); c.setLineWidth(1.3)
    c.line(x1,y1,x2,y2)
    a=atan2(y2-y1,x2-x1); r=7
    p=c.beginPath(); p.moveTo(x2,y2)
    p.lineTo(x2-r*cos(a-pi/6),y2-r*sin(a-pi/6))
    p.lineTo(x2-r*cos(a+pi/6),y2-r*sin(a+pi/6)); p.close()
    c.drawPath(p,fill=1,stroke=0)
    if label: txt(c, (x1+x2)/2+4,(y1+y2)/2+6,label,size,MUTED)


def frame(c, x, y, w, h, label):
    c.setStrokeColor(colors.HexColor("#9aa8b4")); c.setDash(5,4)
    c.roundRect(x,y,w,h,10,fill=0,stroke=1); c.setDash()
    txt(c,x+8,y+h+8,label,12,ORANGE)


def page(c, num, title, subtitle):
    c.setFillColor(WHITE); c.rect(0,0,W,H,fill=1,stroke=0)
    txt(c,42,H-48,f"A{num}  {title}",25,INK)
    txt(c,43,H-72,subtitle,11,MUTED)
    c.setStrokeColor(colors.HexColor("#c9d2d9")); c.line(42,53,W-42,53)
    txt(c,43,34,"阶段性技术总结 | 结构附图 | 矢量图，可放大查看",9,MUTED)
    txt(c,W-90,34,f"A{num} / 7",9,MUTED)


def p1(c):
    page(c,1,"总体框架与端口联系","实线箭头为运行时数据流；下层虚线组为训练/评测回路。输入形状以单个决策点计。")
    frame(c,43,365,1104,374,"在线决策：Game.step(action) 循环推进一局")
    box(c,58,505,225,180,"Game / 规则状态",["输入：Action、随机种子、规则配置","内部：阶段/牌墙/四座手牌/公开牌","输出：view(seat)、legal_actions()","step 后更新状态与分数"],YELLOW)
    box(c,335,505,214,180,"Encoder",["输入：Game、当前 seat","输出：X: R^404","D: binary[27]；C: binary[4]","只读取本手牌及公开信息"],GREEN)
    box(c,603,505,226,180,"PolicyValueNet",["输入：X、D、C","主干：404->512->512->256","输出：D logits[27] / C logits[4]","及状态价值 V[1]"],BLUE)
    box(c,882,505,249,180,"ModelAI / 决策路由",["REACTION->响应头 argmax","纯 DISCARD->出牌头 argmax","换牌/定缺/自摸/杠->BaselineAI","输出：合法 Action"],PINK)
    arrow(c,283,590,335,590,"状态+合法集"); arrow(c,549,590,603,590,"X,D,C")
    arrow(c,829,590,882,590,"logits,V")
    arrow(c,1005,505,1005,450); arrow(c,1005,450,170,450)
    arrow(c,170,450,170,505,"Action->step")
    txt(c,59,390,"核心边界：规则负责动作合法性和转移；网络只对出牌与弃牌响应排序。",12)
    frame(c,43,100,1104,222,"离线学习与评测：从状态/行动轨迹产生监督和策略更新")
    box(c,59,143,236,137,"collect / BC",["四规则座自对弈","(X,D,C,Y,T,R,G,S)->sl.npz","按局 G 分割；三项监督损失"],GREEN)
    box(c,331,143,237,137,"DAgger",["模型座行动推进环境","专家在同一访问状态标注 Y","合并 sl+dagger 后再训练"],YELLOW)
    box(c,604,143,237,137,"PPO",["两策略座采样 + 两对手座","终局 R；A=R-V_old","裁剪目标 + 价值 + 熵"],PINK)
    box(c,877,143,253,137,"evaluate",["两模型座 vs 两规则座","每局轮换座位；记录四座分数","输出 Delta：模型均分-基线均分"],BLUE)
    arrow(c,295,211,331,211,"BC 权重")
    arrow(c,568,211,604,211,"可选初始化")
    arrow(c,841,211,877,211,"检查点")
    arrow(c,170,365,170,281,"轨迹"); arrow(c,1006,365,1006,281,"局面/得分")


def p2(c):
    page(c,2,"规则状态机与规则策略内部结构","每个转移由 Game.step(Action) 校验；合法集来自 Game.legal_actions()。抢杠胡是已实现分支。")
    frame(c,43,392,1104,342,"状态机与响应仲裁")
    phases=[(60,"SWAP / 换三张"),(328,"MISSING / 定缺"),(596,"TURN / 回合"),(864,"REACTION / 响应")]
    for x,t in phases:
        lines={"SWAP / 换三张":["四座各选三张；按规则方向交换","收齐后进入定缺"],"MISSING / 定缺":["四座声明缺门","收齐后发起摸牌/行动"],"TURN / 回合":["摸牌->可自摸/杠/弃牌","弃牌后建立待响应牌"],"REACTION / 响应":["收集各座合法响应","胡优先；再明杠/碰；否则过"]}[t]
        box(c,x,545,225,133,t,lines,YELLOW if x<596 else BLUE)
    for x in (285,553,821): arrow(c,x,611,x+43,611)
    arrow(c,975,545,975,505); arrow(c,975,505,708,505); arrow(c,708,505,708,545,"响应处理后继续")
    box(c,67,420,445,70,"特殊终局 / 续打",["胡牌座退出；其余座继续。活跃人数、牌墙和流局条件决定 OVER。"],GREEN)
    box(c,548,420,561,70,"TURN 内的抢杠胡分支",["补杠触发抢杠窗口时仅允许 HU / PASS；无人胡后完成补杠。"],PINK)
    frame(c,43,96,1104,259,"BaselineAI.choose(game, seat, legal_actions) 微观决策")
    cards=[(58,"换牌",["候选牌连接性排序","选低连接性的三张"]),(278,"定缺",["比较各花色手牌强度","选弱门"]),(498,"胡/杠",["合法自摸胡、点炮胡优先","可杠时按规则处理"]),(718,"碰",["比较碰前后向听数","仅在规则条件满足时碰"]),(938,"弃牌",["按向听/连接性/未见张数","及随机平局项排序"])]
    for x,t,lines in cards: box(c,x,163,195,139,t,lines,GREEN,10)
    txt(c,65,119,"各决策支路按当前阶段与合法动作集选用，返回 Action -> Game.step；网络回退也使用同一接口。",12)


def p3(c):
    page(c,3,"404 维状态编码与独立动作掩码","左侧为精确半开区间 [start:end)；特征来自本人可见信息，不读取对手暗牌。")
    frame(c,43,126,730,603,"encode_state(game, seat) -> X[404]")
    rows=[("0:27","本人手牌","count/4"),("27:54","本人碰掩码","0/1"),("54:81","本人杠掩码","0/1"),("81:189","四座弃牌","4x27，count/4"),("189:297","四座明副露牌","4x27，count/4"),("297:309","四座定缺","4x3，one-hot"),("309:310","牌墙剩余","length/55"),("310:314","四座分数","score/50"),("314:318","四座已胡","0/1"),("318:345","本人刚摸牌","27，one-hot"),("345:346","响应阶段标记","0/1"),("346:373","待响应弃牌","27，one-hot"),("373:377","弃牌者相对座位","(discarder-seat)%4"),("377:404","每种牌剩余张数","max(0,4-visible)/4")]
    for i,(rng,name,desc) in enumerate(rows):
        y=681-i*39
        c.setFillColor(GREEN if i%2 else BLUE); c.setStrokeColor(INK)
        c.roundRect(59,y-21,697,31,5,fill=1,stroke=1)
        txt(c,67,y-10,f"[{rng})",11,ORANGE); txt(c,177,y-10,name,11)
        txt(c,430,y-10,desc,10,MUTED)
    frame(c,808,406,337,323,"由 legal_actions() 独立生成")
    box(c,827,611,298,82,"D[27] 出牌掩码",["D[t]=1 iff 存在 DISCARD(tile=t)","非法 t 在网络中 masked_fill 为极小值"],YELLOW,10)
    box(c,827,503,298,82,"C[4] 响应掩码",["顺序：胡 / 碰 / 明杠 / 过","C[k]=1 iff 对应 kind 在合法集中"],PINK,10)
    box(c,827,421,298,59,"阶段与端口",["D、C 均不占 X 的 404 维。"],GREEN,10)
    frame(c,808,126,337,239,"剩余张数子图")
    box(c,828,256,296,71,"可见张数 visible[t]",["本人手牌 + 四座弃牌 + 四座明副露"],BLUE,10)
    arrow(c,976,256,976,218)
    box(c,828,150,296,66,"remaining[t]",["max(0, 4-visible[t]) / 4"],YELLOW,11)


def p4(c):
    page(c,4,"三头 MLP 与 ModelAI 决策路由","层宽、ReLU、输出和掩码操作均按 model.py / policy_ai.py；当前 hidden=512，总参数 609568。")
    frame(c,43,353,1104,377,"PolicyValueNet.forward(X,D,C) 的逐层结构")
    layers=[(58,92,"X",["float32 [B,404]"]),(181,160,"Linear 1",["404->512","ReLU"]),(376,160,"Linear 2",["512->512","ReLU"]),(571,160,"Linear 3",["512->256","ReLU"])]
    for x,w,t,l in layers: box(c,x,550,w,101,t,l,BLUE)
    arrow(c,150,600,181,600); arrow(c,341,600,376,600); arrow(c,536,600,571,600)
    box(c,783,627,333,79,"出牌头 / discard_head",["Linear 256->27；~D 位 masked_fill(dtype.min)"],GREEN,11)
    box(c,783,526,333,79,"响应头 / call_head",["Linear 256->4；~C 位 masked_fill(dtype.min)"],YELLOW,11)
    box(c,783,425,333,79,"价值头 / value_head",["Linear 256->1；squeeze(-1) -> V[B]"],PINK,11)
    arrow(c,731,600,783,665); arrow(c,731,600,783,565); arrow(c,731,600,783,465)
    txt(c,60,397,"三个头共享 256 维表示；掩码只作用于两个动作 logits，价值头不受掩码影响。",12)
    frame(c,43,108,1104,211,"ModelAI.act 路由与输出 Action")
    box(c,58,158,240,115,"读取 Game",["phase、current_player","legal_actions() 的动作 kind 集"],BLUE)
    box(c,340,158,241,115,"REACTION",["encode->call logits->argmax","索引映射 胡/碰/明杠/过","若映射无效，回退 BaselineAI"],GREEN,10)
    box(c,623,158,241,115,"仅含 DISCARD",["encode->discard logits->argmax","检查 D[t]；必要时选首个合法 t"],YELLOW,10)
    box(c,906,158,223,115,"其他阶段",["换牌/定缺/自摸/杠","直接 BaselineAI.choose"],PINK,10)
    arrow(c,298,215,340,215)
    c.line(180,273,180,291); c.line(180,291,740,291); arrow(c,740,291,740,273)
    c.line(180,291,180,300); c.line(180,300,1018,300); arrow(c,1018,300,1018,273)
    txt(c,72,126,"三路最终都返回 Action，经 Game.step 校验并推进局面；推理阶段不使用价值头。",11)


def p5(c):
    page(c,5,"BC 与单轮 DAgger 的微观数据流","监督标签 Y 来自规则专家；模型访问状态上的专家重标是 DAgger 的关键。")
    frame(c,43,427,1104,302,"规则自对弈收集 -> 行为克隆 BC")
    box(c,58,543,205,139,"四座 BaselineAI",["Game 每个决策点产生合法集","仅出牌或 >=2 合法响应入样","专家动作 -> Y；动作类型 -> T"],GREEN,10)
    box(c,304,543,205,139,"样本字段",["X[404],D[27],C[4]","Y,T；终局 R=score/50","G=局号，S=座位；写 sl.npz"],BLUE,10)
    box(c,550,543,205,139,"按局切分",["按 G 分训练 / 验证","相同一局不跨集合","minibatch=512，Adam 3e-4"],YELLOW,10)
    box(c,796,543,332,139,"监督损失与检查点",["L=CE_discard (T=0)+CE_call (T=1)","  +0.5xMSE(V,R)；只计算存在的类型","保存验证总损失最低的权重"],PINK,10)
    for x in (263,509,755): arrow(c,x,612,x+41,612)
    txt(c,63,463,"说明：响应样本过滤掉只有一个合法响应的决策点；R 在终局后回填至本座该局样本。",11)
    frame(c,43,111,1104,282,"模型分布访问状态 -> 专家重标 -> 合并再训练")
    box(c,58,221,205,126,"模型座轮换",["每局两模型座、两专家座","ModelAI 的动作推进 Game"],BLUE,10)
    box(c,304,221,205,126,"同状态专家查询",["在模型访问的同一局面","BaselineAI 给出标签 Y","不以专家动作推进模型座"],GREEN,10)
    box(c,550,221,205,126,"新增样本",["X,D,C,Y,T,R,G,S","写 dagger.npz","当前实现仅一次聚合"],YELLOW,10)
    box(c,796,221,332,126,"聚合与再训练",["sl.npz + dagger.npz 显式合并","重新按 G 切分；同一监督目标","输出 DAgger 检查点"],PINK,10)
    for x in (263,509,755): arrow(c,x,281,x+41,281)
    txt(c,63,142,"DAgger 的数据分布来自模型实际行动；标签仍由规则专家产生。",11)


def p6(c):
    page(c,6,"PPO 采样、优势与裁剪更新","按当前 ppo.py：终局座位得分回填、无 GAE；熵正则保留梯度。")
    frame(c,43,434,1104,295,"rollout：模型座策略采样 -> transition 缓冲")
    box(c,59,545,212,135,"两策略座 / 两对手座",["Game 状态与 legal_actions","仅出牌或响应交给网络","其余由规则策略处理"],BLUE,10)
    box(c,315,545,212,135,"Masked Categorical",["T=0 取 27 维 D 头","T=1 取 4 维 C 头","采样动作 a；记录 old_logp"],GREEN,10)
    box(c,571,545,212,135,"每步 transition",["X,D,C,T,a,old_logp","V_old, seat；推进 Game","对手由规则/可选历史池行动"],YELLOW,10)
    box(c,827,545,301,135,"终局回填",["R=clip(该 seat 最终 score/50,-5,5)","同座每个决策点使用相同 R","A=R-V_old；整批标准化 A"],PINK,10)
    for x in (271,527,783): arrow(c,x,612,x+44,612)
    frame(c,43,115,1104,282,"update：4 epoch x minibatch 512；Adam lr=3e-5，梯度裁剪 0.5")
    box(c,59,236,248,111,"重算当前策略",["对同一 X,D,C,T,a 计算","logp_new、V_new、当前动作头熵"],BLUE,10)
    box(c,348,236,248,111,"重要性比率",["r=exp(logp_new-old_logp)","r_clip=clip(r,1-0.2,1+0.2)"],GREEN,10)
    box(c,637,236,248,111,"策略目标",["Lpi=-mean(min(rA,r_clip A))","每个样本按其 T 选对应动作头"],YELLOW,10)
    box(c,926,236,202,111,"总损失",["L=Lpi+0.5 MSE(V,R)","   -0.01 entropy"],PINK,10)
    for x in (307,596,885): arrow(c,x,291,x+41,291)
    txt(c,62,168,"实现边界：无折扣回报、GAE、KL 早停或动态对手池更新；熵项概率保留梯度。",11)


def p7(c):
    page(c,7,"评测模块与得分指标","evaluate.py 的单位是“分/模型座”；历史表格数值是既有记录，本图展示当前评测算法。")
    frame(c,43,263,1104,466,"逐局 2v2 座位轮换、动作执行与分数汇总")
    box(c,57,515,249,169,"局 k / 随机种子",["创建 Game(seed+k)","s0=k%4","模型座={s0,(s0+1)%4}","另外两座=规则 AI"],BLUE,11)
    box(c,358,515,249,169,"各决策点",["当前座选相应 ModelAI/BaselineAI","agent.act(game)->Action","Game.step；结束或步数>5000 停止"],GREEN,11)
    box(c,659,515,249,169,"单局结算",["获取四座 score","累加 model_score","累加 base_score"],YELLOW,11)
    arrow(c,306,603,358,603); arrow(c,607,603,659,603)
    box(c,951,515,181,169,"跨局汇总",["games=N","模型/基线各 2N 座","计算 Delta"],PINK,11)
    arrow(c,908,603,951,603)
    box(c,184,328,817,126,"指标定义",["Delta = (Sum模型座分数 - Sum规则座分数) / (2N)","每局换位减少固定座位偏差；正值表示模型座平均分高于基线座。"],BLUE,14)
    txt(c,65,285,"可复现记录应包含：每局 seed、四座身份及得分、规则配置、检查点、代码版本。",11)
    frame(c,43,113,1104,113,"本次报告的结果证据范围")
    txt(c,66,181,"既有实验文档只保留 300 局的点估计，未保存逐局明细；不能从现有资料计算置信区间。",12)
    txt(c,66,150,"该指标评估整套“网络决策 + BaselineAI 回退 + 规则引擎”的对局表现，而非孤立网络层的质量。",12)


def build(path: Path):
    c=canvas.Canvas(str(path),pagesize=(W,H),pageCompression=1)
    c.setTitle("阶段性技术总结｜模块结构附图")
    captions=("总体框架与端口联系", "规则状态机与规则策略内部结构",
              "404 维状态编码与独立动作掩码", "三头 MLP 与决策路由",
              "BC 与单轮 DAgger 数据流", "PPO 采样与裁剪更新",
              "评测模块与得分指标")
    for i,fn in enumerate((p1,p2,p3,p4,p5,p6,p7),1):
        fn(c)
        c.setFont(FONT, 13); c.setFillColor(INK)
        c.drawCentredString(W/2, 76, f"图 A-{i} {captions[i-1]}")
        c.showPage()
    c.save()


if __name__=="__main__":
    build(Path(__file__).with_name("模块结构附图.pdf"))
