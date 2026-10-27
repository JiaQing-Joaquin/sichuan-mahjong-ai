# 阶段性技术总结

当前 PDF 为 `阶段性技术总结.pdf`，由 `build_report.py` 生成正文并合并 `detailed_figures.py` 生成的七页 A3 横向矢量附图。附图参考用户提供的 RSRT-DETR 方法总图的视觉语言：浅色模块、虚线分组、黑色数据流箭头和局部放大。A1 展示总体端口关系，A2-A7 依次展开规则状态机、404 维编码与掩码、三头 MLP 和推理路由、BC/DAgger、PPO、评测。图示内容按本项目现有代码绘制。结果表引用 `docs/M1_TRAINING.md`、`docs/M2_RL.md` 的历史实验记录；仓库没有对应的逐局评测日志，故报告明确标注为历史点估计。

正文采用通行的中文研究生论文版式：A4、左 30 mm／右 25 mm／上下 25 mm 页边距，宋体正文、黑体标题、章节编号、居中的图题与表题、三线表，以及章内编号的居中公式。跨页表重复表头并以“续表”标识。附图为 A3 横向页面，便于展示微观网络与端口，图题居中。具体学校若有专用模板，应以该校模板的字号、行距、封面和参考文献规定为准。

## 重新构建

需要 Python、ReportLab 和 Windows 字体 `C:\Windows\Fonts\simhei.ttf`：

```powershell
python paper/build_report.py
```

旧版 `build_paper.py`、`make_figures.py`、`finalize_pdf.py`、`figures/` 和 DOCX 保留作为历史源稿，不再作为当前 PDF 的构建入口。其中旧图的服务层属于规划内容，学习曲线为脚本内固定数值，不能当作当前实现或原始日志使用。
