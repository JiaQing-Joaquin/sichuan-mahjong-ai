import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "paper" / "report.pdf"
dst = root / "paper" / "川麻血战到底AI技术报告.pdf"
shutil.copyfile(src, dst)
data = dst.read_bytes()
print("size_bytes", len(data))
print("signature", data[:8])
