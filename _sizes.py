from pathlib import Path

root = Path(r"D:\Joaquin\MaJiangAI")
for name in ["data", "checkpoints", "paper", ".idea", "docs", "src", "tests", "training", "scripts"]:
    p = root / name
    if not p.exists():
        continue
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"{name:12s} {total / 1e6:8.2f} MB")

print("\nTop files:")
files = []
for f in root.rglob("*"):
    if f.is_file() and ".git" not in f.parts:
        files.append((f.stat().st_size, f.relative_to(root)))
for size, rel in sorted(files, reverse=True)[:15]:
    print(f"{size / 1e6:8.2f} MB  {rel}")
