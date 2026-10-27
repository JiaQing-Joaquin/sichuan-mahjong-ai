"""Run every tests/test_*.py module and aggregate the result.

Usage::

    python scripts/run_tests.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

failed = 0
ran = 0
for path in sorted((ROOT / "tests").glob("test_*.py")):
    ran += 1
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        import traceback

        print(f"[ERROR] {path.name} failed to import")
        traceback.print_exc()
        failed += 1
        continue
    print(f"--- {path.name}")
    rc = module.main()
    if rc != 0:
        failed += 1

print(f"\n{ran} module(s), {failed} failed")
raise SystemExit(1 if failed else 0)
