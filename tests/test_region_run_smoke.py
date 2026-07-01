from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "region_run.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(1)

    stdout = result.stdout
    if "Epoch 1" not in stdout:
        raise SystemExit("training output missing")
    if "dispatch_benefit=" not in stdout:
        raise SystemExit("benefit output missing")
    if "mode=" not in stdout:
        raise SystemExit("policy output missing")
    if "safety_state=" not in stdout:
        raise SystemExit("safety output missing")
    if "nan" in stdout.lower():
        raise SystemExit("numeric instability detected")
    benefit_values = [float(v) for v in re.findall(r"dispatch_benefit=([-+]?\d+(?:\.\d+)?)", stdout)]
    if not benefit_values:
        raise SystemExit("no benefit values parsed")
    if max(map(abs, benefit_values)) < 1e-6:
        raise SystemExit("benefit is effectively zero")
    return 0


def test_smoke_run():
    rc = main()
    assert rc == 0, f"region_run.py exited with {rc}"


if __name__ == "__main__":
    raise SystemExit(main())