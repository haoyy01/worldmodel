from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import re


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(1)

    stdout = result.stdout
    if "Epoch 1" not in stdout:
        raise SystemExit("training output missing")
    if "risk_curve=" not in stdout:
        raise SystemExit("risk output missing")
    if "recommended_speed=" not in stdout:
        raise SystemExit("planning output missing")
    if "behavior=" not in stdout:
        raise SystemExit("behavior planner output missing")
    if "steer_cmd=" not in stdout:
        raise SystemExit("controller output missing")
    if "safety_state=" not in stdout:
        raise SystemExit("safety supervisor output missing")
    if "nan" in stdout.lower():
        raise SystemExit("numeric instability detected")
    if "RuntimeWarning" in result.stderr:
        raise SystemExit("runtime warning detected")
    steer_values = [float(value) for value in re.findall(r"steer_hint=([-+]?\d+(?:\.\d+)?)", stdout)]
    brake_values = [float(value) for value in re.findall(r"brake_hint=([-+]?\d+(?:\.\d+)?)", stdout)]
    speed_values = [float(value) for value in re.findall(r"recommended_speed=([-+]?\d+(?:\.\d+)?)", stdout)]
    curve_risk_values = [float(value) for value in re.findall(r"risk_curve=([-+]?\d+(?:\.\d+)?)", stdout)]
    steer_cmd_values = [float(value) for value in re.findall(r"steer_cmd=([-+]?\d+(?:\.\d+)?)", stdout)]
    brake_cmd_values = [float(value) for value in re.findall(r"brake_cmd=([-+]?\d+(?:\.\d+)?)", stdout)]
    event_matches = re.findall(r"events=\{([^}]*)\}", stdout)
    if not steer_values or not brake_values or not speed_values or not curve_risk_values or not steer_cmd_values or not brake_cmd_values:
        raise SystemExit("control values missing")
    if any(abs(value) > 1.0 for value in steer_values):
        raise SystemExit("steer out of range")
    if any(value < 0.0 or value > 1.0 for value in brake_values):
        raise SystemExit("brake out of range")
    if max(steer_values) - min(steer_values) < 0.05:
        raise SystemExit("steer output is effectively constant")
    if max(brake_values) - min(brake_values) < 0.01:
        raise SystemExit("brake output is effectively constant")
    if any(value < 0.0 for value in speed_values):
        raise SystemExit("recommended speed out of range")
    if any(value < 0.0 or value > 1.0 for value in curve_risk_values):
        raise SystemExit("curve risk out of range")
    if any(abs(value) > 1.0 for value in steer_cmd_values):
        raise SystemExit("steer command out of range")
    if any(value < 0.0 or value > 1.0 for value in brake_cmd_values):
        raise SystemExit("brake command out of range")
    if not any(match.strip() for match in event_matches):
        raise SystemExit("no events triggered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
