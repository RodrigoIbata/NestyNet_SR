#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Launch the full DE-CoE control run as a detached process.

The shell wrapper is convenient for humans, but some execution environments
clean up background children when the parent shell exits.  Launching the worker
in a fresh session gives the overnight benchmark a separate process group.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main(argv: list[str]) -> int:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    os.chdir(root)

    stamp = os.environ.get("STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(
        os.environ.get("RESULTS_ROOT") or f"results/feynman_de_coe_full_adjudicate_{stamp}"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    log_path = results_root / "overnight_launcher.log"
    pid_path = results_root / "overnight.pid"
    worker = root / "examples" / "feynman_de_coe" / "run_full_adjudicate_control.sh"
    cmd = [str(worker), *argv]

    env = os.environ.copy()
    env["STAMP"] = stamp
    env["RESULTS_ROOT"] = str(results_root)
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("ab", buffering=0) as log:
        line = "$ " + " ".join(shlex.quote(part) for part in cmd) + "\n"
        log.write(line.encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print("Started DE-CoE full adjudicate control run")
    print(f"PID: {proc.pid}")
    print(f"Results root: {results_root}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_path}")
    print(f"Monitor: tail -f {log_path}")
    print(f"Stop: kill -TERM -{proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
