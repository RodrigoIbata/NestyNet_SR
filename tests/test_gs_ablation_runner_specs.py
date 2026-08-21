# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Pin the gs_ablation runner specs to flags their target scripts accept.

Regression guard for the bug where the feynman_de experiment specs appended
``--gs-enable``-family flags to ``examples/feynman_de/run_benchmark.py``,
whose argparse registers no such options, so the GS-variant arms crashed
before running. Any ``gs_args`` flag must appear literally in the target
script's source (argparse registrations are literal strings there), and the
env channel the GS variant actually relies on must stay consumed by
``run_de.py``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "gs_ablation"))

import runner  # noqa: E402  (examples/gs_ablation/runner.py)


def _target_script(spec) -> Path | None:
    for token in spec.command:
        if str(token).endswith(".py"):
            return REPO_ROOT / str(token)
    return None


def test_gs_args_are_registered_by_their_target_scripts():
    for name, spec in runner.EXPERIMENTS.items():
        if not spec.gs_args:
            continue
        script = _target_script(spec)
        assert script is not None and script.exists(), (name, spec.command)
        src = script.read_text()
        for flag in spec.gs_args:
            base = str(flag).split("=", 1)[0]
            assert base in src, (
                f"experiment {name!r} passes {base!r} to {script.name}, "
                "which does not register it; use the NESTYNET_GS_* env "
                "channel instead (see _variant_env)"
            )


def test_run_de_consumes_the_gs_variant_env_channel():
    src = (REPO_ROOT / "nestynet_sr" / "run_de.py").read_text()
    for env_name in (
        "NESTYNET_GS_ENABLE",
        "NESTYNET_DE_HARD_TAIL_TEMPLATES",
        "NESTYNET_DE_HARD_TAIL_VELOCITY_TEMPLATES",
    ):
        assert env_name in src, f"run_de.py no longer consumes {env_name}"


def test_feynman_specs_carry_no_cli_gs_args():
    for name in ("feynman_de", "feynman_de_coe"):
        assert runner.EXPERIMENTS[name].gs_args == (), (
            f"{name} must rely on the env channel; run_benchmark.py has no "
            "GS flags"
        )


def test_variant_env_sets_the_channel():
    env = runner._variant_env("gs", "auto", Path("/tmp/x"))
    assert env.get("NESTYNET_GS_ENABLE") == "1"
    assert env.get("NESTYNET_DE_HARD_TAIL_TEMPLATES") == "1"
    assert env.get("NESTYNET_DE_HARD_TAIL_VELOCITY_TEMPLATES") == "1"
