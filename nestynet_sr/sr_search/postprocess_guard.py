# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Subprocess airbags for heavy symbolic post-processing.

SymPy and post-hoc expression polishing can occasionally consume unbounded
memory.  A Python ``try`` block cannot catch an OS-level SIGKILL, so expensive
post-processing runs in a short-lived child process with optional resource
limits.  The main SR process keeps the accepted Stage-B state and records a
safe failure if the worker times out or is killed.
"""

from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import os
import queue
import signal
import traceback
from typing import Any


def physical_memory_bytes() -> int | None:
    """Best-effort physical RAM size for memory-limit calculations."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(pages) * int(page_size)
        if total > 0:
            return total
    except Exception:
        pass
    return None


def memory_limit_from_fraction(mem_fraction: float | None) -> int | None:
    """Return a byte limit from a physical-RAM fraction, or ``None``."""
    try:
        frac = float(mem_fraction)
    except Exception:
        return None
    if not math.isfinite(frac) or frac <= 0.0:
        return None
    total = physical_memory_bytes()
    if total is None:
        return None
    return max(256 * 1024 * 1024, int(total * min(frac, 1.0)))


def _apply_worker_limits(memory_bytes: int | None, cpu_seconds: int | None) -> dict[str, Any]:
    """Apply POSIX rlimits where available; return diagnostic metadata."""
    meta: dict[str, Any] = {
        "memory_limit_bytes": memory_bytes,
        "cpu_limit_seconds": cpu_seconds,
        "rlimit_as_applied": False,
        "rlimit_cpu_applied": False,
    }
    try:
        import resource
    except Exception as exc:  # pragma: no cover - platform dependent
        meta["resource_error"] = str(exc)
        return meta

    if memory_bytes is not None and memory_bytes > 0:
        try:
            hard = resource.getrlimit(resource.RLIMIT_AS)[1]
            lim = int(memory_bytes)
            if hard not in (-1, resource.RLIM_INFINITY):
                lim = min(lim, int(hard))
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            meta["memory_limit_bytes"] = lim
            meta["rlimit_as_applied"] = True
        except Exception as exc:
            meta["rlimit_as_error"] = str(exc)

    if cpu_seconds is not None and cpu_seconds > 0:
        try:
            hard = resource.getrlimit(resource.RLIMIT_CPU)[1]
            lim = int(max(1, cpu_seconds))
            if hard not in (-1, resource.RLIM_INFINITY):
                lim = min(lim, int(hard))
            resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
            meta["cpu_limit_seconds"] = lim
            meta["rlimit_cpu_applied"] = True
        except Exception as exc:
            meta["rlimit_cpu_error"] = str(exc)

    return meta


def _resolve_function(function_path: str):
    if ":" not in function_path:
        raise ValueError(f"function_path must be 'module:function', got {function_path!r}")
    module_name, function_name = function_path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = module
    for part in function_name.split("."):
        fn = getattr(fn, part)
    return fn


def _worker_entry(
    q,
    function_path: str,
    kwargs: dict[str, Any],
    memory_bytes: int | None,
    cpu_seconds: int | None,
) -> None:
    limit_meta = _apply_worker_limits(memory_bytes, cpu_seconds)
    try:
        fn = _resolve_function(function_path)
        result = fn(**kwargs)
        q.put({"ok": True, "result": result, "limit_meta": limit_meta})
    except BaseException as exc:  # pragma: no cover - defensive worker boundary
        q.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
                "limit_meta": limit_meta,
            }
        )


def _exit_status(returncode: int | None) -> tuple[str, str]:
    if returncode is None:
        return "unknown", "worker exited with unknown return code"
    if returncode == 0:
        return "ok", "worker exited cleanly"
    if returncode < 0:
        sig = -int(returncode)
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = f"SIG{sig}"
        return "signal", f"worker killed by {sig_name}"
    return "exit", f"worker exited with code {returncode}"


def run_guarded_function(
    function_path: str,
    *,
    kwargs: dict[str, Any] | None = None,
    max_seconds: float = 300.0,
    mem_fraction: float | None = 0.20,
    memory_bytes: int | None = None,
    label: str = "postprocess",
) -> dict[str, Any]:
    """Run an importable function in a resource-limited subprocess.

    Returns a small JSON-friendly dict.  On success, ``result`` contains the
    function return value.  On timeout, crash, or SIGKILL, ``ok`` is false and
    the caller can keep its conservative in-process state.
    """
    kwargs = dict(kwargs or {})
    try:
        timeout = float(max_seconds)
    except Exception:
        timeout = 300.0
    if not math.isfinite(timeout) or timeout <= 0.0:
        timeout = 300.0

    mem_bytes = memory_bytes
    if mem_bytes is None:
        mem_bytes = memory_limit_from_fraction(mem_fraction)

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    cpu_seconds = int(math.ceil(timeout + 5.0))
    proc = ctx.Process(
        target=_worker_entry,
        args=(q, function_path, kwargs, mem_bytes, cpu_seconds),
        name=f"{label}_worker",
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(2.0)
        return {
            "ok": False,
            "status": "timeout",
            "reason": f"{label} worker exceeded {timeout:.1f}s",
            "returncode": proc.exitcode,
            "memory_limit_bytes": mem_bytes,
            "max_seconds": timeout,
        }

    payload = None
    try:
        payload = q.get_nowait()
    except queue.Empty:
        payload = None
    except Exception as exc:
        payload = {"ok": False, "status": "queue_error", "error": str(exc)}

    if payload is not None:
        if payload.get("ok"):
            payload.setdefault("status", "success")
            payload.setdefault("returncode", proc.exitcode)
            payload.setdefault("memory_limit_bytes", mem_bytes)
            payload.setdefault("max_seconds", timeout)
            return payload
        payload.setdefault("status", "error")
        payload.setdefault("returncode", proc.exitcode)
        payload.setdefault("memory_limit_bytes", mem_bytes)
        payload.setdefault("max_seconds", timeout)
        return payload

    status, reason = _exit_status(proc.exitcode)
    return {
        "ok": False,
        "status": status,
        "reason": reason,
        "returncode": proc.exitcode,
        "memory_limit_bytes": mem_bytes,
        "max_seconds": timeout,
    }
