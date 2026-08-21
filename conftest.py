# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
"""Pin which NestyNet this workspace builds on.

The 2026-08-01 merge retired the *_plus* staging forks: this workspace now
pairs with the sibling ``NestyNet`` checkout.  Without this file ``import
nestynet`` resolves to whatever editable install happens to be active, which
fails silently when it differs: the package imports, the tests pass, and the
paired checkout is not actually exercised.

The pairing stays overridable (e.g. on the HPC, or to test another checkout):

    pytest                                  # against ../NestyNet (default)
    NESTYNET_BASE=/path/to/checkout pytest  # explicit pairing

This only affects pytest.  For CLI runs set ``PYTHONPATH`` to the same root.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT.parent / "NestyNet"


def _resolve_base() -> Path:
    override = os.environ.get("NESTYNET_BASE")
    base = Path(override).expanduser() if override else DEFAULT_BASE
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    return base.resolve()


BASE_ROOT = _resolve_base()
BASE_PKG = BASE_ROOT / "nestynet"


def _is_pinned_nestynet(module) -> bool:
    try:
        src = Path(getattr(module, "__file__", "")).resolve()
    except Exception:
        return False
    return src.is_file() and str(src).startswith(str(BASE_PKG))


def _load_pinned_nestynet() -> None:
    if not (BASE_PKG / "__init__.py").is_file():
        raise RuntimeError(
            f"NestyNet_SR expects a NestyNet checkout at {BASE_ROOT}; "
            f"no package found at {BASE_PKG}.  Set NESTYNET_BASE to override."
        )

    root_s = str(BASE_ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    self_s = str(ROOT)
    if self_s not in sys.path:
        sys.path.insert(0, self_s)

    existing = sys.modules.get("nestynet")
    if existing is not None and _is_pinned_nestynet(existing):
        return

    for name in list(sys.modules):
        if name == "nestynet" or name.startswith("nestynet."):
            sys.modules.pop(name, None)

    spec = importlib.util.spec_from_file_location(
        "nestynet",
        BASE_PKG / "__init__.py",
        submodule_search_locations=[str(BASE_PKG)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build an import spec for {BASE_PKG}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["nestynet"] = module
    spec.loader.exec_module(module)

    if not _is_pinned_nestynet(module):
        raise RuntimeError(
            f"nestynet resolved to {getattr(module, '__file__', '<unknown>')}, "
            f"expected a module under {BASE_PKG}"
        )


_load_pinned_nestynet()


def pytest_report_header(config):
    """Make the pairing visible in every run header, not just on failure."""
    return f"nestynet base: {BASE_ROOT}"
