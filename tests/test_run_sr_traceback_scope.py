# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import ast
from pathlib import Path


def test_run_sr_main_does_not_shadow_global_traceback_import():
    """Portfolio exception logging must be able to call traceback.format_exc().

    A local ``import traceback`` anywhere inside ``main`` makes ``traceback`` a
    function-local name throughout ``main``.  That previously crashed pb097 when
    a later y-transform portfolio branch failed after an earlier branch had
    already solved the problem.
    """
    root = Path(__file__).resolve().parents[1]
    source = (root / "nestynet_sr" / "run_SR.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    offenders = []
    for node in ast.walk(main):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "traceback" and alias.asname is None:
                    offenders.append(node.lineno)
    assert offenders == []
