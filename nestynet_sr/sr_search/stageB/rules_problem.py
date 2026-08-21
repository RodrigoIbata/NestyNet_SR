# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-B rules for explicitly flagged problem leaves."""

from __future__ import annotations

from typing import List

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    ConstNode,
    Node,
    ast_to_human_readable,
    atom_problem_label,
    get_input_exprs,
    replace_atom_in_ast,
)

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash
from .helpers import _collect_all_atoms


class RuleNonsenseUnitsZeroPrune(StageBRule):
    """Replace flagged nonsense-units NN leaves with zero and let LM refit."""

    name = "nonsense_units_zero_prune"
    exhaustive = False

    def iter_targets(self, ctx: StageBContext):
        targets: List[AtomNode] = []
        for atom in _collect_all_atoms(ctx.state.root):
            if not isinstance(atom, AtomNode):
                continue
            if str(getattr(atom, "kind", "")).lower() != "nn":
                continue
            if atom_problem_label(atom) != "nonsense_units":
                continue
            targets.append(atom)
        return targets

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if (
            not isinstance(target, AtomNode)
            or str(getattr(target, "kind", "")).lower() != "nn"
            or atom_problem_label(target) != "nonsense_units"
        ):
            return []

        cand_root = replace_atom_in_ast(ctx.state.root, target, ConstNode(0.0))
        if cand_root is None:
            return []

        problem_inputs = [ast_to_human_readable(inp) for inp in get_input_exprs(target)]
        problem_kw = dict(getattr(target, "kwargs", {}) or {})
        return [
            Candidate(
                "nonsense_units_zero_prune",
                cand_root,
                meta={
                    "structural": True,
                    "pattern_family": "nonsense_units_zero_prune",
                    "signature": (
                        atom_content_hash(target),
                        hash("nonsense_units_zero_prune"),
                    ),
                    "problem_label": "nonsense_units",
                    "problem_tag": getattr(target, "tag", None),
                    "problem_inputs": problem_inputs,
                    "problem_message": problem_kw.get("_problem_msg"),
                    "log": (
                        f"[Stage B]  Trying nonsense_units zero-prune on "
                        f"{self.describe_target(target)}: replace flagged leaf with 0"
                    ),
                },
            )
        ]
