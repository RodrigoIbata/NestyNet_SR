# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace

from nestynet_sr.sr_search.factorized_search.repair_policy import (
    _repair_parent_preview_retry_gate,
    _repair_parent_record_attempt,
    _repair_preview_signature,
)


def _parent(expr, mse):
    return SimpleNamespace(best_expr=expr, best_mse=float(mse))


def test_repair_parent_preview_retry_gate_blocks_repeated_signature():
    state = {}
    stats = {
        "parent_reset_rel_improve": 0.05,
        "parent_min_eval_gap": 0,
        "parent_preview_max_repeats": 1,
    }
    parent = _parent(("add", ("var", 0), ("var", 1)), 10.0)
    preview_expr = ("mul", ("var", 0), ("var", 1))
    preview_meta = {"selected_path": [1], "selected_target_mode": "identity"}
    signature = _repair_preview_signature(preview_expr, preview_meta)

    _repair_parent_record_attempt(
        "p0",
        parent,
        5,
        state,
        stats,
        count_attempt=True,
        preview_signature=signature,
    )

    allowed, reason = _repair_parent_preview_retry_gate(
        "p0",
        parent,
        preview_expr,
        preview_meta,
        state,
        stats,
    )

    assert allowed is False
    assert reason == "repeat_signature"


def test_repair_parent_preview_retry_gate_resets_after_parent_improves():
    state = {}
    stats = {
        "parent_reset_rel_improve": 0.05,
        "parent_min_eval_gap": 0,
        "parent_preview_max_repeats": 1,
    }
    parent0 = _parent(("add", ("var", 0), ("var", 1)), 10.0)
    preview_expr = ("mul", ("var", 0), ("var", 1))
    preview_meta = {"selected_path": [1], "selected_target_mode": "identity"}
    signature = _repair_preview_signature(preview_expr, preview_meta)

    _repair_parent_record_attempt(
        "p0",
        parent0,
        5,
        state,
        stats,
        count_attempt=True,
        preview_signature=signature,
    )

    improved_parent = _parent(("add", ("var", 0), ("var", 1)), 9.0)
    allowed, reason = _repair_parent_preview_retry_gate(
        "p0",
        improved_parent,
        preview_expr,
        preview_meta,
        state,
        stats,
    )

    assert allowed is True
    assert reason == "ok"
