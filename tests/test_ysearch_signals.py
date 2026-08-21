# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.ysearch_signals import structural_progress


def test_structural_progress_marks_non_sep_unlocks_provisional():
    parent = {
        "sep_score": 0.20,
        "best_split_score": 0.10,
        "trig_affine_conf": 0.40,
        "split_success": 0.0,
        "logquad_ok": 0.0,
        "squarequad_ok": 0.0,
    }
    child = {
        "sep_score": 0.22,
        "best_split_score": 0.11,
        "trig_affine_conf": 0.45,
        "split_success": 0.0,
        "squarequad_ok": 1.0,
    }
    ok, reasons = structural_progress(parent, child)
    assert ok is False
    assert "provisional:squarequad_ok" in reasons


def test_structural_progress_marks_trig_threshold_crossing_provisional():
    parent = {"trig_affine_conf": 0.70}
    child = {"trig_affine_conf": 0.93}
    ok, reasons = structural_progress(
        parent,
        child,
        trig_affine_thr=0.90,
    )
    assert ok is False
    assert "provisional:trig_affine_up" in reasons


def test_structural_progress_false_when_no_family_improves():
    parent = {
        "sep_score": 0.65,
        "best_split_score": 0.70,
        "trig_affine_conf": 0.50,
        "split_success": 0.0,
    }
    child = {
        "sep_score": 0.66,
        "best_split_score": 0.71,
        "trig_affine_conf": 0.52,
        "split_success": 0.0,
    }
    ok, reasons = structural_progress(
        parent,
        child,
        sep_min_thr=0.8,
        sep_delta_thr=0.2,
        split_score_thr=0.9,
        split_margin_thr=0.15,
        trig_affine_thr=0.9,
    )
    assert ok is False
    assert reasons == []


def test_structural_progress_marks_simplicity_hint_cross_provisional():
    parent = {"simplicity_hint_ok": 0.0}
    child = {"simplicity_hint_ok": 1.0}
    ok, reasons = structural_progress(parent, child)
    assert ok is False
    assert "provisional:simplicity_hint_ok" in reasons
