"""Blinded mode: the search must never open the ground-truth answer key.

``run_SR.py --blinded`` sets ``NESTYNET_SR_BLINDED``; every ground-truth
evaluation then short-circuits *before* the canary registry is opened, so no
file containing a target expression is read during the run.  Scoring is done
afterwards by a separate process (``scripts/score_blinded_run.py``).
"""

import builtins
import json
import os

import nestynet_sr.sr_search.truth_eval as te
from scripts.score_blinded_run import (
    _noise_level,
    discovered_expression,
    main as score_blinded_main,
)


def _set_blinded(value):
    if value is None:
        os.environ.pop("NESTYNET_SR_BLINDED", None)
    else:
        os.environ["NESTYNET_SR_BLINDED"] = value


def test_blinded_active_reads_env():
    prev = os.environ.get("NESTYNET_SR_BLINDED")
    try:
        _set_blinded(None)
        assert te._blinded_active() is False
        for on in ("1", "true", "yes", "on"):
            _set_blinded(on)
            assert te._blinded_active() is True
        for off in ("0", "false", "no", "off", ""):
            _set_blinded(off)
            assert te._blinded_active() is False
    finally:
        _set_blinded(prev)


def test_evaluate_canary_short_circuits_without_opening_answer_key():
    prev = os.environ.get("NESTYNET_SR_BLINDED")
    te._canary_registry_cache = None
    opened = []
    real_open = builtins.open

    def spy_open(path, *a, **k):
        if "aif_canaries" in str(path) or "canary_truths" in str(path):
            opened.append(str(path))
        return real_open(path, *a, **k)

    try:
        builtins.open = spy_open
        _set_blinded("1")
        result = te.evaluate_canary("pb000_I_6_2a_data", "exp(-x0**2/2)/(sqrt(2*pi))")
        assert result["skipped"] is True
        assert result["blinded"] is True
        assert result["success"] is False
        assert opened == [], f"answer key opened while blinded: {opened}"
        # the registry loader is also guarded
        assert te.load_ground_truth_registry() == {}
        assert opened == []
    finally:
        builtins.open = real_open
        _set_blinded(prev)
        te._canary_registry_cache = None


def test_manifest_strips_formula_but_preserves_units():
    """The units manifest drops the formula column yet parses to identical units."""
    import tempfile

    from nestynet_sr.run_sr_input_utils import _load_units_from_equations
    from scripts.make_units_manifest import strip_eqn

    # id vars xmin xmax  eqn  y_units  x_units
    line = "003 ['x0' 'x1'] [1. 1.] [5. 5.]    sqrt((-x0 + x1)**2)     [1.0, 0.0]     [[1.0, 0.0], [1.0, 0.0]]"
    manifest_line = strip_eqn(line)
    # no formula tokens survive
    assert "sqrt" not in manifest_line and "**" not in manifest_line
    # units still resolve, identically to the full line
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f_full:
        f_full.write(line + "\n")
        full_path = f_full.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f_man:
        f_man.write(manifest_line + "\n")
        man_path = f_man.name
    try:
        assert _load_units_from_equations(full_path, "003") == _load_units_from_equations(man_path, "003")
    finally:
        os.unlink(full_path)
        os.unlink(man_path)


def test_blinded_scorer_refuses_diagnostic_ineligible_expression():
    report = {
        "stageC": {"y_expr_str": "x0 + x0/x1"},
        "final_polish": {"status": "no_safe_unit_valid_replacement"},
        "final_selection": {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "expr": "x0 + x0/x1",
        },
    }

    assert discovered_expression(report) is None


def test_blinded_scorer_prefers_explicit_eligible_final_selection():
    report = {
        "stageC": {"y_expr_str": "x0"},
        "final_polish": {"status": "no_safe_unit_valid_replacement"},
        "final_selection": {
            "source": "coe_committee",
            "applied": True,
            "eligible_for_success": True,
            "expr": "pi*x0",
            "unit_admissibility": {"checked": True, "valid": True},
        },
    }

    assert discovered_expression(report) == "pi*x0"


def test_blinded_scorer_infers_campaign_noise_conventions():
    report = {"metadata": {"dataset": "/runs/SRBench_0.000/data/pb001.csv"}}
    assert _noise_level(report) == 0.0
    assert _noise_level({}, "/runs/SRBench_0.010_blind/results") == 0.01
    assert _noise_level(
        {"metadata": {"dataset": "/runs/noise_0.001/pb001.csv"}}
    ) == 0.001


def test_blinded_scorer_counts_ineligible_report_as_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    evaluated = []
    monkeypatch.setattr(
        te,
        "evaluate_canary",
        lambda **kwargs: evaluated.append(kwargs) or {"success": True},
    )
    report = {
        "stageC": {"y_expr_str": "x0 + x0/x1"},
        "final_selection": {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "reason": "no safe unit-valid replacement",
            "expr": "x0 + x0/x1",
        },
    }
    (tmp_path / "pb061.report.json").write_text(json.dumps(report))

    assert score_blinded_main([str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "INELIGIBLE" in output
    assert "Scored 1/1 reports; exact recoveries: 0/1" in output
    assert evaluated == []
