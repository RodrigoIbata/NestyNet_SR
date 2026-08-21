import importlib.util
from pathlib import Path


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "generalized_symmetries" / "gs_smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("gs_smoke_benchmark_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SMOKE = _load_smoke_module()
build_paper_summary = _SMOKE.build_paper_summary
write_paper_markdown = _SMOKE.write_paper_markdown


def _payload():
    return {
        "sr_noise": [
            {"case": "radial_SO2", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "ratio_scaling", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "difference_translation", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "lorentz_interval", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "learned_oblique_translation", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "affine_output_equivariance", "relative_noise": 0.0, "detection_rate": 1.0, "median_accepted_count": 1.0},
            {"case": "generic_negative_control", "relative_noise": 0.0, "false_positive_rate": 0.0, "median_accepted_count": 0.0},
        ],
        "stagea_oblique": {
            "baseline_proposals": [{"expression": "x0-x1"}],
            "gs_proposals": [{"expression": "x0-0.707107*x1"}],
            "baseline_coordinate_angle_residual": 0.169,
            "gs_coordinate_angle_residual": 1.0e-12,
        },
        "de_prolongation": {
            "harmonic_oscillator": {
                "tested_generators": 8,
                "accepted_count": 1,
                "u_scaling": {"metric": 1.0e-12, "accepted": True},
            },
            "nonhomogeneous_oscillator": {
                "tested_generators": 8,
                "accepted_count": 0,
                "u_scaling": {"metric": 0.5, "accepted": False},
            },
            "radial_first_order": {
                "tested_generators": 8,
                "accepted_count": 1,
                "x_scaling": {"metric": 1.0e-12, "accepted": True},
            },
        },
    }


def test_paper_summary_separates_coordinate_discovery_from_library_priors():
    summary = build_paper_summary(_payload())
    rows = summary["benchmark_rows"]
    by_case = {row["case"]: row for row in rows}

    assert summary["schema"] == "nestynet_sr_gs_geometry_paper_summary_v1"
    assert by_case["learned_oblique_translation"]["claim_tier"] == "coordinate_discovery"
    assert by_case["radial_first_order"]["claim_tier"] == "library_prior_requires_matched_control"
    assert by_case["generic_negative_control"]["claim_tier"] == "negative_control"
    assert any("neutral hard-tail vocabulary" in item for item in summary["matched_library_controls"])
    assert any("vector/PDE" in item for item in summary["limitations"])


def test_paper_markdown_contains_claim_tiers_and_controls(tmp_path: Path):
    summary = build_paper_summary(_payload())
    out = tmp_path / "gs_summary.md"

    write_paper_markdown(summary, out)

    text = out.read_text(encoding="utf-8")
    assert "`coordinate_discovery`" in text
    assert "`library_prior_requires_matched_control`" in text
    assert "Matched Controls" in text
