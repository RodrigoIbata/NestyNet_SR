from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_paper4_reproduction.py"
BUILDER = REPO_ROOT / "scripts" / "build_paper4_artifact.py"
SETUP = REPO_ROOT / "scripts" / "setup_paper4_reproduction.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def test_protocol_dry_run_pins_complex_stlsq(tmp_path: Path) -> None:
    result = _run(
        str(RUNNER),
        "complex_stlsq",
        "--workspace",
        str(tmp_path / "runs"),
        "--data-root",
        str(tmp_path / "data"),
        "--smoke",
    )
    assert "--engine sparse" in result.stdout
    assert "--sparse_library class" in result.stdout
    assert "--only C000" in result.stdout
    manifest = json.loads((tmp_path / "runs" / "paper4_run_manifest.json").read_text())
    command = manifest["runs"][0]["command"]
    assert command[command.index("--engine") + 1] == "sparse"
    assert command[command.index("--sparse_library") + 1] == "class"
    assert "--all" not in command
    assert "working_tree_dirty" in manifest["source"]


def test_strict_builder_rejects_missing_final_summaries(tmp_path: Path) -> None:
    result = _run(
        str(BUILDER),
        "--workspace",
        str(tmp_path / "workspace"),
        "--output-dir",
        str(tmp_path / "artifact"),
        "--archive",
        str(tmp_path / "artifact.tgz"),
        check=False,
    )
    assert result.returncode != 0
    assert "refusing to build final Paper IV artifact" in result.stdout
    assert "scalar_stlsq" in result.stdout


def test_incomplete_artifact_round_trip_localizes_kepler_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    kepler_root = tmp_path / "kepler_source"
    raw_root = kepler_root / "raw"
    raw_root.mkdir(parents=True)
    csv_path = raw_root / "mp_1_ceres.csv"
    csv_path.write_text(
        "t_day,x_au,y_au,z_au,vx_au_per_d,vy_au_per_d,vz_au_per_d\n"
        "0,1,0,0,0,0.01,0\n",
        encoding="utf-8",
    )
    source_manifest = kepler_root / "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json"
    source_manifest.write_text(
        json.dumps(
            [
                {
                    "orbit_id": "mp_1_ceres",
                    "body_name": "Ceres",
                    "split": "candidate",
                    "csv_path": str(csv_path),
                    "horizons_command": "1;",
                    "center_name": "Sun (10)",
                    "start_date": "1980-01-01",
                    "stop_date": "2009-12-31",
                    "cadence_days": 1.0,
                    "n_rows": 1,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    artifact = tmp_path / "paper4_fixture"
    archive = tmp_path / "paper4_fixture.tgz"
    _run(
        str(BUILDER),
        "--workspace",
        str(workspace),
        "--kepler-data-root",
        str(kepler_root),
        "--kepler-manifest",
        str(source_manifest),
        "--paper-figures",
        str(tmp_path / "no_figures"),
        "--output-dir",
        str(artifact),
        "--archive",
        str(archive),
        "--allow-incomplete",
    )
    assert archive.is_file()
    assert (artifact / "SHA256SUMS").is_file()
    archived_manifest = json.loads(
        (artifact / "data" / "kepler" / source_manifest.name).read_text()
    )
    assert archived_manifest[0]["csv_path"].startswith("data/kepler/")
    assert ("/" + "Users/") not in json.dumps(archived_manifest)

    recreation = tmp_path / "recreation"
    # The repository record pins the published Zenodo archive; the fixture
    # archive must be verified against a record of its own (null = unchecked).
    null_record = tmp_path / "null_record.json"
    null_record.write_text(json.dumps({"schema_version": 1, "doi": None, "record_url": None,
                                       "download_url": None, "filename": archive.name,
                                       "bytes": None, "sha256": None}))
    _run(
        str(SETUP),
        str(recreation),
        "--record",
        str(null_record),
        "--archive",
        str(archive),
        "--source-root",
        str(REPO_ROOT),
    )
    localized_path = (
        recreation
        / "data"
        / "kepler"
        / "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.localized.json"
    )
    localized = json.loads(localized_path.read_text())
    installed_csv = Path(localized[0]["csv_path"])
    assert installed_csv.is_absolute()
    assert installed_csv.is_file()
    assert (recreation / "NestyNet_SR").is_symlink()
    manifest = json.loads((recreation / "paper4_reproduction_manifest.json").read_text())
    assert manifest["kepler_bodies"] == 1
    # Re-running with the same archive is idempotent and re-verifies the files.
    _run(
        str(SETUP),
        str(recreation),
        "--record",
        str(null_record),
        "--archive",
        str(archive),
        "--source-root",
        str(REPO_ROOT),
    )


def test_all_smoke_skips_steps_without_bounded_recipe(tmp_path: Path) -> None:
    result = _run(
        str(RUNNER),
        "--all",
        "--smoke",
        "--workspace",
        str(tmp_path / "runs"),
        "--data-root",
        str(tmp_path / "data"),
    )
    assert "[skip smoke] kepler_direct" in result.stdout
    manifest = json.loads((tmp_path / "runs" / "paper4_run_manifest.json").read_text())
    ids = {row["id"] for row in manifest["runs"]}
    assert "kepler_direct" not in ids
    assert {"scalar_stlsq", "scalar_hybrid", "complex_stlsq", "compositional_fss", "maxwell_exact", "maxwell_noise"} <= ids


def test_checksum_verification_requires_complete_unique_coverage(tmp_path: Path) -> None:
    setup = _load_script(SETUP, "paper4_setup_checksums")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    checked = artifact / "checked.txt"
    checked.write_text("checked\n")
    (artifact / "extra.txt").write_text("not listed\n")
    digest = hashlib.sha256(checked.read_bytes()).hexdigest()
    (artifact / "SHA256SUMS").write_text(f"{digest}  checked.txt\n")
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        setup._verify_internal_checksums(artifact)

    (artifact / "SHA256SUMS").write_text(
        f"{digest}  checked.txt\n{digest}  checked.txt\n"
    )
    with pytest.raises(RuntimeError, match="duplicate checksum"):
        setup._verify_internal_checksums(artifact)


def test_tar_extractor_rejects_normalized_duplicate_names(tmp_path: Path) -> None:
    setup = _load_script(SETUP, "paper4_setup_tar")
    archive = tmp_path / "duplicate.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in (("artifact/file.txt", b"first"), ("artifact/./file.txt", b"second")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    with pytest.raises(RuntimeError, match="non-canonical|duplicate"):
        setup._extract_archive(archive, tmp_path)


def test_strict_result_guard_requires_exact_unique_problem_ids(tmp_path: Path) -> None:
    builder = _load_script(BUILDER, "paper4_builder_results")
    protocol = json.loads((REPO_ROOT / "reproducibility" / "paper4" / "protocol.json").read_text())
    results = tmp_path / "results"
    for name in ("scalar_stlsq", "scalar_hybrid", "complex_stlsq"):
        expected = protocol["expected_results"][name]
        rows = []
        for status, count in expected["statuses"].items():
            rows.extend({"id": "DUPLICATE", "status": status} for _ in range(count))
        path = results / name / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"problems": rows}))
    comp = results / "compositional_fss" / "compositional_summary.json"
    comp.parent.mkdir(parents=True)
    comp.write_text(
        json.dumps(
            {
                "ids": ["900", "901", "902", "903"],
                "counts": {"PASS": 4},
                "cases": [
                    {"problem_id": pid, "status": "PASS"}
                    for pid in ("900", "901", "902", "903")
                ],
            }
        )
    )
    errors = builder._validate_headline_results(results, protocol)
    assert any("duplicate problem IDs" in error for error in errors)
    assert any("problem-ID mismatch" in error for error in errors)

    scalar_ids = builder._benchmark_ids(REPO_ROOT / "data" / "feynman_de_benchmark.txt")
    complex_ids = builder._benchmark_ids(REPO_ROOT / "data" / "feynman_complex_benchmark.txt")
    for name, ids in (
        ("scalar_stlsq", scalar_ids),
        ("scalar_hybrid", scalar_ids),
        ("complex_stlsq", complex_ids),
    ):
        status_by_id = protocol["expected_results"][name]["status_by_id"]
        rows = [{"id": pid, "status": status_by_id[pid]} for pid in ids]
        (results / name / "summary.json").write_text(json.dumps({"problems": rows}))
    assert builder._validate_headline_results(results, protocol) == []


def test_strict_data_guard_requires_every_case_trajectory(tmp_path: Path) -> None:
    builder = _load_script(BUILDER, "paper4_builder_data")
    source = tmp_path / "feynman_de"
    source.mkdir()
    (source / "de000_ic0.csv").write_text("x,u\n0,1\n")
    with pytest.raises(RuntimeError, match="incomplete feynman_de input inventory"):
        builder._copy_selected_data(
            "feynman_de", source, tmp_path / "out", [], strict=True
        )


def test_strict_run_manifest_requires_clean_matching_full_protocol(tmp_path: Path) -> None:
    builder = _load_script(BUILDER, "paper4_builder_run_manifest")
    protocol = json.loads((REPO_ROOT / "reproducibility" / "paper4" / "protocol.json").read_text())
    source_state = {"git_revision": "abc123", "working_tree_dirty": False}
    payload = {
        "executed": True,
        "smoke": False,
        "source": source_state,
        "protocol": protocol,
        "environment": {"python_executable": "/usr/bin/python3"},
        "data_root": "/paper4/data",
        "results_root": "/paper4/results",
        "kepler_manifest": "/paper4/data/kepler/manifest.json",
        "kepler_selection": "/paper4/data/kepler/selection.json",
        "jobs": 1,
        "runs": [
            {
                "id": row["id"],
                "returncode": 0,
                "command": [
                    str(token)
                    .replace("{python}", "/usr/bin/python3")
                    .replace("{repo}", "")
                    .replace("{data}", "/paper4/data")
                    .replace("{results}", "/paper4/results")
                    .replace("{jobs}", "1")
                    .replace("{kepler_manifest}", "/paper4/data/kepler/manifest.json")
                    .replace("{kepler_selection}", "/paper4/data/kepler/selection.json")
                    for token in row["argv"]
                ],
            }
            for row in protocol["steps"]
        ],
    }
    (tmp_path / "paper4_run_manifest.json").write_text(json.dumps(payload))
    assert builder._validate_run_manifest(tmp_path, protocol, source_state) == []
    payload["source"]["working_tree_dirty"] = True
    (tmp_path / "paper4_run_manifest.json").write_text(json.dumps(payload))
    errors = builder._validate_run_manifest(tmp_path, protocol, source_state)
    assert any("dirty or unknown" in error for error in errors)
    payload["source"]["working_tree_dirty"] = False
    payload["runs"][0]["command"] = ["python", "totally_wrong.py"]
    (tmp_path / "paper4_run_manifest.json").write_text(json.dumps(payload))
    errors = builder._validate_run_manifest(tmp_path, protocol, source_state)
    assert any("command mismatch" in error for error in errors)


def test_strict_kepler_guard_enforces_unique_daily_1980_2009_contract(tmp_path: Path) -> None:
    builder = _load_script(BUILDER, "paper4_builder_kepler")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "orbit_id": f"body-{index}",
                    "horizons_command": f"{index};",
                    "csv_path": f"body-{index}.csv",
                    "split": "candidate",
                    "start_date": "2024-01-01",
                    "stop_date": "2024-01-02",
                    "cadence_days": 999.0,
                    "n_rows": 1,
                    "center_name": "Elsewhere",
                }
                for index in range(308)
            ]
        )
    )
    with pytest.raises(RuntimeError, match="start_date"):
        builder._stage_kepler(
            manifest,
            tmp_path,
            tmp_path / "out",
            [],
            allow_incomplete=False,
        )
