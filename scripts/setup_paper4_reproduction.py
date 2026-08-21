#!/usr/bin/env python3
"""Install the NestyNet Paper IV Zenodo artifact into a recreation directory.

This standard-library-only utility verifies both the archive checksum (once
published) and the per-file ``SHA256SUMS`` inside the artifact.  It rejects
links and unsafe tar paths and never deletes experimental results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = SOURCE_ROOT / "reproducibility" / "paper4" / "archive_record.json"
MARKER_NAME = ".nestynet-paper4-reproduction"
MANIFEST_NAME = "paper4_reproduction_manifest.json"
KEPLER_MANIFEST_NAME = "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json"
KEPLER_LOCALIZED_NAME = "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.localized.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
        return {"git_revision": revision, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "working_tree_dirty": None}


def _load_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported archive record: {path}")
    return payload


def _verify_archive(path: Path, expected_bytes: int | None, expected_sha256: str | None) -> str:
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"archive size mismatch: expected {expected_bytes}, got {path.stat().st_size}: {path}"
        )
    actual = _sha256(path)
    if expected_sha256 is not None and actual.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            "archive SHA-256 mismatch:\n"
            f"  expected {expected_sha256}\n  actual   {actual}\n  file     {path}"
        )
    return actual


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "NestyNet-Paper-IV-setup/1"})
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        print(f"Downloading {url}")
        with temporary:
            with urlopen(request) as response:  # noqa: S310 - caller/config supplies URL
                shutil.copyfileobj(response, temporary, length=8 * 1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _ensure_symlink(link: Path, target: Path, force: bool) -> None:
    wanted = os.path.relpath(target, start=link.parent)
    if os.path.lexists(link):
        if link.is_symlink() and os.readlink(link) == wanted:
            return
        if not force:
            raise RuntimeError(f"refusing to replace managed path without --force: {link}")
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"refusing to replace a directory with a symlink: {link}")
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(wanted)


def _validate_root(root: Path, force: bool) -> None:
    if root == SOURCE_ROOT or root in SOURCE_ROOT.parents:
        raise RuntimeError("the recreation directory cannot be the source checkout or its parent")
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"recreation path is not a directory: {root}")
    if any(root.iterdir()) and not (root / MARKER_NAME).is_file() and not force:
        raise RuntimeError(
            f"refusing to adopt non-empty unmarked directory: {root}\n"
            "Choose an empty directory or use --force to adopt it without deleting anything."
        )


def _validate_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe archive path: {member.name!r}")
    if member.name.rstrip("/") != path.as_posix():
        raise RuntimeError(f"non-canonical archive path: {member.name!r}")
    if not (member.isdir() or member.isfile()):
        raise RuntimeError(f"archive links/special members are not permitted: {member.name!r}")
    return path


def _extract_archive(archive: Path, destination_parent: Path) -> Path:
    temporary_root = Path(tempfile.mkdtemp(prefix=".paper4-extract-", dir=destination_parent))
    try:
        top_levels: set[str] = set()
        seen: set[str] = set()
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                path = _validate_member(member)
                normalized_name = path.as_posix()
                if normalized_name in seen:
                    raise RuntimeError(f"duplicate archive member: {member.name!r}")
                seen.add(normalized_name)
                top_levels.add(path.parts[0])
                destination = temporary_root.joinpath(*path.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read archive member: {member.name!r}")
                with source, destination.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
                os.chmod(destination, 0o644)
        if len(top_levels) != 1:
            raise RuntimeError(f"archive must have one top-level directory, found {sorted(top_levels)}")
        extracted = temporary_root / next(iter(top_levels))
        if not extracted.is_dir():
            raise RuntimeError("archive top-level member is not a directory")
        return extracted
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _verify_internal_checksums(artifact_root: Path) -> int:
    checksum_path = artifact_root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise RuntimeError(f"artifact has no SHA256SUMS: {artifact_root}")
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise RuntimeError(f"malformed checksum row: {line!r}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise RuntimeError(f"unsafe checksum path: {relative!r}")
        if relative in listed:
            raise RuntimeError(f"duplicate checksum path: {relative!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise RuntimeError(f"malformed checksum digest: {digest!r}") from exc
        path = artifact_root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"artifact file checksum mismatch: {relative}")
        listed.add(relative)

    actual: set[str] = set()
    for path in artifact_root.rglob("*"):
        if path == checksum_path or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unexpected non-regular artifact member: {path}")
        actual.add(path.relative_to(artifact_root).as_posix())
    if listed != actual:
        missing_checksums = sorted(actual - listed)
        missing_files = sorted(listed - actual)
        raise RuntimeError(
            "SHA256SUMS coverage mismatch; "
            f"unlisted_files={missing_checksums}, missing_files={missing_files}"
        )
    return len(listed)


def _install_reference(extracted: Path, root: Path) -> Path:
    reference = root / "reference"
    if reference.exists():
        raise RuntimeError(
            f"reference artifact already exists: {reference}\n"
            "Reuse this setup or choose a new recreation directory."
        )
    temporary_root = extracted.parent
    os.replace(extracted, reference)
    shutil.rmtree(temporary_root, ignore_errors=True)
    return reference


def _localize_kepler_manifest(reference: Path, data_root: Path) -> int:
    source = reference / "data" / "kepler" / KEPLER_MANIFEST_NAME
    rows = json.loads(source.read_text(encoding="utf-8"))
    localized = []
    for row in rows:
        copy = dict(row)
        relative = PurePosixPath(str(copy["csv_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Kepler CSV path in artifact: {copy['csv_path']!r}")
        if relative.parts and relative.parts[0] == "data":
            path = reference.joinpath(*relative.parts)
        else:
            path = reference / "data" / "kepler" / relative.name
        if not path.is_file():
            raise RuntimeError(f"missing installed Kepler CSV: {path}")
        copy["csv_path"] = str(path.resolve())
        localized.append(copy)
    destination = data_root / "kepler" / KEPLER_LOCALIZED_NAME
    _atomic_write(destination, (json.dumps(localized, indent=2) + "\n").encode())
    return len(localized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up a NestyNet Paper IV recreation directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recreation_dir", type=Path)
    parser.add_argument("--archive", type=Path, default=None, help="Use an already downloaded artifact")
    parser.add_argument("--url", default=None, help="Download URL override")
    parser.add_argument("--sha256", default=None, help="Expected archive SHA-256 override")
    parser.add_argument("--bytes", type=int, default=None, help="Expected archive size override")
    parser.add_argument("--record", type=Path, default=RECORD_PATH, help="Published archive record JSON")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="NestyNet-SR checkout")
    parser.add_argument("--force", action="store_true", help="Adopt a non-empty root or replace managed links/files; never deletes results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.recreation_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    record = _load_record(args.record.expanduser().resolve())
    _validate_root(root, bool(args.force))
    root.mkdir(parents=True, exist_ok=True)

    archive = args.archive.expanduser().resolve() if args.archive else root / "downloads" / str(record["filename"])
    url = args.url or record.get("download_url")
    if not archive.is_file():
        if not url:
            raise RuntimeError(
                "the Paper IV Zenodo record has not been filled in yet. Pass --archive for a "
                "local build, pass --url, or update reproducibility/paper4/archive_record.json."
            )
        _download(str(url), archive)

    expected_sha256 = args.sha256 or record.get("sha256")
    expected_bytes = args.bytes if args.bytes is not None else record.get("bytes")
    archive_sha256 = _verify_archive(archive, expected_bytes, expected_sha256)

    reference = root / "reference"
    if reference.is_dir():
        installed_manifest_path = root / MANIFEST_NAME
        if installed_manifest_path.is_symlink() or not installed_manifest_path.is_file():
            raise RuntimeError(
                f"existing reference has no trustworthy installation manifest: {installed_manifest_path}"
            )
        installed_manifest = json.loads(installed_manifest_path.read_text(encoding="utf-8"))
        installed_sha256 = installed_manifest.get("archive", {}).get("sha256")
        if installed_sha256 != archive_sha256:
            raise RuntimeError(
                "the existing reference was installed from a different archive:\n"
                f"  installed {installed_sha256}\n  requested {archive_sha256}\n"
                "Choose a new recreation directory."
            )
        internal_count = _verify_internal_checksums(reference)
    else:
        extracted = _extract_archive(archive, root)
        internal_count = _verify_internal_checksums(extracted)
        reference = _install_reference(extracted, root)

    data_root = root / "data"
    for name in ("feynman_de", "feynman_complex", "feynman_de_compositional", "maxwell_benchmark"):
        target = reference / "data" / name
        if target.is_dir():
            _ensure_symlink(data_root / name, target, bool(args.force))
    kepler_reference = reference / "data" / "kepler"
    (data_root / "kepler").mkdir(parents=True, exist_ok=True)
    for path in sorted(kepler_reference.iterdir()):
        if path.name == KEPLER_LOCALIZED_NAME:
            continue
        _ensure_symlink(data_root / "kepler" / path.name, path, bool(args.force))
    n_kepler = _localize_kepler_manifest(reference, data_root)

    _ensure_symlink(root / "NestyNet_SR", source_root, bool(args.force))
    if (reference / "results_ref").is_dir():
        _ensure_symlink(root / "results_ref", reference / "results_ref", bool(args.force))
    if (reference / "figures").is_dir():
        _ensure_symlink(root / "figures_ref", reference / "figures", bool(args.force))
    (root / "runs").mkdir(exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "doi": record.get("doi"),
            "files_verified": internal_count,
        },
        "source": {"path": str(source_root), **_git_state(source_root)},
        "reference": str(reference),
        "data_root": str(data_root),
        "kepler_bodies": n_kepler,
    }
    _atomic_write(root / MANIFEST_NAME, (json.dumps(manifest, indent=2) + "\n").encode())
    _atomic_write(root / MARKER_NAME, b"NestyNet Paper IV reproduction directory\n")

    print(f"Paper IV recreation directory: {root}")
    print(f"Verified artifact files: {internal_count}")
    print(f"Localized HORIZONS bodies: {n_kepler}")
    print("\nInspect commands:")
    print(f"  cd {root / 'NestyNet_SR'}")
    print(
        "  python scripts/run_paper4_reproduction.py --list\n"
        f"  python scripts/run_paper4_reproduction.py --workspace {root / 'runs'} "
        f"--data-root {data_root} --all"
    )
    print("Add --run only when ready to execute the full protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
