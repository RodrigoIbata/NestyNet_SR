#!/usr/bin/env python3
"""Create the four AI Feynman workspaces used by NestyNet Paper III.

The data archive is stored separately on Zenodo. This standard-library-only
utility verifies the immutable archive, rejects unsafe tar members, and
installs one benchmark workspace per noise level. It never deletes results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ARCHIVE_NAME = "AIF_data_zenodo.tgz"
ARCHIVE_BYTES = 2_060_232_099
ARCHIVE_SHA256 = "8c692d6db840df3ba2c3276cfc445dcfd33d6581ac00b926369533df4d1cd3a2"
DATA_DOI = "10.5281/zenodo.21390410"
ZENODO_RECORD_URL = "https://zenodo.org/records/21390410"
ZENODO_DOWNLOAD_URL = f"{ZENODO_RECORD_URL}/files/{ARCHIVE_NAME}?download=1"
DEFAULT_URL = os.environ.get("NESTYNET_AIF_DATA_URL", ZENODO_DOWNLOAD_URL)

NOISE_LEVELS = ("0.000", "0.001", "0.010", "0.100")
FILES_PER_LEVEL = 120
MARKER_NAME = ".nestynet-paper3-reproduction"
MANIFEST_NAME = "paper3_reproduction_manifest.json"
TEMPLATE_FILES = {
    Path("scripts/run_allstages_all.sh"): 0o755,
    Path("scripts/summarize.sh"): 0o755,
    Path("run_pb.sh"): 0o755,
}
SOURCE_FILES = {
    Path("scripts/run_allstages_escalating.sh"): 0o755,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    size = path.stat().st_size
    if size != ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive size mismatch: expected {ARCHIVE_BYTES} bytes, got {size} ({path})"
        )
    actual = sha256_file(path)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(
            "archive SHA-256 mismatch:\n"
            f"  expected {ARCHIVE_SHA256}\n"
            f"  actual   {actual}\n"
            f"  file     {path}"
        )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "NestyNet-Paper-III-setup/1"})
    print(f"Downloading {url}")
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{ARCHIVE_NAME}.", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            with urlopen(request) as response:  # noqa: S310 - caller controls URL
                shutil.copyfileobj(response, temporary, length=8 * 1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved {destination} ({destination.stat().st_size} bytes)")


def atomic_write(path: Path, content: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", dir=path.parent, delete=False
    )
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


def install_file(path: Path, content: bytes, mode: int, force: bool) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"managed path is not a regular file: {path}")
        if path.read_bytes() == content:
            os.chmod(path, mode)
            return
        if not force:
            raise RuntimeError(f"refusing to replace modified file without --force: {path}")
    atomic_write(path, content, mode)


def ensure_symlink(link: Path, target: Path, force: bool) -> None:
    wanted = os.path.relpath(target, start=link.parent)
    if os.path.lexists(link):
        if link.is_symlink() and os.readlink(link) == wanted:
            return
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"refusing to replace directory with a symlink: {link}")
        if not force:
            raise RuntimeError(f"refusing to replace path without --force: {link}")
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(wanted)


def validate_root(root: Path, source_root: Path, force: bool) -> None:
    if root == source_root or root in source_root.parents:
        raise RuntimeError("the recreation directory cannot be the source checkout or its parent")
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"recreation path is not a directory: {root}")
    if any(root.iterdir()) and not (root / MARKER_NAME).is_file() and not force:
        raise RuntimeError(
            f"refusing to adopt non-empty unmarked directory: {root}\n"
            "Choose an empty directory, or use --force to adopt it without deleting anything."
        )


def find_template_root(source_root: Path, requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser().resolve())
    candidates.extend(
        (
            source_root / "reproducibility" / "paper3_template",
            source_root.parent / "SRBench_0.000",
        )
    )
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in TEMPLATE_FILES):
            return candidate
    checked = "\n".join(f"  {candidate}" for candidate in candidates)
    raise RuntimeError(
        "could not find the Paper III runner template. Checked:\n"
        f"{checked}\nPass --template-root if it is stored elsewhere."
    )


def render_runner(template: bytes, level: str, relative: Path) -> bytes:
    if relative == Path("scripts/summarize.sh"):
        return template
    text = template.decode("utf-8")
    assignment = 'NOISE_FRAC="${NOISE_FRAC:-0.000}"'
    if text.count(assignment) != 1:
        raise RuntimeError(f"template has no single canonical noise assignment: {relative}")
    text = text.replace(assignment, f'NOISE_FRAC="${{NOISE_FRAC:-{level}}}"')
    text = text.replace("#   NOISE_FRAC=0.000", f"#   NOISE_FRAC={level}")
    if relative == Path("scripts/run_allstages_all.sh"):
        lines = text.splitlines()
        lines[1] = f"# Run the AI Feynman all-stages suite at noise fraction {level}."
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    elif relative == Path("run_pb.sh"):
        canonical = 'CANONICAL_INIT="${CANONICAL_INIT:-0}"'
        if text.count(canonical) != 1:
            raise RuntimeError("run_pb.sh template has no single canonical-init assignment")
        text = text.replace(canonical, 'CANONICAL_INIT="${CANONICAL_INIT:-1}"')
    return text.encode()


def validate_member(member: tarfile.TarInfo, seen: set[str]) -> tuple[str, str]:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"unsafe path in archive: {member.name!r}")
    if not member.isfile():
        raise RuntimeError(f"unexpected non-file archive member: {member.name!r}")
    if member.name in seen:
        raise RuntimeError(f"duplicate archive member: {member.name!r}")
    if len(pure.parts) != 2:
        raise RuntimeError(f"unexpected archive layout: {member.name!r}")
    level_dir, filename = pure.parts
    if not level_dir.startswith("noise_"):
        raise RuntimeError(f"unexpected archive directory: {level_dir!r}")
    level = level_dir.removeprefix("noise_")
    if level not in NOISE_LEVELS:
        raise RuntimeError(f"unexpected noise level: {level!r}")
    if not re.fullmatch(r"pb\d{3}.*_data\.csv", filename):
        raise RuntimeError(f"unexpected data filename: {filename!r}")
    if member.size <= 0:
        raise RuntimeError(f"empty archive member: {member.name!r}")
    seen.add(member.name)
    return level, filename


def copy_member(source, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with source, temporary:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
                temporary.write(block)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def manifest_data(
    root: Path, manifest: dict | None, verify_existing: bool
) -> dict[str, list[dict]] | None:
    if not manifest or manifest.get("archive", {}).get("sha256") != ARCHIVE_SHA256:
        return None
    by_level: dict[str, list[dict]] = {}
    for level in NOISE_LEVELS:
        entries = manifest.get("levels", {}).get(level, {}).get("files", [])
        if len(entries) != FILES_PER_LEVEL:
            return None
        names = set()
        for entry in entries:
            name = entry.get("name")
            path = root / f"SRBench_{level}" / "data" / str(name)
            if name in names or path.is_symlink() or not path.is_file():
                return None
            if path.stat().st_size != entry.get("bytes"):
                return None
            if verify_existing and sha256_file(path) != entry.get("sha256"):
                raise RuntimeError(f"installed data SHA-256 mismatch: {path}")
            names.add(name)
        by_level[level] = entries
    canonical = {entry["name"] for entry in by_level[NOISE_LEVELS[0]]}
    if any({entry["name"] for entry in by_level[level]} != canonical for level in NOISE_LEVELS):
        return None
    return by_level


def extract_all(archive_path: Path, root: Path, force: bool) -> dict[str, list[dict]]:
    existing = [
        path
        for level in NOISE_LEVELS
        for path in (root / f"SRBench_{level}" / "data").glob("pb*_data.csv")
    ]
    if existing and not force:
        raise RuntimeError(
            f"found {len(existing)} data CSVs without a complete manifest; "
            "use --force to replace managed data"
        )

    by_level: dict[str, list[dict]] = {level: [] for level in NOISE_LEVELS}
    seen: set[str] = set()
    total = len(NOISE_LEVELS) * FILES_PER_LEVEL
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for index, member in enumerate(archive, start=1):
            level, filename = validate_member(member, seen)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            destination = root / f"SRBench_{level}" / "data" / filename
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise RuntimeError(f"data destination is not a regular file: {destination}")
            digest = copy_member(source, destination)
            by_level[level].append(
                {"name": filename, "bytes": member.size, "sha256": digest}
            )
            if index == 1 or index % 20 == 0 or index == total:
                print(f"Installed {index:3d}/{total} data files")

    canonical_names = None
    for level in NOISE_LEVELS:
        entries = by_level[level]
        if len(entries) != FILES_PER_LEVEL:
            raise RuntimeError(
                f"noise_{level} has {len(entries)} files; expected {FILES_PER_LEVEL}"
            )
        names = {entry["name"] for entry in entries}
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise RuntimeError(f"noise_{level} does not contain the canonical filename set")
        entries.sort(key=lambda entry: entry["name"])
    return by_level


def combined_digest(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["name"]):
        digest.update(f"{entry['name']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def git_state(source_root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def root_readme() -> str:
    commands = "\n\n".join(
        f"```bash\ncd SRBench_{level}\nJOBS=1 ./scripts/run_allstages_all.sh\n```"
        for level in NOISE_LEVELS
    )
    return f"""# NestyNet Paper III recreation workspaces

This directory uses the AI Feynman data archived at <https://doi.org/{DATA_DOI}>.
Archive and installed-file checksums are in `{MANIFEST_NAME}`. The `NestyNet_SR`
link identifies the source checkout used by all four workspaces.

Run each fixed 2,000-training/2,000-validation benchmark from its directory:

{commands}

These are the parenthesized noisy runs in Paper III. The non-parenthesized
noisy results used 50,000/50,000 rows plus committee-of-experts mode; see
`NestyNet_SR/PAPER3_REPRODUCIBILITY.md` for that separate protocol.

For new truth-blind cheap-first campaigns, replace `run_allstages_all.sh` with
`run_allstages_escalating.sh`. The latter writes a deterministic escalation
manifest and resumes completed work.

Outputs remain inside each `SRBench_*` directory. Re-running setup never
deletes them.
"""


def benchmark_readme(level: str) -> str:
    return f"""# AI Feynman benchmark: noise fraction {level}

The 120 CSVs in `data/` are `noise_{level}/` from
<https://doi.org/{DATA_DOI}>. Run the fixed Paper III 2,000/2,000 protocol:

```bash
JOBS=1 ./scripts/run_allstages_all.sh
```

For a current cheap-first run that escalates only internal symbolic failures:

```bash
JOBS=1 ./scripts/run_allstages_escalating.sh
```

See the recreation root README and `NestyNet_SR/PAPER3_REPRODUCIBILITY.md`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recreation_dir",
        type=Path,
        help="Directory that will contain SRBench_0.000 through SRBench_0.100.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Use an existing Zenodo archive.")
    source.add_argument("--url", help=f"Download URL (default: DOI {DATA_DOI}).")
    parser.add_argument(
        "--cache",
        type=Path,
        help=f"Download cache (default: RECREATION_DIR/.cache/{ARCHIVE_NAME}).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="NestyNet_SR checkout to link (default: this checkout).",
    )
    parser.add_argument("--template-root", type=Path, help="Paper III runner template.")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Hash all installed CSVs instead of checking only sizes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace managed files/adopt a target; never delete results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.recreation_dir.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    if not (source_root / "nestynet_sr" / "run_allstages_suite.py").is_file():
        raise RuntimeError(f"not a NestyNet_SR source checkout: {source_root}")
    validate_root(root, source_root, args.force)
    root.mkdir(parents=True, exist_ok=True)
    install_file(
        root / MARKER_NAME,
        b"Managed by NestyNet_SR/scripts/setup_paper3_reproduction.py\n",
        0o644,
        args.force,
    )

    template_root = find_template_root(source_root, args.template_root)
    templates = {
        relative: (template_root / relative).read_bytes() for relative in TEMPLATE_FILES
    }
    source_files = {}
    for relative in SOURCE_FILES:
        source_path = source_root / relative
        if not source_path.is_file():
            raise RuntimeError(f"managed source file not found: {source_path}")
        source_files[relative] = source_path.read_bytes()
    equations_path = source_root / "data" / "equations.txt"
    if not equations_path.is_file():
        raise RuntimeError(f"equations manifest not found: {equations_path}")
    equations = equations_path.read_bytes()

    ensure_symlink(root / "NestyNet_SR", source_root, args.force)
    for level in NOISE_LEVELS:
        benchmark = root / f"SRBench_{level}"
        (benchmark / "data").mkdir(parents=True, exist_ok=True)
        ensure_symlink(
            benchmark / "nestynet_sr", root / "NestyNet_SR" / "nestynet_sr", args.force
        )
        for relative, mode in TEMPLATE_FILES.items():
            content = render_runner(templates[relative], level, relative)
            install_file(benchmark / relative, content, mode, args.force)
        for relative, mode in SOURCE_FILES.items():
            install_file(benchmark / relative, source_files[relative], mode, args.force)
        install_file(benchmark / "data/equations.txt", equations, 0o644, args.force)
        install_file(benchmark / "README.md", benchmark_readme(level).encode(), 0o644, args.force)

    if args.archive is not None:
        archive_path = args.archive.expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
    else:
        archive_path = (
            args.cache.expanduser().resolve()
            if args.cache is not None
            else root / ".cache" / ARCHIVE_NAME
        )
        if archive_path.exists():
            print(f"Using cached archive: {archive_path}")
        else:
            download(args.url or DEFAULT_URL, archive_path)

    print(f"Verifying {archive_path}")
    verify_archive(archive_path)
    print(f"Verified archive size and SHA-256: {ARCHIVE_SHA256}")

    manifest_path = root / MANIFEST_NAME
    try:
        old_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    except (OSError, json.JSONDecodeError):
        old_manifest = None
    by_level = None if args.force else manifest_data(root, old_manifest, args.verify_existing)
    if by_level is None:
        by_level = extract_all(archive_path, root, args.force)
    else:
        detail = "hashes" if args.verify_existing else "sizes"
        print(f"All 480 data files match the manifest ({detail}).")

    revision, dirty = git_state(source_root)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_doi": DATA_DOI,
        "archive": {
            "name": ARCHIVE_NAME,
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
        },
        "source": {"git_revision": revision, "working_tree_dirty": dirty},
        "equations_txt_sha256": hashlib.sha256(equations).hexdigest(),
        "runner_template_sha256": {
            str(relative): hashlib.sha256(payload).hexdigest()
            for relative, payload in templates.items()
        },
        "source_runner_sha256": {
            str(relative): hashlib.sha256(payload).hexdigest()
            for relative, payload in source_files.items()
        },
        "levels": {
            level: {
                "archive_directory": f"noise_{level}",
                "file_count": len(by_level[level]),
                "combined_sha256": combined_digest(by_level[level]),
                "files": sorted(by_level[level], key=lambda entry: entry["name"]),
            }
            for level in NOISE_LEVELS
        },
    }
    atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    install_file(root / "README.md", root_readme().encode(), 0o644, args.force)

    print(f"\nPaper III recreation directory is ready: {root}")
    for level in NOISE_LEVELS:
        print(f"  cd {root / f'SRBench_{level}'}")
        print("  JOBS=1 ./scripts/run_allstages_all.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
