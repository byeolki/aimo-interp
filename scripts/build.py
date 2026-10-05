#!/usr/bin/env python3
"""Build a deterministic Codabench ZIP from one directory under submissions/.

Usage: uv run scripts/build.py always-true [--main]

Small Models track is the default because that is the track this repo targets.
"""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
LIBRARY = ROOT / "src" / "aimo_interp"
DIST = ROOT / "dist"
# Fixed timestamps keep the ZIP hash stable, so docs/submissions.md can identify uploads.
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
SMALL_MARKER = "small.txt"


def collect_entries(directory: Path) -> dict[str, bytes]:
    entries = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        is_cache = "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
        if path.is_file() and not is_cache and not path.name.startswith("."):
            entries[relative.as_posix()] = path.read_bytes()
    return entries


def with_library(entries: dict[str, bytes]) -> dict[str, bytes]:
    """Vendor src/aimo_interp into the ZIP when any submission module imports it."""
    imports_library = any(
        name.endswith(".py") and b"aimo_interp" in content for name, content in entries.items()
    )
    if not imports_library:
        return entries
    library = {f"aimo_interp/{name}": content for name, content in collect_entries(LIBRARY).items()}
    return {**entries, **library}


def apply_track_marker(entries: dict[str, bytes], is_small: bool) -> dict[str, bytes]:
    without_marker = {
        name: content
        for name, content in entries.items()
        if not ("/" not in name and name.casefold() == SMALL_MARKER)
    }
    if is_small:
        return {**without_marker, SMALL_MARKER: b""}
    return without_marker


def write_zip(output: Path, entries: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(entries):
            info = ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, entries[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="directory name under submissions/")
    parser.add_argument("--main", action="store_true", help="build for the Main track")
    args = parser.parse_args()

    source = SUBMISSIONS / args.name
    if not (source / "solution.py").is_file():
        raise SystemExit(f"missing solution.py in {source}")

    is_small = not args.main
    entries = apply_track_marker(with_library(collect_entries(source)), is_small)
    suffix = "small" if is_small else "main"
    output = DIST / f"{args.name}-{suffix}.zip"
    write_zip(output, entries)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()[:12]
    print(f"{output.relative_to(ROOT)}  sha256:{digest}")
    for name in sorted(entries):
        print(f"  {name}")


if __name__ == "__main__":
    main()
