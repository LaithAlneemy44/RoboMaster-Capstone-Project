"""Verify the local dataset against the committed checksum manifest.

data/manifest.sha256 records a SHA-256 for all 7973 files of the pinned Roboflow
export. Running this after a fresh download proves you are training on exactly
the same bytes as every other run of this project, and catches truncated or
corrupt image files before they turn into confusing training errors.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --quick     # check presence + size only

Exits non-zero if anything is missing, extra, or altered.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.sha256"
CHUNK = 1 << 20  # 1 MiB


def parse_manifest(path: Path) -> dict[str, str]:
    """Parse GNU sha256sum output into {relative_path: digest}."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition(" ")
        # sha256sum marks binary-mode files with a leading '*'.
        entries[name.lstrip(" *")] = digest
    return entries


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check that every file exists and is non-empty (no hashing).",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"Manifest not found: {MANIFEST}")

    expected = parse_manifest(MANIFEST)
    missing: list[str] = []
    altered: list[str] = []

    for i, (rel, digest) in enumerate(sorted(expected.items()), start=1):
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        if args.quick:
            if path.stat().st_size == 0:
                altered.append(rel)
        elif sha256(path) != digest:
            altered.append(rel)

        if i % 500 == 0:
            print(f"  checked {i}/{len(expected)}...", flush=True)

    # Files present on disk but absent from the manifest - usually a different
    # export version, or stray files that should not be trained on.
    tracked = {(ROOT / rel) for rel in expected}
    extra = [
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "Datasets").rglob("*")
        if p.is_file() and p not in tracked
    ] if (ROOT / "Datasets").exists() else []

    print(f"\nmanifest entries : {len(expected)}")
    print(f"missing          : {len(missing)}")
    print(f"altered          : {len(altered)}")
    print(f"unexpected extra : {len(extra)}")

    for label, items in (("MISSING", missing), ("ALTERED", altered), ("EXTRA", extra)):
        for item in items[:10]:
            print(f"  [{label}] {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")

    if missing or altered or extra:
        sys.exit(
            "\nFAILED - local dataset does not match the manifest.\n"
            "Re-download with: python scripts/download_data.py --overwrite"
        )
    print("\nOK - dataset matches the manifest exactly.")


if __name__ == "__main__":
    main()
