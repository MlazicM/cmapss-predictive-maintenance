"""Fetch the NASA C-MAPSS turbofan dataset into data/raw/.

The archive is public benchmark data. If the mirror below is unreachable,
download it manually from the NASA PCoE repository and unzip the .txt files
into data/raw/ -- everything downstream only needs those files.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile

import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.config import RAW_DATA_DIR

MIRROR = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
EXPECTED = [f"{kind}_FD00{i}.txt" for i in range(1, 5) for kind in ("train", "test", "RUL")]


def _extract_txt_files(archive: zipfile.ZipFile, destination) -> list[str]:
    """Pull the .txt files out, ignoring however deeply they are nested.

    The archive has contained a nested zip in some revisions, so recurse once
    rather than hard-coding a directory layout.
    """
    written = []
    for name in archive.namelist():
        if name.endswith(".txt") and any(name.endswith(f"/{e}") or name == e for e in EXPECTED):
            target = destination / name.rsplit("/", 1)[-1]
            target.write_bytes(archive.read(name))
            written.append(target.name)
        elif name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as nested:
                written.extend(_extract_txt_files(nested, destination))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=MIRROR)
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in RAW_DATA_DIR.glob("*.txt")}
    if set(EXPECTED) <= present:
        print(f"all {len(EXPECTED)} files already in {RAW_DATA_DIR}")
        return 0

    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=300) as response:
        payload = response.read()
    print(f"  {len(payload) / 1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        written = _extract_txt_files(archive, RAW_DATA_DIR)

    missing = sorted(set(EXPECTED) - {p.name for p in RAW_DATA_DIR.glob("*.txt")})
    print(f"extracted {len(written)} files into {RAW_DATA_DIR}")
    if missing:
        print(f"WARNING: still missing {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
