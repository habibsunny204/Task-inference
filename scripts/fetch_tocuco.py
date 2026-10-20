"""Download TOC-UCO (Ayllón-Gavilán et al.) into data/external/TOC-UCO/ and check it against the files this study used.

TOC-UCO has no licence file, so its tables are not redistributed with this repository. The 20 TOC-UCO benchmark
targets, the training pool and the Colab bundles read them from data/external/TOC-UCO/TOC-UCO/ (the layout of the
official zip, extracted into data/external/TOC-UCO/).

    python scripts/fetch_tocuco.py           # download (if needed), extract, verify
    python scripts/fetch_tocuco.py --check   # verify an existing copy only

Verification compares every file with data/tocuco_sha256.csv, the sha256 of the copy used in the paper. A mismatch
means the upstream archive has changed; the numbers of the paper were computed from the recorded version.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.uco.es/grupos/ayrna/datasets/TOC-UCO.zip"
DEST = ROOT / "data" / "external" / "TOC-UCO"
ZIP = ROOT / "data" / "external" / "TOC-UCO.zip"
MANIFEST = ROOT / "data" / "tocuco_sha256.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    man = pd.read_csv(MANIFEST)
    missing, changed = [], []
    for r in man.itertuples():
        p = DEST / r.path
        if not p.exists():
            missing.append(r.path)
        elif sha256(p) != r.sha256:
            changed.append(r.path)
    print(f"{len(man) - len(missing) - len(changed)}/{len(man)} files identical to the recorded copy")
    for name, files in (("missing", missing), ("different", changed)):
        if files:
            print(f"{name}: " + ", ".join(files[:10]) + (" ..." if len(files) > 10 else ""))
    return not missing and not changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify an existing copy")
    args = ap.parse_args()
    if not args.check:
        DEST.mkdir(parents=True, exist_ok=True)
        if not ZIP.exists():
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, ZIP)
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(DEST)
        print(f"extracted to {DEST.relative_to(ROOT)}")
    ok = verify()
    if not ok:
        print("If the archive unpacked into a different folder, the expected layout is "
              "data/external/TOC-UCO/TOC-UCO/{metadata.csv, data/...}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
