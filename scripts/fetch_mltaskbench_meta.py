"""Fetch the MLTaskBench metadata needed to plan the external check (F39). Metadata only: no table is downloaded here.

    .venv/Scripts/python scripts/fetch_mltaskbench_meta.py

MLTaskBench version 2, Zenodo record 21649607 (doi:10.5281/zenodo.21649607, CC BY 4.0), Tsialis et al. 2026.

writes data/external/mltaskbench/
  meta/exp1_{train,val,test}.csv  the tabular metadata files, unchanged (from meta_datasets.zip, md5 checked)
  openml_ids.csv                  MLTaskBench download link -> OpenML file id -> OpenML dataset id (from the OpenML listing)
  zip_members.csv                 the tabular CSV files inside dataset.zip (name, sizes, offset), read from the zip's
                                  central directory with HTTP range requests (about 3.4 MB of the 4.2 GB file)
  provenance.json                 record, file checksums and the date of the fetch
"""
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from datetime import date
from hashlib import md5, sha256
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "mltaskbench"
RECORD = "https://zenodo.org/api/records/21649607"
FILES = "https://zenodo.org/records/21649607/files/{}?download=1"
OPENML_LIST = "https://www.openml.org/api/v1/json/data/list/limit/10000/offset/{}/status/all"
UA = {"User-Agent": "task-inference-research/1.0"}
BLOCK = 4 << 20


def get(url, timeout=300):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


class HttpFile:
    """A read-only, seekable view of a remote file through HTTP range requests (4 MB blocks, cached)."""

    def __init__(self, url):
        self.pos, self.fetched, self.cache = 0, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.geturl()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo, hi = b * BLOCK, min((b + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise OSError(f"no range support (HTTP {r.status})")
                self.cache[b] = r.read()
            self.fetched += len(self.cache[b])
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b, o = divmod(self.pos, BLOCK)
            chunk = self._block(b)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def main():
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    rec = json.loads(get(RECORD))
    files = {f["key"]: f for f in rec["files"]}

    # 1. tabular metadata, unchanged
    raw = get(FILES.format("meta_datasets.zip"))
    assert "md5:" + md5(raw).hexdigest() == files["meta_datasets.zip"]["checksum"], "meta_datasets.zip checksum"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for s in ("train", "val", "test"):
            (OUT / "meta" / f"exp1_{s}.csv").write_bytes(z.read(f"meta_datasets/tabular_meta_datasets/exp1_{s}.csv"))
    meta = pd.concat([pd.read_csv(OUT / "meta" / f"exp1_{s}.csv").assign(split=s) for s in ("train", "val", "test")])

    # 2. OpenML dataset ids, from the file id in each download link
    rows, off = [], 0
    while True:
        try:
            d = json.loads(get(OPENML_LIST.format(off)))
        except urllib.error.HTTPError as e:  # OpenML answers "no results" (code 372) past the end with HTTP 412
            if e.code == 412:
                break
            raise
        if "data" not in d:
            break
        rows += [{"did": int(x["did"]), "openml_name": x["name"], "version": int(x["version"]),
                  "file_id": int(x["file_id"])} for x in d["data"]["dataset"] if x.get("file_id") not in (None, "")]
        off += 10000
    ol = pd.DataFrame(rows).drop_duplicates("file_id").set_index("file_id")
    ids = meta[["dataset_name", "target_variable", "original_download_link"]].copy()
    ids["file_id"] = ids.original_download_link.str.extract(r"download/(\d+)/")[0].astype(int)
    ids = ids.join(ol, on="file_id")
    assert ids.did.notna().all(), "a download link without an OpenML dataset id"
    assert (ids.openml_name.str.lower() == ids.dataset_name.str.lower()).all(), "OpenML name differs"
    ids["did"] = ids.did.astype(int)
    ids.drop(columns="original_download_link").to_csv(OUT / "openml_ids.csv", index=False)

    # 3. tabular files inside dataset.zip (central directory only)
    f = HttpFile(FILES.format("dataset.zip"))
    assert f.size == files["dataset.zip"]["size"]
    with zipfile.ZipFile(f) as z:
        mem = pd.DataFrame([{"member": i.filename, "size": i.file_size, "compressed": i.compress_size,
                             "crc32": f"{i.CRC:08x}", "header_offset": i.header_offset}
                            for i in z.infolist() if i.filename.startswith("dataset/Tabular/") and i.filename.endswith(".csv")])
    mem.to_csv(OUT / "zip_members.csv", index=False)

    prov = {"record": "https://doi.org/10.5281/zenodo.21649607", "version": "2", "licence": "CC BY 4.0",
            "publication_date": rec["metadata"]["publication_date"], "fetched": date.today().isoformat(),
            "zenodo_checksums": {k: v["checksum"] for k, v in files.items()},
            "dataset_zip_bytes_read": f.fetched, "openml_listing_rows": len(ol),
            "sha256": {p.relative_to(OUT).as_posix(): sha256(p.read_bytes()).hexdigest()
                       for p in sorted(OUT.rglob("*.csv"))}}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(f"{len(meta)} tabular targets ({meta.sub_problem_type.value_counts().to_dict()}); {len(mem)} tabular files in "
          f"dataset.zip ({mem.compressed.sum() / 1e6:.0f} MB compressed); {f.fetched / 1e6:.1f} MB of the zip read")


if __name__ == "__main__":
    main()
