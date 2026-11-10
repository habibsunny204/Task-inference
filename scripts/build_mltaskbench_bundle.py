"""Build the MLTaskBench bundle (F39, declared in docs/findings_log.md on 2026-11-10 before any table was read).

    .venv/Scripts/python scripts/build_mltaskbench_bundle.py

1. data/external/mltaskbench/targets.csv   the 319 tabular targets with their OpenML id, zip member, label and the
                                            overlap rule of F39 (OpenML id, family(), manual aliases): `primary` marks the
                                            210 targets of the primary analysis. Labels stay here; the bundle has none.
2. colab/task_inference_mltaskbench/
     rules_lib.py           the frozen rule code, copied from scripts/build_candidate_pool.py and scripts/run_rules.py
                            (source segments, checked to be identical)
     config.json            record, zip size and md5, scikit-learn version, seeds, store settings, smoke targets
     mltaskbench_kaggle.ipynb  one self-contained notebook: code and inputs packed inside (tar.xz, base64, checksums)
   mltaskbench_lib.py and README.md are written by hand.
"""
import ast
import base64
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "colab" / "task_inference_mltaskbench"
EXT = ROOT / "data" / "external" / "mltaskbench"
sys.path.insert(0, str(ROOT / "scripts"))
from build_silver_pool import family  # noqa: E402

LABEL = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}


# ---------------------------------------------------------------------------------------------------- 1. targets
def targets():
    meta = pd.concat([pd.read_csv(EXT / "meta" / f"exp1_{s}.csv").assign(split=s) for s in ("train", "val", "test")],
                     ignore_index=True)
    assert len(meta) == 319 and not meta.duplicated(["dataset_name", "target_variable"]).any()
    ids = pd.read_csv(EXT / "openml_ids.csv").set_index("dataset_name")
    mem = pd.read_csv(EXT / "zip_members.csv", dtype={"crc32": str})
    parts = mem.member.str.split("/")
    mem["key"] = parts.str[3] + "/" + parts.str[-1]
    assert not mem.key.duplicated().any()
    al = pd.read_csv(EXT / "overlap_aliases.csv").set_index("dataset_name")

    t = meta[["dataset_name", "target_variable", "sub_problem_type", "split", "dataset_description", "instances",
              "features"]].copy()
    t["did"] = t.dataset_name.map(ids.did).astype(int)
    key = t.dataset_name + "/" + meta.file_path.str.split("/").str[-1]
    t = t.join(mem.set_index("key")[["member", "size", "compressed", "crc32", "header_offset"]], on=key)
    assert t.member.notna().all(), "a target without its file in dataset.zip"
    folder = t.member.str.split("/").str[2].str.replace("_classification", "")
    assert (folder == t.sub_problem_type).all(), "zip folder and metadata label disagree"
    t["label"] = t.sub_problem_type.map(LABEL)

    # overlap rule (F39): (a) OpenML id, (b) family(), (c) manual alias
    man = pd.read_csv(ROOT / "data" / "store" / "manifest.csv")
    man = man[man.role.isin(["silver_train", "gold_test"])].copy()
    oid = man.candidate_id.where(man.candidate_id.astype(str).str.startswith("openml_")).str.replace("openml_", "")
    man["did"] = pd.to_numeric(oid, errors="coerce")
    did_fam = man.dropna(subset=["did"]).assign(did=lambda d: d.did.astype(int)).groupby("did").family.first()
    fam_roles = man.groupby("family").role.agg(lambda r: "|".join(sorted(set(r))))
    t["family"] = t.dataset_name.map(family)
    t["overlap_did"] = t.did.map(did_fam)
    t["overlap_family"] = t.family.where(t.family.isin(fam_roles.index))
    t["overlap_alias"] = t.dataset_name.map(al.our_family)
    t["overlap_with"] = t.overlap_did.fillna(t.overlap_family).fillna(t.overlap_alias)
    t["overlap_rule"] = (t.overlap_did.notna().map({True: "id ", False: ""}) + t.overlap_family.notna().map({True: "family ", False: ""})
                         + t.overlap_alias.notna().map({True: "alias", False: ""})).str.strip().replace("", None)
    t["overlap_roles"] = t.overlap_with.map(fam_roles)
    t["primary"] = t.overlap_with.isna()
    t = t.sort_values(["dataset_name", "target_variable"], key=lambda s: s.str.lower()).reset_index(drop=True)
    t.insert(0, "instance_id", [f"M{i + 1:03d}" for i in range(len(t))])
    n = t.primary.sum()
    assert n == 210 and (~t.primary).sum() == 109, (n, (~t.primary).sum())  # the counts declared in F39
    assert t[t.primary].label.value_counts().to_dict() == {"regression": 93, "multiclass": 64, "binary": 53}
    t.to_csv(EXT / "targets.csv", index=False, lineterminator="\r\n")
    print(f"targets.csv: {len(t)} targets, {n} primary ({t[t.primary].label.value_counts().to_dict()}); left out "
          f"{(~t.primary).sum()}: {t[~t.primary].overlap_roles.str.contains('silver').sum()} overlap a training table")
    return t


# ---------------------------------------------------------------------------------------------------- 2. bundle
RULE_SOURCES = {
    ROOT / "scripts" / "build_candidate_pool.py": ["AG_MULTICLASS_UPPER_LIMIT", "AG_LARGE_DATA_THRESHOLD",
                                                   "AG_REGRESS_THRESHOLD_LARGE", "AG_REGRESS_THRESHOLD_SMALL",
                                                   "rule_autogluon", "rule_sklearn"],
    ROOT / "scripts" / "run_rules.py": ["B, M, O, CNT, CONT", "AG_MAP", "SK_MAP", "FEW_LEVELS", "MID_LEVELS", "KEYWORDS",
                                        "as_numeric", "profile", "heuristic_set", "compatible", "tokens",
                                        "heuristic_names"],
}
RULES_HEADER = '''"""The frozen rule baselines (generated by scripts/build_mltaskbench_bundle.py; do not edit).

Source segments copied unchanged from scripts/build_candidate_pool.py (AutoGluon replica, type_of_target) and
scripts/run_rules.py (heuristic set, heuristic set + names), frozen 2026-09-26.
"""
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.utils.multiclass import type_of_target
'''


def segment_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        return ", ".join(ast.unparse(t) for t in node.targets).strip("()")
    return None


def rules_lib():
    out = [RULES_HEADER]
    for path, names in RULE_SOURCES.items():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        found = {}
        for node in tree.body:
            name = segment_name(node)
            if name in names:
                found[name] = ast.get_source_segment(src, node)
        missing = [n for n in names if n not in found]
        assert not missing, f"{path.name}: {missing}"
        out.append(f"\n\n# ---- from {path.relative_to(ROOT).as_posix()}")
        for n in names:
            out.append("\n\n" + found[n] if "def " in found[n][:4] else "\n" + found[n])
            assert found[n] in src
    text = "".join(out).rstrip() + "\n"
    (B / "rules_lib.py").write_text(text, encoding="utf-8", newline="\n")
    return text


def config(t):
    rec = json.loads((EXT / "provenance.json").read_text(encoding="utf-8"))
    store = json.loads((ROOT / "colab" / "task_inference_data" / "config.json").read_text())
    small = t.sort_values("compressed").head(3).instance_id.tolist()  # by file size only
    cfg = {"bundle": "task_inference_mltaskbench", "declared": "2026-11-10 (docs/findings_log.md, F39, before any run)",
           "record": rec["record"], "zip_url": "https://zenodo.org/records/21649607/files/dataset.zip?download=1",
           "zip_size": 4189308425, "zip_md5": rec["zenodo_checksums"]["dataset.zip"], "sklearn": "1.6.1", "seed": 0,
           "n_folds": 5, "fidelity_tol": 1e-9, "smoke_ids": small,
           "store": {k: store[k] for k in ("seed", "sample_seed", "max_target_rows", "sample_rows", "sample_cols")}}
    (B / "config.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    return cfg


def frozen_check():
    """TIA-v2 (v2:forced) and the robustness re-run of TIA-v1 (rerun:v1) on the 76 test instances."""
    cols = ["instance_id", "condition", "system", "most_likely_task"] + [f"p_{x}" for x in
                                                                         ("BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION",
                                                                          "CONTINUOUS_REGRESSION")]
    v2 = pd.read_csv(ROOT / "results" / "systems" / "tia_v2" / "predictions.csv")
    rr = pd.read_csv(ROOT / "results" / "systems" / "tia_robustness" / "predictions.csv")
    out = pd.concat([v2[v2.system == "v2:forced"][cols], rr[rr.system == "rerun:v1"][cols]])
    assert len(out) == 2 * 228
    return out.to_csv(index=False, lineterminator="\n").encode()


def bundle_targets(t):
    cols = ["instance_id", "dataset_name", "target_variable", "dataset_description", "did", "member", "size", "compressed",
            "crc32", "header_offset"]
    return t[cols].to_csv(index=False, lineterminator="\n").encode()


FILES = {  # arcname -> project file
    "mltaskbench_lib.py": B / "mltaskbench_lib.py",
    "rules_lib.py": B / "rules_lib.py",
    "config.json": B / "config.json",
    "method_lib.py": ROOT / "colab" / "task_inference_method" / "method_lib.py",
    "lexicons.py": ROOT / "colab" / "task_inference_method" / "lexicons.py",
    "robustness_lib.py": ROOT / "colab" / "task_inference_robustness" / "robustness_lib.py",
    "data_lib.py": ROOT / "colab" / "task_inference_data" / "data_lib.py",
    "inputs/features.csv": ROOT / "results" / "systems" / "tia_v1" / "features.csv",
    "inputs/manifest.csv": ROOT / "data" / "store" / "manifest.csv",
}


def payload(t):
    blobs = {k: p.read_bytes() for k, p in FILES.items()}
    blobs["inputs/frozen_check.csv"] = frozen_check()
    blobs["inputs/targets.csv"] = bundle_targets(t)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tar:
        for name, data in sorted(blobs.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue(), {k: hashlib.sha256(v).hexdigest() for k, v in sorted(blobs.items())}


# ---------------------------------------------------------------------------------------------------- notebook
def lines(text):
    out = text.strip("\n").split("\n")
    return [s + "\n" for s in out[:-1]] + [out[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text, hidden=False):
    meta = {"jupyter": {"source_hidden": True}} if hidden else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": lines(text)}


INTRO = """
# TIA and the rule baselines on MLTaskBench (Kaggle, self-contained)

An external check for the paper: the task-inference model TIA and four rule baselines predict the task of the 319
tabular targets of MLTaskBench v2 (Zenodo, CC BY 4.0), from values only (C0), with names (C1) and with documentation (C2).
Scoring happens later on our side; this notebook never reads the MLTaskBench labels.

**Everything is inside this notebook.** No dataset needs to be attached. The 319 tables (about 108 MB of the 4.2 GB
archive) are read straight from Zenodo.

**Settings (right-hand panel):** Accelerator **None** (CPU). **Internet On** (needed for Zenodo and for installing
scikit-learn 1.6.1).

**Run:** Save Version › **Save & Run All (Commit)**, then you can close the browser. About 20–30 minutes. Step 3
stops the run if this environment cannot reproduce TIA's frozen predictions. When the version has finished, open it ›
**Output** › download `mltaskbench_results.zip` and put it in the project folder.
"""

UNPACK = """
import base64, hashlib, io, os, tarfile
from pathlib import Path

if Path("/kaggle/working").is_dir():
    WORK, OUT_ZIP = Path("/tmp/mltaskbench"), Path("/kaggle/working/mltaskbench_results.zip")
elif Path("/content").is_dir():
    WORK, OUT_ZIP = Path("/content/mltaskbench"), Path("/content/mltaskbench_results.zip")
else:
    WORK = Path(os.environ.get("MLTB_WORK", Path.cwd() / "mltaskbench_run"))
    OUT_ZIP = WORK.parent / "mltaskbench_results.zip"
raw = base64.b64decode("".join(PAYLOAD.split()))
assert hashlib.sha256(raw).hexdigest() == PAYLOAD_SHA256, "the embedded bundle is damaged: download the notebook again"
for d in (WORK, WORK.parent / "mltaskbench_smoke"):
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(d, filter="data")
        else:
            tar.extractall(d)
    for name, sha in FILE_SHA256.items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == sha, f"{name} does not match its checksum"
print(f"unpacked to {WORK}: {len(FILE_SHA256)} files, checksums ok")
"""

INSTALL = """
%pip install -q "scikit-learn==1.6.1"
"""

CHECK = """
import importlib, json, sys
assert "WORK" in globals(), "run the cells from the top"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
import mltaskbench_lib as mt
CFG, TARGETS = mt.load(WORK)
mt.check_sklearn(CFG)
print(len(TARGETS), "targets;", mt.versions())
"""

SMOKE = """
smoke = mt.run(WORK.parent / "mltaskbench_smoke", only=set(CFG["smoke_ids"]), smoke=True)
assert smoke["read_ok"] == len(CFG["smoke_ids"]), "the smoke test could not read its tables: stop and send the output"
print("smoke test ok")
"""

RUN = """
report = mt.run(WORK, dest_zip=OUT_ZIP)
print(json.dumps({k: report[k] for k in ("fetch", "read_ok", "read_failed", "fidelity_passed", "seconds")}, indent=1, default=str))
print("wrote", OUT_ZIP)
"""


def notebook(raw, shas):
    sha = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode()
    body = "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))
    embed = (f"# Embedded bundle: code and inputs, ~{len(b64) // 1000} kB\nPAYLOAD_SHA256 = \"{sha}\"\n"
             f"FILE_SHA256 = {json.dumps(shas, indent=1)}\nPAYLOAD = \"\"\"\n{body}\n\"\"\"")
    cells = [
        md(INTRO),
        md("## 1. Unpack the bundle\n\nThe first cell holds the embedded code and inputs; the second unpacks them and "
           "checks every file against its checksum."),
        code(embed, hidden=True),
        code(UNPACK),
        md("## 2. Install scikit-learn 1.6.1 (needs Internet On)\n\nTIA was trained with this version; the next cell "
           "refuses to continue with any other."),
        code(INSTALL),
        code(CHECK),
        md("## 3. Smoke test (about a minute)\n\nThree small tables, a reduced training set, nothing kept. It checks that "
           "Zenodo can be reached and that the code runs here."),
        code(SMOKE),
        md("## 4. The run (about 20 minutes)\n\nReads the 319 tables, builds the inputs, runs the rules, retrains TIA and "
           "checks it against its frozen predictions (the run stops here if they differ), then predicts. The results "
           "zip is written after the rules and again at the end."),
        code(RUN),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = B / "mltaskbench_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out, sha


def main():
    t = targets()
    rules_lib()
    cfg = config(t)
    raw, shas = payload(t)
    out, sha = notebook(raw, shas)
    print(f"wrote {out.relative_to(ROOT).as_posix()}: {out.stat().st_size / 1e6:.2f} MB, payload {len(raw) / 1e3:.0f} kB xz, "
          f"sha256 {sha[:12]}; smoke targets {cfg['smoke_ids']}")
    for k, v in shas.items():
        print(f"  {k}: {v[:12]}")


if __name__ == "__main__":
    main()
