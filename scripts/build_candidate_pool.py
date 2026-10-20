"""Build the pilot candidate pool from OpenML suites, multi-type targets and TOC-UCO.

For every candidate (dataset, target) it stores provenance + target statistics and the
predictions of two frozen rule baselines. `intended_task` is left EMPTY on purpose: it must
come from documentation / annotators, never from these statistics (no circular labels).

Two views of each target are profiled:
  typed view : dtypes as delivered by OpenML (ARFF nominal => pandas 'category').
               NOTE: the ARFF type already encodes the uploader's task decision.
  csv view   : values written to text and re-parsed like a user's raw CSV (pd.read_csv
               semantics), i.e. the information a task detector really gets.

Outputs:
  benchmark/candidate_pool.csv
  data/raw/<source>_<id>/{data.parquet, metadata.json, description.txt, sha256.txt}
"""
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_numeric_dtype
from scipy import stats
from sklearn.utils.multiclass import type_of_target

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"
openml.config.set_root_cache_directory(str(ROOT / "data" / "external" / "openml_cache"))

SUITES = {99: "OpenML-CC18", 353: "OpenML-CTR23", 271: "AMLB-classification", 269: "AMLB-regression"}
MAX_ROWS = 200_000
MAX_CELLS = 20_000_000

# AutoGluon core/constants.py @ 3df8157 (2026-08-03)
AG_MULTICLASS_UPPER_LIMIT = 1000
AG_LARGE_DATA_THRESHOLD = 1000
AG_REGRESS_THRESHOLD_LARGE = 0.05
AG_REGRESS_THRESHOLD_SMALL = 0.1


def rule_autogluon(y: pd.Series) -> str:
    """Replica of autogluon.core.utils.utils.infer_problem_type (same commit)."""
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(y)
    if n == 0:
        return "error"
    k = y.nunique()
    thr = AG_REGRESS_THRESHOLD_LARGE if n > AG_LARGE_DATA_THRESHOLD else AG_REGRESS_THRESHOLD_SMALL
    if k == 2:
        return "binary"
    # pandas 3 names its default string dtype "str"; AutoGluon's list predates it.
    if y.dtype.name in ("object", "category", "string", "str", "bool") or not is_numeric_dtype(y):
        return "multiclass"
    few = (k / n <= thr) and (k <= AG_MULTICLASS_UPPER_LIMIT)
    if is_float_dtype(y):
        return "multiclass" if few and np.array_equal(y, y.astype(int)) else "regression"
    if is_integer_dtype(y):
        return "multiclass" if few else "regression"
    return "error"


def rule_sklearn(y: pd.Series) -> str:
    try:
        return type_of_target(y.dropna().to_numpy())
    except Exception as ex:  # e.g. mixed types
        return f"error:{type(ex).__name__}"


def csv_view(y: pd.Series) -> pd.Series:
    """Round-trip through CSV text so dtype inference matches a user's raw file."""
    buf = io.StringIO()
    y.astype(object).to_frame("t").to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)["t"]


def target_stats(y: pd.Series) -> dict:
    n = len(y)
    miss = int(y.isna().sum())
    yy = y.dropna()
    k = int(yy.nunique())
    d = {"target_dtype": str(y.dtype), "n_unique_target": k,
         "cardinality_ratio": round(k / max(1, len(yy)), 6),
         "missing_target_pct": round(100 * miss / max(1, n), 3)}
    num = yy if is_numeric_dtype(yy) and not is_bool_dtype(yy) else None
    if num is not None and len(num):
        v = num.to_numpy(dtype=float)
        uniq = np.unique(v)
        gaps = np.diff(uniq)
        d.update(min_target=v.min(), max_target=v.max(), mean_target=v.mean(),
                 median_target=float(np.median(v)), std_target=v.std(ddof=1) if len(v) > 1 else 0.0,
                 zero_count=int((v == 0).sum()), negative_count=int((v < 0).sum()),
                 integer_valued=bool(np.all(np.mod(v, 1) == 0)),
                 skewness=float(stats.skew(v)) if len(uniq) > 1 else 0.0,
                 gap_regularity=float(gaps.std() / gaps.mean()) if len(gaps) > 1 and gaps.mean() > 0 else np.nan,
                 contiguous_integers=bool(np.all(np.mod(v, 1) == 0) and len(uniq) == uniq.max() - uniq.min() + 1))
    p = yy.value_counts(normalize=True).to_numpy()
    d["entropy_norm"] = float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 0.0
    d["target_sample"] = "|".join(map(str, yy.drop_duplicates().head(8).tolist()))
    return d


def observed_representation(s: dict) -> str:
    k, dt = s["n_unique_target"], s["target_dtype"]
    if "integer_valued" not in s:
        return f"non-numeric ({dt}) with {k} unique values"
    kind = "integer" if s["integer_valued"] else "float"
    return f"{kind} with {k} unique values" + (" (contiguous)" if s.get("contiguous_integers") else "")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_openml_candidates() -> pd.DataFrame:
    rows = []
    for sid, sname in SUITES.items():
        suite = openml.study.get_suite(sid)
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        for _, t in tasks.iterrows():
            rows.append({"did": int(t["did"]), "target_column": t["target_feature"],
                         "openml_task_id": int(t["tid"]), "openml_task_type": t["task_type"],
                         "origin": sname})
    multi = pd.read_csv(META / "openml_multitype_targets.csv")
    for _, m in multi.iterrows():
        rows.append({"did": int(m["did"]), "target_column": m["target_feature"],
                     "openml_task_id": m["task_ids"], "openml_task_type": "classification+regression",
                     "origin": "OpenML-multitype"})
    df = pd.DataFrame(rows)
    # One row per (did, target); keep every origin label for provenance.
    return (df.groupby(["did", "target_column"])
              .agg(openml_task_id=("openml_task_id", lambda s: " ".join(sorted(set(map(str, s))))),
                   openml_task_type=("openml_task_type", lambda s: "+".join(sorted(set(s)))),
                   origin=("origin", lambda s: "+".join(sorted(set(s)))))
              .reset_index())


def profile_openml(row) -> dict | None:
    did, target = int(row.did), row.target_column
    ds = openml.datasets.get_dataset(did, download_data=False, download_qualities=True,
                                     download_features_meta_data=True)
    q = ds.qualities or {}
    n, f = q.get("NumberOfInstances", 0) or 0, q.get("NumberOfFeatures", 0) or 0
    if n > MAX_ROWS or n * f > MAX_CELLS:
        return {"skip_reason": f"too large ({int(n)} x {int(f)})"}
    X, _, _, names = ds.get_data(dataset_format="dataframe")
    if target not in X.columns:
        return {"skip_reason": f"target '{target}' not in data"}
    folder = RAW / f"openml_{did}"
    folder.mkdir(parents=True, exist_ok=True)
    X.to_parquet(folder / "data.parquet", index=False)
    (folder / "description.txt").write_text(ds.description or "", encoding="utf-8")
    meta = {"dataset_id": did, "name": ds.name, "version": ds.version, "source": "OpenML",
            "source_url": f"https://www.openml.org/d/{did}", "original_data_url": ds.original_data_url,
            "paper_url": ds.paper_url, "citation": ds.citation, "creator": ds.creator,
            "licence": ds.licence, "upload_date": ds.upload_date, "default_target": ds.default_target_attribute,
            "target_column": target, "openml_task_id": row.openml_task_id,
            "openml_task_type": row.openml_task_type, "feature_names": [c for c in X.columns if c != target]}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    digest = sha256(folder / "data.parquet")
    (folder / "sha256.txt").write_text(f"{digest}  data.parquet\n")
    y = X[target]
    typed, csv = target_stats(y), target_stats(csv_view(y))
    return {"dataset_name": ds.name, "source": "OpenML", "source_url": meta["source_url"],
            "version": ds.version, "licence": ds.licence, "n_samples": len(X), "n_features": X.shape[1] - 1,
            "local_file": str((folder / "data.parquet").relative_to(ROOT)), "sha256": digest,
            "has_description": bool((ds.description or "").strip()),
            "typed_dtype": typed["target_dtype"], **{k: v for k, v in csv.items()},
            "observed_representation": observed_representation(csv),
            "rule_autogluon_typed": rule_autogluon(y), "rule_autogluon_csv": rule_autogluon(csv_view(y)),
            "rule_sklearn_csv": rule_sklearn(csv_view(y))}


def profile_tocuco() -> list[dict]:
    base = ROOT / "data" / "external" / "TOC-UCO" / "TOC-UCO"
    meta = pd.read_csv(base / "metadata.csv") if (base / "metadata.csv").exists() else None
    out = []
    for p in sorted((base / "data").glob("*.csv")):
        X = pd.read_csv(p)
        y = X["y"]
        s = target_stats(y)
        group = "discretised-regression" if p.stem.startswith("dr") else "originally-ordinal"
        out.append({"did": p.stem, "target_column": "y", "origin": f"TOC-UCO ({group})",
                    "openml_task_type": "", "openml_task_id": "", "dataset_name": p.stem.split("_", 1)[1],
                    "source": "TOC-UCO", "source_url": "https://www.uco.es/grupos/ayrna/tocuco",
                    "n_samples": len(X), "n_features": X.shape[1] - 1,
                    "local_file": str(p.relative_to(ROOT)), "sha256": sha256(p), "has_description": False,
                    "typed_dtype": s["target_dtype"], **s, "observed_representation": observed_representation(s),
                    "rule_autogluon_typed": rule_autogluon(y), "rule_autogluon_csv": rule_autogluon(y),
                    "rule_sklearn_csv": rule_sklearn(y),
                    "documented_task": "ordinal (TOC-UCO repository)", "task_evidence_level": "E2"})
    if meta is not None:
        meta.to_csv(META / "tocuco_metadata.csv", index=False)
    return out


def main():
    cands = collect_openml_candidates()
    print("OpenML candidates:", len(cands), flush=True)
    def work(r):
        try:
            prof = profile_openml(r)
        except Exception as ex:
            prof = {"skip_reason": f"error: {type(ex).__name__}: {str(ex)[:120]}"}
        return {**r._asdict(), **prof}

    # OpenML downloads are latency-bound; a small thread pool gives a large speed-up.
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, row in enumerate(ex.map(work, cands.itertuples(index=False)), 1):
            rows.append(row)
            print(f"[{i}/{len(cands)}] {row['did']} {row['target_column']}: "
                  f"{row.get('skip_reason', 'ok')}", flush=True)
    rows += profile_tocuco()
    pool = pd.DataFrame(rows)
    pool.insert(0, "candidate_id", [f"{'openml' if str(d).isdigit() else 'tocuco'}_{d}" for d in pool["did"]])
    # Annotation fields: deliberately empty (filled by documentation review + annotators).
    for col in ["documented_task", "task_evidence_level", "evidence_source", "target_description",
                "intended_task", "plausible_task_set", "ambiguity_type", "resolvable_from_table",
                "requires_external_context", "annotator_1", "annotator_2", "adjudicated_label",
                "annotation_notes"]:
        if col not in pool.columns:
            pool[col] = ""
    out = ROOT / "benchmark" / "candidate_pool.csv"
    pool.to_csv(out, index=False)
    ok = pool["skip_reason"].isna() if "skip_reason" in pool else pd.Series(True, index=pool.index)
    print(f"wrote {out} ({ok.sum()} profiled, {(~ok).sum()} skipped)")


if __name__ == "__main__":
    sys.exit(main())
