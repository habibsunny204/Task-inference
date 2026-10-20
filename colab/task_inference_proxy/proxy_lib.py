"""B2 proxy refit: multiclass HGB fits with stored class probabilities. Fixed 2026-10-19, before the run (post hoc for
the paper; declared in docs/findings_log.md).

The cost-aware ask rule estimates regret from cheap HGB fits on folds 0-2. Its ORDINAL formulation there is the
Frank-Hall decomposition, while AutoGluon's ORDINAL formulation is a multiclass model decided by the median of the
predicted class distribution. This run refits the HGB **multiclass** arm so the ORDINAL row of the proxy can use
AutoGluon's decision rule (median, or argmax for the sensitivity analysis). The proxy itself is built locally.

Same as scripts/run_downstream_ambiguity.py (the frozen HGB run):
  table as a plain CSV shows it (OpenML parquet written to CSV and read back; TOC-UCO CSV read as is); rows with a
  missing target dropped; X prepared by prep_hgb_X; KFold(5, shuffle, seed 0); HistGradientBoostingClassifier with
  categorical features from dtype, early stopping off, random_state 0. Table, decision and metric code is copied
  unchanged from colab/task_inference_killtest/killtest_lib.py.

scikit-learn is pinned to 1.9.1, the version of the frozen run (config.json): a first Colab run with 1.6.1 gave
different multiclass fits on 21 of the 35 targets (docs/findings_log.md). `run` refuses any other version.

Per (target, fold): results/runs/<id>__f<fold>.json (metrics of the argmax, median and mean decisions) and
results/proba/<id>__f<fold>.csv.gz (test rows: position, true value, class probabilities). Finished runs are
skipped, so an interrupted session resumes. `summarize` writes results/runs_long_proxy.csv and results/fidelity.csv
(argmax metrics against the frozen multiclass run, frozen_multiclass.csv).
"""
import io
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["mae", "rmse", "exact_acc", "macro_f1", "qwk", "spearman"]


# ---------------------------------------------------------------------------- copied unchanged from killtest_lib.py
def numeric_target(y):
    """Target as float on its original scale (category codes like '1','2' become numbers)."""
    if isinstance(y.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y.astype(str), errors="coerce")
    y = y.astype(float)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} target values are missing or non-numeric")
    return y.to_numpy()


def make_folds(n, n_folds, seed):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n)))


def decisions_from_proba(P, classes):
    """Point predictions from a distribution over numeric levels: mode, median, mean."""
    classes = np.asarray(classes, dtype=float)
    order = np.argsort(classes)
    classes, P = classes[order], np.clip(np.asarray(P, dtype=float)[:, order], 0, None)
    P = P / P.sum(axis=1, keepdims=True)
    return {
        "native": classes[P.argmax(axis=1)],
        "median": classes[(P.cumsum(axis=1) >= 0.5).argmax(axis=1)],
        "mean": P @ classes,
    }


def snap(pred, levels):
    """Map each prediction to the nearest observed training level."""
    levels = np.asarray(levels, dtype=float)
    if len(levels) == 1:
        return np.full(len(pred), levels[0])
    idx = np.clip(np.searchsorted(levels, pred), 1, len(levels) - 1)
    left, right = levels[idx - 1], levels[idx]
    return np.where(np.abs(pred - left) <= np.abs(right - pred), left, right)


def compute_metrics(y_true, pred, levels, qwk_max_levels):
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score, f1_score

    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    s = snap(pred, levels)
    # F1 needs discrete labels; encode values as indices so non-integer levels work too.
    codes = {v: i for i, v in enumerate(np.union1d(y_true, s))}
    t_idx, s_idx = np.vectorize(codes.get)(y_true), np.vectorize(codes.get)(s)
    out = {
        "mae": float(np.mean(np.abs(y_true - pred))),
        "rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
        "exact_acc": float(np.mean(s == y_true)),
        "macro_f1": float(f1_score(t_idx, s_idx, average="macro", zero_division=0)),
        "qwk": float("nan"),
        "spearman": float("nan"),
    }
    labels = np.union1d(y_true, s)
    if len(np.unique(levels)) <= qwk_max_levels and len(labels) > 1:
        out["qwk"] = float(cohen_kappa_score(t_idx, s_idx, labels=np.arange(len(labels)), weights="quadratic"))
    if np.std(pred) > 0:
        out["spearman"] = float(spearmanr(y_true, pred).statistic)
    return out


def prep_hgb_X(X):
    X = X.loc[:, X.notna().any()].copy()  # all-missing columns break HistGradientBoosting binning
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_bool_dtype(col):
            X[c] = col.astype(float)
        elif not isinstance(col.dtype, pd.CategoricalDtype) and not pd.api.types.is_numeric_dtype(col):
            X[c] = col.astype("category")
        if isinstance(X[c].dtype, pd.CategoricalDtype) and len(X[c].cat.categories) > 250:
            # HistGradientBoosting accepts at most 255 categories; fall back to integer codes.
            X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
    return X


def fit_multiclass(X_tr, y_tr, X_te, seed):
    """The multiclass arm of killtest_lib.fit_predict_hgb, also returning the class probabilities."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    kw = dict(categorical_features="from_dtype", early_stopping=False, random_state=seed)
    m = HistGradientBoostingClassifier(**kw).fit(X_tr, y_tr)
    P = m.predict_proba(X_te)
    dec = decisions_from_proba(P, m.classes_)
    dec["native"] = m.predict(X_te).astype(float)
    return dec, P, np.asarray(m.classes_, dtype=float)


# ---------------------------------------------------------------------------- bundle
def load_bundle(base):
    base = Path(base)
    return json.loads((base / "config.json").read_text(encoding="utf-8")), pd.read_csv(base / "targets.csv")


def load_table(base, row):
    """As scripts/build_annotation_packets.load and run_downstream_ambiguity: plain-CSV view, missing targets dropped."""
    p = Path(base) / row.data_file
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        df = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    else:
        df = pd.read_csv(p)
    y_raw = df[row.target_column].dropna()
    return df.loc[y_raw.index], y_raw


def versions():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def check_sklearn(cfg, results=None):
    """The frozen HGB fits used scikit-learn 1.9.1; with 1.6.1, 21 of the 35 targets give different multiclass fits."""
    import sklearn
    want = cfg["sklearn"]
    if sklearn.__version__ != want:
        raise RuntimeError(f"scikit-learn {sklearn.__version__} is loaded, but this run needs {want}: run the install "
                           "cell, then Runtime > Restart session, and run the cells again from the top")
    env = Path(results) / "env.json" if results else None
    if env and env.exists() and json.loads(env.read_text()).get("sklearn") != want:
        raise RuntimeError(f"{results} holds fits from another scikit-learn version: use a new results folder")


def write_env(results):
    env = {**versions(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / "env.json").write_text(json.dumps(env, indent=1))


def _atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def run(base, results, only=None, folds=None, retry_errors=False, log=print):
    """Refit the multiclass arm; folds in order (0-2, the decision folds, first). Finished runs are skipped."""
    cfg, targets = load_bundle(base)
    results = Path(results)
    check_sklearn(cfg, results)
    for d in ("runs", "proba"):
        (results / d).mkdir(parents=True, exist_ok=True)
    write_env(results)
    folds = folds if folds is not None else cfg["folds"]
    cache = {}
    t_start = time.time()
    for fold in folds:
        for row in targets.itertuples():
            if only and row.instance_id not in only:
                continue
            key = f"{row.instance_id}__f{fold}"
            p = results / "runs" / f"{key}.json"
            if p.exists() and not (retry_errors and json.loads(p.read_text()).get("status") == "error"):
                continue
            if row.instance_id not in cache:
                df, y_raw = load_table(base, row)
                y = numeric_target(y_raw)
                X = prep_hgb_X(df.drop(columns=[row.target_column]))
                cache[row.instance_id] = (X, y, make_folds(len(y), cfg["n_folds"], cfg["seed"]))
            X, y, splits = cache[row.instance_id]
            tr, te = splits[fold]
            levels = np.unique(y[tr])
            rec = {"instance_id": row.instance_id, "fold": fold, "k": int(len(np.unique(y))), "n_train": len(tr),
                   "n_test": len(te), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            t0 = time.time()
            try:
                dec, P, classes = fit_multiclass(X.iloc[tr], y[tr], X.iloc[te], cfg["seed"])
                rec.update(status="ok", fit_seconds=round(time.time() - t0, 2), n_classes_model=int(len(classes)),
                           metrics={d: compute_metrics(y[te], v, levels, cfg["max_ordinal"]) for d, v in dec.items()})
                prob = pd.DataFrame(P, columns=[f"p_{c:g}" for c in classes])
                prob.insert(0, "y_true", y[te])
                prob.insert(0, "row", te)
                _atomic(results / "proba" / f"{key}.csv.gz",
                        lambda t: prob.to_csv(t, index=False, compression="gzip", float_format="%.6g"))
            except Exception as e:  # recorded, not retried silently
                rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                           fit_seconds=round(time.time() - t0, 2))
            _atomic(p, lambda t: t.write_text(json.dumps(rec, indent=1, default=str)))
            log(f"{key}: {rec['status']} ({rec['fit_seconds']}s, {(time.time() - t_start) / 60:.1f} min so far)"
                + (f"  {rec.get('message', '')[:120]}" if rec["status"] == "error" else ""))


def summarize(results, base):
    """runs_long_proxy.csv (one row per target, fold and decision) and fidelity.csv (argmax vs the frozen run)."""
    results = Path(results)
    recs = [json.loads(p.read_text()) for p in sorted((results / "runs").glob("*.json"))]
    ok = [r for r in recs if r["status"] == "ok"]
    runs = pd.DataFrame([{"instance_id": r["instance_id"], "fold": r["fold"], "k": r["k"], "decision": d,
                          **{m: r["metrics"][d][m] for m in METRICS}, "fit_seconds": r["fit_seconds"]}
                         for r in ok for d in ("native", "median", "mean")])
    runs.to_csv(results / "runs_long_proxy.csv", index=False)
    frozen = pd.read_csv(Path(base) / "frozen_multiclass.csv")
    nat = runs[runs.decision == "native"] if len(runs) else pd.DataFrame(columns=["instance_id", "fold", *METRICS])
    fid = nat.merge(frozen, on=["instance_id", "fold"], suffixes=("", "_frozen"))
    for m in ("mae", "rmse", "exact_acc", "qwk"):
        fid[f"diff_{m}"] = (fid[m] - fid[f"{m}_frozen"]).abs()
    fid.to_csv(results / "fidelity.csv", index=False)
    errors = [r for r in recs if r["status"] != "ok"]
    print(f"runs: {len(recs)} (ok {len(ok)}, errors {len(errors)}); expected {len(frozen)} for the full run")
    if len(fid):
        worst = fid[[c for c in fid.columns if c.startswith("diff_")]].max().max()
        same = int((fid[[c for c in fid.columns if c.startswith("diff_")]].fillna(0).max(axis=1) < 1e-9).sum())
        print(f"fidelity (argmax vs the frozen multiclass run): {same}/{len(fid)} identical, largest difference {worst:.2g}")
    for r in errors:
        print(f"  error {r['instance_id']} f{r['fold']}: {r.get('message', '')[:150]}")
    return runs, fid
