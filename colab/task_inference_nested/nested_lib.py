"""Nested decision fits for the cost-aware ask rule. Fixed 2026-10-19, before the run (post hoc for the paper; declared
in docs/findings_log.md after the fourth external reading).

The frozen HGB fits (scripts/run_downstream_ambiguity.py) use KFold(5): the fits whose test folds are 0-2, from which
the cost-aware rule estimates regret, were trained on rows that include folds 3-4, on which the realized regret is
measured. This run refits the decision stage so that no row of folds 3-4 enters any decision fit:

  design "nested":   for each test fold j in {0, 1, 2}, every arm is fitted on the rows of the other two of folds 0-2
                     and tested on fold j (3 fits per arm). The code asserts that no row of folds 3-4 is used.
  design "fidelity": fold 0 of the original design (trained on folds 1-4), every arm; its native metrics must equal
                     the frozen run (frozen_fold0.csv), which checks that the environment reproduces the frozen learner.

Same as the frozen run: table as a plain CSV shows it (OpenML parquet written to CSV and read back; TOC-UCO CSV read as
is); rows with a missing target dropped; X prepared by prep_hgb_X; fold assignment KFold(5, shuffle, seed 0); the five
arms of every target; HistGradientBoosting with categorical features from dtype, early stopping off, random_state 0.
Table, decision, metric and fit code is copied unchanged from colab/task_inference_killtest/killtest_lib.py.

scikit-learn is pinned to 1.9.1, the version of the frozen run (config.json); `run` refuses any other version.

Per fit: results/runs/<id>__<arm>__<design><fold>.json (metrics of every decision the arm has: native for the
regressors; native = argmax, median and mean for the classifiers). Finished fits are skipped, so an interrupted session
resumes. `summarize` writes results/runs_long_nested.csv and results/fidelity.csv.
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
TASK_OF = {"regression_squared": "CONTINUOUS_REGRESSION", "regression_absolute": "CONTINUOUS_REGRESSION",
           "regression_poisson": "COUNT_REGRESSION", "multiclass": "MULTICLASS", "ordinal_frank_hall": "ORDINAL"}


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


class Skip(Exception):
    pass


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


def fit_predict_hgb(arm, X_tr, y_tr, X_te, seed, max_ordinal_levels):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    # Early stopping off for every arm: sklearn's default ("auto", on above 10k rows) holds out a
    # stratified split, which crashes the classifiers on classes with a single member and would
    # give arms different training budgets.
    kw = dict(categorical_features="from_dtype", early_stopping=False, random_state=seed)
    info = {}
    losses = {"regression_squared": "squared_error", "regression_absolute": "absolute_error",
              "regression_poisson": "poisson"}
    if arm in losses:
        loss = losses[arm]
        m = HistGradientBoostingRegressor(loss=loss, **kw).fit(X_tr, y_tr)
        return {"native": m.predict(X_te)}, info
    if arm == "multiclass":
        m = HistGradientBoostingClassifier(**kw).fit(X_tr, y_tr)
        dec = decisions_from_proba(m.predict_proba(X_te), m.classes_)
        dec["native"] = m.predict(X_te).astype(float)
        info["n_classes_model"] = int(len(m.classes_))
        return dec, info
    if arm == "ordinal_frank_hall":
        levels = np.unique(y_tr)
        if len(levels) > max_ordinal_levels:
            raise Skip(f"{len(levels)} levels > max_ordinal_levels={max_ordinal_levels}")
        # Frank & Hall (2001): one binary model per threshold, P(y > level_k).
        p_gt = np.column_stack([
            HistGradientBoostingClassifier(**kw).fit(X_tr, (y_tr > lv).astype(int)).predict_proba(X_te)[:, 1]
            for lv in levels[:-1]
        ])
        p_gt = np.minimum.accumulate(p_gt, axis=1)  # enforce P(y > k) non-increasing in k
        P = np.column_stack([1 - p_gt[:, 0], p_gt[:, :-1] - p_gt[:, 1:], p_gt[:, -1]])
        info["n_classes_model"] = int(len(levels))
        return decisions_from_proba(P, levels), info
    raise ValueError(f"unknown hgb arm {arm}")


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


def splits_for(design, splits, cfg):
    """(test fold, train rows, test rows) of a design. Nested: no row of the evaluation folds is used at all."""
    if design == "fidelity":
        return [(f, splits[f][0], splits[f][1]) for f in cfg["fidelity_folds"]]
    if design != "nested":
        raise ValueError(f"unknown design {design}")
    dev, held = cfg["dev_folds"], np.concatenate([splits[f][1] for f in cfg["eval_folds"]])
    out = []
    for j in dev:
        tr = np.sort(np.concatenate([splits[f][1] for f in dev if f != j]))
        te = splits[j][1]
        assert not np.intersect1d(tr, held).size and not np.intersect1d(te, held).size, "evaluation rows in a decision fit"
        assert not np.intersect1d(tr, te).size
        out.append((j, tr, te))
    return out


def run(base, results, design="nested", only=None, retry_errors=False, log=print):
    """Fit every arm of every target under `design`, cheapest targets first. Finished fits are skipped."""
    cfg, targets = load_bundle(base)
    results = Path(results)
    check_sklearn(cfg, results)
    (results / "runs").mkdir(parents=True, exist_ok=True)
    write_env(results)
    t_start = time.time()
    for row in targets.sort_values("frozen_fit_seconds").itertuples():
        if only and row.instance_id not in only:
            continue
        arms = row.arms.split("|")
        todo = [(j, arm) for j in (cfg["dev_folds"] if design == "nested" else cfg["fidelity_folds"]) for arm in arms]
        keys = {(j, arm): f"{row.instance_id}__{arm}__{design}{j}" for j, arm in todo}
        paths = {k: results / "runs" / f"{v}.json" for k, v in keys.items()}
        if all(p.exists() and not (retry_errors and json.loads(p.read_text()).get("status") == "error")
               for p in paths.values()):
            continue
        df, y_raw = load_table(base, row)
        y = numeric_target(y_raw)
        X = prep_hgb_X(df.drop(columns=[row.target_column]))
        splits = make_folds(len(y), cfg["n_folds"], cfg["seed"])
        held = np.concatenate([splits[f][1] for f in cfg["eval_folds"]])
        for j, tr, te in splits_for(design, splits, cfg):
            levels = np.unique(y[tr])
            for arm in arms:
                p = paths[(j, arm)]
                if p.exists() and not (retry_errors and json.loads(p.read_text()).get("status") == "error"):
                    continue
                rec = {"instance_id": row.instance_id, "arm": arm, "task": TASK_OF[arm], "design": design, "fold": j,
                       "k": int(len(np.unique(y))), "n_train": len(tr), "n_test": len(te),
                       "eval_rows_in_train": int(np.intersect1d(tr, held).size),
                       "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                t0 = time.time()
                try:
                    dec, info = fit_predict_hgb(arm, X.iloc[tr], y[tr], X.iloc[te], cfg["seed"], cfg["max_ordinal"])
                    rec.update(status="ok", fit_seconds=round(time.time() - t0, 2), **info,
                               metrics={d: compute_metrics(y[te], v, levels, cfg["max_ordinal"]) for d, v in dec.items()})
                except Skip as e:
                    rec.update(status="skipped", message=str(e), fit_seconds=round(time.time() - t0, 2))
                except Exception as e:  # recorded, not retried silently
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                               fit_seconds=round(time.time() - t0, 2))
                _atomic(p, lambda t: t.write_text(json.dumps(rec, indent=1, default=str)))
                log(f"{keys[(j, arm)]}: {rec['status']} ({rec['fit_seconds']}s, "
                    f"{(time.time() - t_start) / 60:.1f} min so far)"
                    + (f"  {rec.get('message', '')[:120]}" if rec["status"] != "ok" else ""))


def summarize(results, base):
    """runs_long_nested.csv (one row per target, arm, test fold and decision) and fidelity.csv (fold 0 vs frozen)."""
    results = Path(results)
    cfg, targets = load_bundle(base)
    recs = [json.loads(p.read_text()) for p in sorted((results / "runs").glob("*.json"))]
    ok = [r for r in recs if r["status"] == "ok"]
    cols = ["instance_id", "arm", "task", "design", "fold", "k", "n_train", "n_test", "eval_rows_in_train"]
    long = pd.DataFrame([{**{c: r[c] for c in cols}, "decision": d, **{m: r["metrics"][d][m] for m in METRICS},
                          "fit_seconds": r["fit_seconds"]} for r in ok for d in r["metrics"]],
                        columns=cols + ["decision", *METRICS, "fit_seconds"])
    nested = long[long.design == "nested"].drop(columns="design")
    nested.to_csv(results / "runs_long_nested.csv", index=False)
    frozen = pd.read_csv(Path(base) / "frozen_fold0.csv")
    fid = long[(long.design == "fidelity") & (long.decision == "native")].merge(
        frozen, on=["instance_id", "arm", "fold"], suffixes=("", "_frozen"))
    for m in ("mae", "rmse", "exact_acc", "qwk"):
        fid[f"diff_{m}"] = (fid[m] - fid[f"{m}_frozen"]).abs()
    fid.to_csv(results / "fidelity.csv", index=False)
    n_arms = int(targets.arms.str.split("|").str.len().sum())
    n_nested = nested[nested.decision == "native"].shape[0]
    errors = [r for r in recs if r["status"] != "ok"]
    print(f"nested fits: {n_nested} ok of {n_arms * len(cfg['dev_folds'])} expected; "
          f"evaluation rows in any nested training set: {int(nested.eval_rows_in_train.sum())}")
    if len(fid):
        d = fid[[c for c in fid.columns if c.startswith("diff_")]]
        same = int((d.fillna(0).max(axis=1) < 1e-9).sum())
        print(f"fidelity (fold 0 vs the frozen run): {same}/{len(fid)} identical, largest difference "
              f"{d.max().max():.2g}; {n_arms * len(cfg['fidelity_folds'])} expected for the full run")
    for r in errors:
        print(f"  {r['status']} {r['instance_id']} {r['arm']} {r['design']}{r['fold']}: {r.get('message', '')[:150]}")
    return nested, fid
