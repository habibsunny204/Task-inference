"""Downstream kill test: does fitting a target under the wrong task formulation cost anything
on a utility fixed in advance?

Two tiers, same folds:
  hgb  controlled comparison: one learner (sklearn HistGradientBoosting), only the task
       formulation changes (squared / absolute / Poisson regression, multiclass,
       ordinal Frank & Hall).
  ag   AutoGluon forced to each task type. The "inferred" arm is whichever of these AutoGluon
       picks on its own for the raw CSV target, so it is not trained twice.

Every (tier, target, arm, fold) run writes one JSON (metrics + metadata) and one parquet
(out-of-fold predictions) into results/. Existing runs are skipped, so an interrupted
Colab session resumes where it stopped.

Used by killtest.ipynb in Colab. After downloading results, summarise locally with
    python killtest_lib.py summarize <results_dir> --bundle <bundle_dir>
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------- bundle / data

def load_bundle(base):
    base = Path(base)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    targets = pd.read_csv(base / "targets.csv")
    return cfg, targets


def load_target(base, row):
    df = pd.read_parquet(Path(base) / row.data_file)
    y = df[row.target_column]
    X = df.drop(columns=[row.target_column])
    return X, y


def numeric_target(y):
    """Target as float on its original scale (category codes like '1','2' become numbers)."""
    if isinstance(y.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y.astype(str), errors="coerce")
    y = y.astype(float)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} target values are missing or non-numeric")
    return y.to_numpy()


def csv_view(y):
    """The target as pandas would read it back from a plain CSV (no type metadata)."""
    buf = io.StringIO(y.to_frame(name="target").to_csv(index=False))
    return pd.read_csv(buf)["target"]


def make_folds(n, n_folds, seed):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n)))


# ---------------------------------------------------------------------------- decisions / metrics

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


# ---------------------------------------------------------------------------- run bookkeeping

def run_key(tier, cid, arm, fold):
    return f"{tier}__{cid}__{arm}__f{fold}"


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _done(results, key, retry_errors):
    p = Path(results) / "runs" / f"{key}.json"
    if not p.exists():
        return False
    if retry_errors and json.loads(p.read_text(encoding="utf-8")).get("status") == "error":
        return False
    return True


def _save_run(results, key, record, y_true=None, test_idx=None, decisions=None):
    results = Path(results)
    (results / "runs").mkdir(parents=True, exist_ok=True)
    if decisions is not None:
        (results / "predictions").mkdir(parents=True, exist_ok=True)
        preds = pd.DataFrame({"row_index": test_idx, "y_true": y_true})
        for name in ("native", "median", "mean"):
            preds[f"pred_{name}"] = decisions.get(name, np.full(len(test_idx), np.nan))
        tmp = results / "predictions" / f"{key}.parquet.tmp"
        preds.to_parquet(tmp, index=False)
        os.replace(tmp, results / "predictions" / f"{key}.parquet")
    _atomic_write_text(results / "runs" / f"{key}.json", json.dumps(record, indent=1, default=str))


def _base_record(tier, row, arm, fold, cfg):
    return {
        "tier": tier, "candidate_id": row.candidate_id, "dataset_name": row.dataset_name,
        "target_column": row.target_column, "arm": arm, "fold": fold,
        "seed": cfg["seed"], "n_folds": cfg["n_folds"], "config_version": cfg["version"],
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _evaluate(record, y_te, decisions, levels, cfg):
    record["metrics"] = {
        name: compute_metrics(y_te, pred, levels, cfg["metrics"]["qwk_max_levels"])
        for name, pred in decisions.items()
    }


def write_env(results, tier):
    import sklearn
    env = {
        "tier": tier, "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if tier == "ag":
        import autogluon.tabular
        env["autogluon"] = getattr(autogluon.tabular, "__version__", "unknown")
    Path(results).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(Path(results) / f"env_{tier}.json", json.dumps(env, indent=1))
    return env


# ---------------------------------------------------------------------------- tier: hgb

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


def run_hgb(base, results, only=None, retry_errors=False, n_folds=None, log=print):
    cfg, targets = load_bundle(base)
    n_folds = n_folds or cfg["n_folds"]
    write_env(results, "hgb")
    for row in targets.itertuples():
        if only and row.candidate_id not in only:
            continue
        X, y_raw = load_target(base, row)
        X, y = prep_hgb_X(X), numeric_target(y_raw)
        for fold, (tr, te) in enumerate(make_folds(len(y), cfg["n_folds"], cfg["seed"])[:n_folds]):
            levels = np.unique(y[tr])
            for arm in cfg["hgb"]["arms"]:
                key = run_key("hgb", row.candidate_id, arm, fold)
                if _done(results, key, retry_errors):
                    continue
                rec = _base_record("hgb", row, arm, fold, cfg)
                t0 = time.time()
                try:
                    dec, info = fit_predict_hgb(arm, X.iloc[tr], y[tr], X.iloc[te],
                                                cfg["seed"], cfg["hgb"]["max_ordinal_levels"])
                    rec.update(info, status="ok", fit_seconds=round(time.time() - t0, 2),
                               n_train=len(tr), n_test=len(te), n_levels_train=int(len(levels)),
                               n_test_unseen_level=int((~np.isin(y[te], levels)).sum()))
                    _evaluate(rec, y[te], dec, levels, cfg)
                    _save_run(results, key, rec, y[te], te, dec)
                except Skip as e:
                    rec.update(status="skipped", message=str(e))
                    _save_run(results, key, rec)
                except Exception as e:
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                               fit_seconds=round(time.time() - t0, 2))
                    _save_run(results, key, rec)
                log(f"{key}: {rec['status']} ({rec.get('fit_seconds', 0)}s)")


# ---------------------------------------------------------------------------- tier: ag

def _ag_infer_fn():
    try:
        from autogluon.core.utils import infer_problem_type
    except ImportError:
        from autogluon.core.utils.utils import infer_problem_type
    return infer_problem_type


def _call_infer(infer, y):
    try:
        return infer(y, silent=True)
    except TypeError:  # older signature without `silent`
        return infer(y)


def record_ag_inference(base, results, log=print):
    """What AutoGluon picks by itself, for the typed (parquet) and the raw CSV view of each target."""
    infer = _ag_infer_fn()
    _, targets = load_bundle(base)
    out = {}
    for row in targets.itertuples():
        _, y = load_target(base, row)
        out[row.candidate_id] = {"typed": _call_infer(infer, y), "csv": _call_infer(infer, csv_view(y))}
        log(f"{row.candidate_id} {row.dataset_name}.{row.target_column}: {out[row.candidate_id]}")
    Path(results).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(Path(results) / "ag_inferred.json", json.dumps(out, indent=1))
    return out


def run_ag(base, results, only=None, retry_errors=False, n_folds=None, time_limit=None,
           workdir=None, log=print):
    from autogluon.tabular import TabularPredictor

    cfg, targets = load_bundle(base)
    n_folds = n_folds or cfg["n_folds"]
    time_limit = time_limit or cfg["ag"]["time_limit_s"]
    write_env(results, "ag")
    record_ag_inference(base, results, log=lambda *_: None)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="ag_killtest_"))
    label = "__target__"
    for row in targets.itertuples():
        if only and row.candidate_id not in only:
            continue
        X, y_raw = load_target(base, row)
        y = numeric_target(y_raw)
        y_int = np.all(y == np.round(y))
        for fold, (tr, te) in enumerate(make_folds(len(y), cfg["n_folds"], cfg["seed"])[:n_folds]):
            levels = np.unique(y[tr])
            for arm in cfg["ag"]["arms"]:
                key = run_key("ag", row.candidate_id, arm, fold)
                if _done(results, key, retry_errors):
                    continue
                rec = _base_record("ag", row, arm, fold, cfg)
                rec.update(presets=cfg["ag"]["presets"], time_limit_s=time_limit)
                ylab = y.astype(np.int64) if (arm != "regression" and y_int) else y
                train = X.iloc[tr].assign(**{label: ylab[tr]})
                path = workdir / key
                t0 = time.time()
                try:
                    predictor = TabularPredictor(label=label, problem_type=arm, path=str(path),
                                                 verbosity=0).fit(train, presets=cfg["ag"]["presets"],
                                                                  time_limit=time_limit)
                    X_te = X.iloc[te]
                    native = predictor.predict(X_te).to_numpy(dtype=float)
                    if arm == "regression":
                        dec = {"native": native}
                    else:
                        P = predictor.predict_proba(X_te)
                        dec = decisions_from_proba(P.to_numpy(), np.asarray(P.columns, dtype=float))
                        dec["native"] = native
                        rec["n_classes_model"] = int(len(predictor.class_labels))
                    best = getattr(predictor, "model_best", None) or predictor.get_model_best()
                    rec.update(status="ok", fit_seconds=round(time.time() - t0, 2),
                               n_train=len(tr), n_test=len(te), n_levels_train=int(len(levels)),
                               n_test_unseen_level=int((~np.isin(y[te], levels)).sum()),
                               model_best=best, eval_metric=predictor.eval_metric.name)
                    _evaluate(rec, y[te], dec, levels, cfg)
                    _save_run(results, key, rec, y[te], te, dec)
                except Exception as e:
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                               fit_seconds=round(time.time() - t0, 2))
                    _save_run(results, key, rec)
                finally:
                    shutil.rmtree(path, ignore_errors=True)
                log(f"{key}: {rec['status']} ({rec.get('fit_seconds', 0)}s)")


# ---------------------------------------------------------------------------- summary

def collect_runs(results):
    rows = []
    for p in sorted((Path(results) / "runs").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        metrics = rec.pop("metrics", None)
        rec.pop("traceback", None)
        if not metrics:
            rows.append({**rec, "decision": None})
            continue
        for decision, m in metrics.items():
            rows.append({**rec, "decision": decision, **m})
    return pd.DataFrame(rows)


def _fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}g}"


def _compare(sub, metric, sign, decision, inf_arm, rule):
    """Arm means for one metric; if inf_arm is given, its regret against the best arm."""
    native = sub[sub.decision == decision].pivot(index="fold", columns="arm", values=metric)
    if native.empty:
        return None
    means = native.mean()
    best = (sign * means).idxmin()
    out = {"best_arm": best, "best_value": means[best],
           "spread_rel": (means.max() - means.min()) / abs(means[best]) if means[best] else np.nan}
    if inf_arm in native.columns:
        others = [a for a in native.columns if a != inf_arm]
        alt = min(others, key=lambda a: sign * means[a]) if others else None
        regret = sign * (means[inf_arm] - means[best])
        both = native[[inf_arm, alt]].dropna() if alt else pd.DataFrame()
        worse = int((sign * (both[inf_arm] - both[alt]) > 0).sum()) if alt else 0
        need = math.ceil(rule["min_fold_fraction"] * len(both)) if len(both) else 1
        rel = regret / abs(means[best]) if means[best] else np.nan
        out.update(inferred_value=means[inf_arm], rel_regret=rel, folds_inferred_worse=worse,
                   folds_compared=len(both),
                   meaningful=bool(len(both) and rel >= rule["min_relative_regret"] and worse >= need))
    return out


def _median_swap(sub, inf_arm, dec0):
    """Other arms as predicted; the inferred arm scored with the median of its distribution."""
    return sub[((sub.arm != inf_arm) & (sub.decision == dec0))
               | ((sub.arm == inf_arm) & (sub.decision == "median"))].assign(decision=dec0)


def summarize(results, base):
    cfg, targets = load_bundle(base)
    results = Path(results)
    runs = collect_runs(results)
    if runs.empty:
        raise SystemExit(f"no runs found in {results / 'runs'}")
    runs.to_csv(results / "runs_long.csv", index=False)
    lower = set(cfg["metrics"]["lower_is_better"])
    rule = cfg["kill_rule"]
    vmetrics = rule["verdict_metrics"]
    dec0 = cfg["metrics"]["primary_decision"]
    inferred = {}
    if (results / "ag_inferred.json").exists():
        inferred = json.loads((results / "ag_inferred.json").read_text(encoding="utf-8"))

    def sign(metric):  # sign * value: lower is better
        return 1 if metric in lower else -1

    ok = runs[runs.status == "ok"]
    per_run = runs.drop_duplicates(["tier", "candidate_id", "arm", "fold"])
    status = per_run.groupby(["tier", "arm", "status"]).size().unstack(fill_value=0)

    arm_rows, target_rows = [], []
    for t in targets.itertuples():
        numeric = t.utility_family == "numeric"
        report_metrics = list(dict.fromkeys([t.primary_metric, t.secondary_metric]
                                            + (vmetrics if numeric else [])))
        for tier in ("hgb", "ag"):
            sub = ok[(ok.tier == tier) & (ok.candidate_id == t.candidate_id)]
            if sub.empty:
                continue
            for metric in report_metrics:
                agg = sub.groupby(["arm", "decision"])[metric].agg(["mean", "std", "count"]).reset_index()
                arm_rows += [{"candidate_id": t.candidate_id, "dataset": t.dataset_name,
                              "target": t.target_column, "tier": tier, "arm": a.arm,
                              "decision": a.decision, "metric": metric,
                              "is_primary": metric == t.primary_metric, "mean": a.mean,
                              "std": a.std, "n_folds": a.count} for a in agg.itertuples()]
            inf = inferred.get(t.candidate_id, {})
            inf_arm = inf.get("csv") if tier == "ag" else None
            prim = _compare(sub, t.primary_metric, sign(t.primary_metric), dec0, inf_arm, rule)
            if prim is None:
                continue
            row = {"candidate_id": t.candidate_id, "dataset": t.dataset_name, "target": t.target_column,
                   "role": t.role, "dedup_group": t.dedup_group, "hypothesis": t.reference_task_hypothesis,
                   "tier": tier, "hyp_metric": t.primary_metric, "best_arm": prim["best_arm"],
                   "best_value": prim["best_value"], "spread_rel": prim["spread_rel"]}
            if tier == "ag":
                arm_runs = per_run[(per_run.tier == "ag") & (per_run.candidate_id == t.candidate_id)]
                n_err = int(((arm_runs.arm == inf_arm) & (arm_runs.status == "error")).sum())
                row.update(ag_inferred_csv=inf_arm, ag_inferred_typed=inf.get("typed"),
                           inferred_arm_errors=n_err, hyp_rel_regret=prim.get("rel_regret"),
                           hyp_meaningful=prim.get("meaningful"))
                if t.role == "control":
                    row["control_ok"] = bool(prim["best_arm"] == inf_arm)
                if numeric:
                    for m in vmetrics:
                        c = _compare(sub, m, sign(m), dec0, inf_arm, rule) or {}
                        cm = _compare(_median_swap(sub, inf_arm, dec0), m, sign(m), dec0, inf_arm, rule) or {}
                        row.update({f"rel_regret_{m}": c.get("rel_regret"),
                                    f"worse_folds_{m}": f"{c.get('folds_inferred_worse', 0)}/{c.get('folds_compared', 0)}",
                                    f"meaningful_{m}": bool(c.get("meaningful", False)),
                                    f"rel_regret_{m}_median_decision": cm.get("rel_regret")})
                    dominance = all(row[f"meaningful_{m}"] for m in vmetrics)
                    failure = n_err >= math.ceil(rule["min_fold_fraction"] * cfg["n_folds"])
                    row["meaningful"] = bool(dominance or failure)
                    row["reason"] = "; ".join(r for r, on in (("worse on all verdict metrics", dominance),
                                                              ("pipeline failure", failure)) if on)
            target_rows.append(row)

    arms_df = pd.DataFrame(arm_rows)
    tgt_df = pd.DataFrame(target_rows)
    arms_df.to_csv(results / "summary_arms.csv", index=False)
    tgt_df.to_csv(results / "summary_targets.csv", index=False)

    vm = " and ".join(m.upper() for m in vmetrics)
    lines = [f"# Kill test summary ({cfg['version']})", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{results}`.", "",
             f"**Kill rule.** A numeric test target has a meaningful cost when AutoGluon's inferred task is "
             f"worse than the best AutoGluon arm by >= {rule['min_relative_regret']:.0%} on {vm}, on >= "
             f"{rule['min_fold_fraction']:.0%} of folds for each, or when the inferred arm crashes on >= "
             f"{rule['min_fold_fraction']:.0%} of folds. A group qualifies when >= "
             f"{rule['group_min_fraction']:.0%} of its targets do. GO-4 needs >= {rule['go_min_groups']} "
             f"groups. Predictions are `{dec0}` (what `predict()` returns).", "",
             "## Run status", "", status.to_markdown(), ""]
    ag = tgt_df[tgt_df.tier == "ag"] if not tgt_df.empty else tgt_df
    if not ag.empty and "meaningful" in ag:
        test = ag[ag.role == "test"]
        cols = (["dataset", "target", "ag_inferred_csv"]
                + [c for m in vmetrics for c in (f"rel_regret_{m}", f"worse_folds_{m}")]
                + ["inferred_arm_errors", "meaningful", "reason", "hyp_metric", "hyp_rel_regret"])
        show = test[[c for c in cols if c in test]].copy()
        for c in show.columns:
            if c.startswith(("rel_regret", "hyp_rel_regret")):
                show[c] = show[c].map(_fmt)
        frac = test.groupby("dedup_group")["meaningful"].apply(lambda s: s.fillna(False).astype(bool).mean())
        passed = sorted(frac[frac >= rule["group_min_fraction"]].index)
        verdict = "GO-4 supported" if len(passed) >= rule["go_min_groups"] else "GO-4 NOT supported"
        lines += ["## Verdict: AutoGluon inferred task vs best forced task", "", show.to_markdown(index=False), "",
                  "Positive `rel_regret` = inferred task worse. `hyp_*` = the per-hypothesis metric from "
                  "targets.csv (secondary analysis, not used in the verdict). Median-decision variants are "
                  "in `summary_targets.csv`.", "",
                  "Group fractions: " + ", ".join(f"{g} {v:.0%}" for g, v in frac.items()), "",
                  f"**{verdict}**: {len(passed)} / {len(frac)} independent test groups qualify "
                  f"({', '.join(passed) or 'none'}).", ""]
        for r in ag[ag.role == "control"].itertuples():
            state = ("OK" if r.control_ok else
                     "WARNING: the inferred (correct) task is not best; check the pipeline before trusting the verdict")
            lines += [f"Control {r.dataset}.{r.target} ({r.hyp_metric}): inferred `{r.ag_inferred_csv}`, "
                      f"best `{r.best_arm}` -> {state}.", ""]
    hg = tgt_df[tgt_df.tier == "hgb"] if not tgt_df.empty else tgt_df
    if not hg.empty:
        piv = arms_df[(arms_df.tier == "hgb") & (arms_df.decision == dec0)].pivot_table(
            index=["dataset", "target", "metric"], columns="arm", values="mean")
        lines += ["## Controlled comparison (HistGradientBoosting, native decision)", "",
                  piv.map(_fmt).to_markdown(), ""]
    (results / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return tgt_df, arms_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("results")
    s.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    h = sub.add_parser("hgb", help="run the HistGradientBoosting tier locally")
    h.add_argument("results")
    h.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    h.add_argument("--only", nargs="*")
    h.add_argument("--n-folds", type=int)
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.results, a.bundle)
        print((Path(a.results) / "summary.md").read_text(encoding="utf-8"))
    elif a.cmd == "hgb":
        run_hgb(a.bundle, a.results, only=a.only, n_folds=a.n_folds)


if __name__ == "__main__":
    main()
