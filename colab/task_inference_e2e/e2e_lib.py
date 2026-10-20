"""End-to-end AutoGluon confirmation of Experiment 4 (H5). Fixed 2026-10-18, before the run.

The offline Experiment 4 (scripts/policy_regret.py) scored every system's task choice with sklearn HGB fits.
This bundle refits the same 35 gold targets with **AutoGluon** forced into each task formulation, on the same rows and
the same folds, so the policy evaluation can be repeated with real AutoML losses.

Same as scripts/run_downstream_ambiguity.py:
  rows with a missing target are dropped; table read as a plain CSV shows it; KFold(5, shuffle, seed 0);
  CONTINUOUS always; COUNT if the target is integer and ≥ 0; MULTICLASS if integer with ≤ 100 levels;
  ORDINAL if integer with ≤ 30 levels; metrics from killtest_lib.compute_metrics (copied below unchanged).
Only folds 3 and 4 are fitted: they are the evaluation folds of policy_regret.py (decisions use HGB folds 0–2).

AutoGluon configuration per task (TabularPredictor, presets and time limit in config.json):
  CONTINUOUS_REGRESSION  problem_type=regression, eval_metric=root_mean_squared_error, prediction as returned
  COUNT_REGRESSION       problem_type=regression, eval_metric=mean_absolute_error, LightGBM / CatBoost / XGBoost with
                         Poisson objectives only; predictions clipped at 0
  MULTICLASS             problem_type=multiclass, eval_metric=accuracy, argmax
  ORDINAL                problem_type=multiclass, eval_metric=quadratic_kappa, median of the predicted class distribution
                         (AutoGluon has no ordinal problem type; this is the ordinal-aware configuration a user can choose)

Every (target, task, fold) writes results/runs/<key>.json at once; finished runs are skipped on a rerun.
`summarize` writes results/runs_long_ag.csv in the layout of results/downstream/hgb_fits/runs_long.csv.
"""
import io
import json
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

MAX_MULTICLASS, MAX_ORDINAL = 100, 30


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


# ---------------------------------------------------------------------------- bundle
def load_bundle(base):
    base = Path(base)
    return json.loads((base / "config.json").read_text(encoding="utf-8")), pd.read_csv(base / "targets.csv")


def load_table(base, row):
    """Table as a plain CSV shows it, rows with a missing target dropped (as in the HGB downstream run)."""
    p = Path(base) / row.data_file
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    y_raw = df[row.target_column].dropna()
    return df.loc[y_raw.index].reset_index(drop=True), y_raw.reset_index(drop=True)


def tasks_for(y):
    levels = np.unique(y)
    integer = bool(np.all(y == np.round(y)))
    out = ["CONTINUOUS_REGRESSION"]
    if integer and y.min() >= 0:
        out.append("COUNT_REGRESSION")
    if integer and len(levels) <= MAX_MULTICLASS:
        out.append("MULTICLASS")
    if integer and len(levels) <= MAX_ORDINAL:
        out.append("ORDINAL")
    return out


def ag_spec(task):
    if task == "CONTINUOUS_REGRESSION":
        return {"problem_type": "regression", "eval_metric": "root_mean_squared_error"}, None, "native"
    if task == "COUNT_REGRESSION":
        hp = {"GBM": [{"objective": "poisson"}], "CAT": [{"loss_function": "Poisson"}],
              "XGB": [{"objective": "count:poisson"}]}
        return {"problem_type": "regression", "eval_metric": "mean_absolute_error"}, hp, "native"
    if task == "MULTICLASS":
        return {"problem_type": "multiclass", "eval_metric": "accuracy"}, None, "native"
    if task == "ORDINAL":
        return {"problem_type": "multiclass", "eval_metric": "quadratic_kappa"}, None, "median"
    raise ValueError(task)


def write_env(results):
    import autogluon.tabular
    import sklearn
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "autogluon": getattr(autogluon.tabular, "__version__", "unknown"),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / "env.json").write_text(json.dumps(env, indent=1))


def run(base, results, only=None, tasks=None, folds=None, time_limit=None, retry_errors=False, log=print):
    from autogluon.tabular import TabularPredictor

    cfg, targets = load_bundle(base)
    results = Path(results)
    (results / "runs").mkdir(parents=True, exist_ok=True)
    write_env(results)
    folds = folds if folds is not None else cfg["eval_folds"]
    time_limit = time_limit or cfg["time_limit_s"]
    workdir = Path(tempfile.mkdtemp(prefix="ag_e2e_"))
    label = "__target__"
    for row in targets.itertuples():
        if only and row.instance_id not in only:
            continue
        df, y_raw = load_table(base, row)
        y = numeric_target(y_raw)
        X = df.drop(columns=[row.target_column])
        k = int(len(np.unique(y)))
        splits = make_folds(len(y), cfg["n_folds"], cfg["seed"])
        for task in tasks_for(y):
            if tasks and task not in tasks:
                continue
            kwargs, hp, decision = ag_spec(task)
            for fold in folds:
                key = f"{row.instance_id}__{task}__f{fold}"
                p = results / "runs" / f"{key}.json"
                if p.exists() and not (retry_errors and json.loads(p.read_text()).get("status") == "error"):
                    continue
                tr, te = splits[fold]
                levels = np.unique(y[tr])
                rec = {"instance_id": row.instance_id, "group": "pilot", "task": task, "arm": f"ag_{task.lower()}",
                       "fold": fold, "k": k, "presets": cfg["presets"], "time_limit_s": time_limit, **kwargs,
                       "decision": decision, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                ylab = y.astype(np.int64) if kwargs["problem_type"] == "multiclass" else y
                train = X.iloc[tr].assign(**{label: ylab[tr]})
                path = workdir / key
                t0 = time.time()
                try:
                    pred = TabularPredictor(label=label, path=str(path), verbosity=0, **kwargs)
                    fit_kw = {"presets": cfg["presets"], "time_limit": time_limit}
                    if hp:
                        fit_kw["hyperparameters"] = hp
                    pred.fit(train, **fit_kw)
                    X_te = X.iloc[te]
                    native = pred.predict(X_te).to_numpy(dtype=float)
                    if kwargs["problem_type"] == "multiclass":
                        P = pred.predict_proba(X_te)
                        dec = decisions_from_proba(P.to_numpy(), np.asarray(P.columns, dtype=float))
                        dec["native"] = native
                    else:
                        dec = {"native": np.clip(native, 0, None) if task == "COUNT_REGRESSION" else native}
                    lb = pred.leaderboard(silent=True)
                    rec.update(status="ok", fit_seconds=round(time.time() - t0, 2), n_train=len(tr), n_test=len(te),
                               model_best=str(pred.model_best), models_trained=list(map(str, lb["model"]))[:20],
                               metrics=compute_metrics(y[te], dec[decision], levels, MAX_ORDINAL),
                               metrics_all_decisions={d: compute_metrics(y[te], v, levels, MAX_ORDINAL) for d, v in dec.items()})
                except Exception as e:  # a pipeline failure is itself a result
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                               fit_seconds=round(time.time() - t0, 2))
                finally:
                    shutil.rmtree(path, ignore_errors=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, indent=1, default=str))
                os.replace(tmp, p)
                log(f"{key}: {rec['status']} ({rec.get('fit_seconds', 0)}s){'  ' + rec.get('message', '')[:120] if rec['status'] == 'error' else ''}")


def summarize(results):
    results = Path(results)
    recs = [json.loads(p.read_text()) for p in sorted((results / "runs").glob("*.json"))]
    ok = [r for r in recs if r["status"] == "ok"]
    runs = pd.DataFrame([{**{k: r[k] for k in ("instance_id", "group", "arm", "task", "fold", "k")}, **r["metrics"],
                          "fit_seconds": r["fit_seconds"], "model_best": r["model_best"]} for r in ok])
    runs.to_csv(results / "runs_long_ag.csv", index=False)
    err = pd.DataFrame([{k: r.get(k) for k in ("instance_id", "task", "fold", "message")} for r in recs if r["status"] != "ok"])
    err.to_csv(results / "errors.csv", index=False)
    print(f"runs: {len(recs)} (ok {len(ok)}, errors {len(err)})")
    if len(runs):
        print(runs.groupby("task").agg(n=("fold", "size"), mean_fit_s=("fit_seconds", "mean")).round(1).to_string())
    if len(err):
        print(err.to_string(index=False))
    return runs
