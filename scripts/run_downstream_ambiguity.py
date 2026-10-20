"""Downstream (utility-grounded) ambiguity for every benchmark target. Frozen 2026-09-27.

For each numeric target with more than two values, fit every applicable task formulation with one
learner (sklearn HistGradientBoosting, as in the kill test's controlled tier) on 5 shared folds:

  regression_squared, regression_absolute -> CONTINUOUS_REGRESSION
  regression_poisson (integer, >= 0)      -> COUNT_REGRESSION
  multiclass (<= MAX_MULTICLASS levels)   -> MULTICLASS
  ordinal_frank_hall (<= 30 levels)       -> ORDINAL

and score each on a fixed utility set: RMSE and MAE always; exact accuracy and quadratic weighted
kappa when the target has at most 30 levels. A task is *downstream-plausible* if its best
formulation's loss (RMSE, MAE, 1 - accuracy, 1 - kappa) is within TOLERANCE (relative) of the
best formulation's on at least one utility, i.e. some
reasonable user would pick it. Binary targets and non-numeric multi-class targets admit a single
formulation and are recorded without training.

Caveat: numeric utilities presuppose that the codes are quantities. For nominal codes
(e.g. digit identities 1..10) they are not meaningful; that is exactly what the human C2 labels
are compared against.

    .venv/Scripts/python scripts/run_downstream_ambiguity.py [--only I001 G003 ...]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "colab" / "task_inference_killtest"))
import killtest_lib as kt  # noqa: E402
from instances import all_instances  # noqa: E402

OUT = ROOT / "results" / "downstream" / "hgb_fits"
SEED, N_FOLDS = 0, 5
MAX_MULTICLASS = 100
MAX_ORDINAL = 30
TOLERANCE = 0.02
TASK_OF = {"regression_squared": "CONTINUOUS_REGRESSION", "regression_absolute": "CONTINUOUS_REGRESSION",
           "regression_poisson": "COUNT_REGRESSION", "multiclass": "MULTICLASS",
           "ordinal_frank_hall": "ORDINAL"}
LOWER = {"rmse", "mae"}


def arms_for(y):
    levels = np.unique(y)
    integer = bool(np.all(y == np.round(y)))
    arms = ["regression_squared", "regression_absolute"]
    if integer and y.min() >= 0:
        arms.append("regression_poisson")
    if integer and len(levels) <= MAX_MULTICLASS:
        arms.append("multiclass")
    if integer and len(levels) <= MAX_ORDINAL:
        arms.append("ordinal_frank_hall")
    return arms


def utilities(k):
    return ["rmse", "mae"] + (["exact_acc", "qwk"] if k <= MAX_ORDINAL else [])


def run(only=None):
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for inst in all_instances():
        if only and inst.instance_id not in only:
            continue
        y_raw = inst.df[inst.target].dropna()
        k = y_raw.nunique()
        try:
            y = kt.numeric_target(y_raw)
        except ValueError:
            y = None
        if k == 2 or y is None:
            rec = {"instance_id": inst.instance_id, "group": inst.group, "k": int(k),
                   "downstream_set": "BINARY" if k == 2 else "MULTICLASS", "status": "fixed",
                   "note": "binary target" if k == 2 else "non-numeric labels: single formulation"}
            (OUT / "runs" / f"{inst.instance_id}__fixed.json").write_text(json.dumps(rec), encoding="utf-8")
            continue
        df = inst.df.loc[y_raw.index]
        X = kt.prep_hgb_X(df.drop(columns=[inst.target]))
        cfg = {"metrics": {"qwk_max_levels": MAX_ORDINAL}}
        for fold, (tr, te) in enumerate(kt.make_folds(len(y), N_FOLDS, SEED)):
            levels = np.unique(y[tr])
            for arm in arms_for(y):
                p = OUT / "runs" / f"{inst.instance_id}__{arm}__f{fold}.json"
                if p.exists():
                    continue
                rec = {"instance_id": inst.instance_id, "group": inst.group, "arm": arm, "task": TASK_OF[arm],
                       "fold": fold, "k": int(k)}
                t0 = time.time()
                try:
                    dec, info = kt.fit_predict_hgb(arm, X.iloc[tr], y[tr], X.iloc[te], SEED, MAX_ORDINAL)
                    rec.update(status="ok", fit_seconds=round(time.time() - t0, 2),
                               metrics=kt.compute_metrics(y[te], dec["native"], levels, MAX_ORDINAL), **info)
                except kt.Skip as e:
                    rec.update(status="skipped", message=str(e))
                except Exception as e:
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc())
                p.write_text(json.dumps(rec, default=str), encoding="utf-8")
                print(f"{inst.instance_id} {arm} f{fold}: {rec['status']} ({rec.get('fit_seconds', 0)}s)", flush=True)


def summarize(tolerance=TOLERANCE):
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in (OUT / "runs").glob("*.json")]
    rows = [{**{k: r[k] for k in ("instance_id", "group", "arm", "task", "fold", "k")}, **r["metrics"]}
            for r in recs if r["status"] == "ok"]
    runs = pd.DataFrame(rows)
    out = []
    for iid, g in runs.groupby("instance_id"):
        k = int(g.k.iloc[0])
        means = g.groupby(["task", "arm"])[["rmse", "mae", "exact_acc", "qwk"]].mean()
        best_by_task = {}
        winners = {}
        for u in utilities(k):
            col = means[u].dropna()
            sign = 1 if u in LOWER else -1
            # Tolerance on the loss scale: RMSE/MAE as is, 1 - accuracy and 1 - kappa for the gains.
            loss = col if u in LOWER else 1 - col
            ok = col[loss <= loss.min() * (1 + tolerance)]
            winners[u] = sorted({t for t, _ in ok.index})
            for (t, a), v in col.items():
                cur = best_by_task.setdefault(t, {})
                if u not in cur or sign * v < sign * cur[u]:
                    cur[u] = v
        plausible = sorted(set().union(*winners.values()))
        out.append({"instance_id": iid, "group": g.group.iloc[0], "k": k,
                    "tasks_tested": "|".join(sorted(g.task.unique())),
                    "downstream_set": "|".join(plausible), "set_size": len(plausible),
                    **{f"best_{u}": "|".join(w) for u, w in winners.items()}})
    res = pd.DataFrame(out)
    fixed = pd.DataFrame([r for r in recs if r["status"] == "fixed"])
    if len(fixed):
        res = pd.concat([res, fixed.drop(columns=["status"]).assign(set_size=1)], ignore_index=True)
    res = res.sort_values(["group", "instance_id"])
    res.to_csv(OUT / f"downstream_sets_tol{int(tolerance * 100):02d}.csv", index=False)
    runs.to_csv(OUT / "runs_long.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        run(a.only)
    for tol in (0.0, 0.02, 0.05):
        r = summarize(tol)
    print(r.groupby("group").set_size.value_counts().sort_index().to_string())
