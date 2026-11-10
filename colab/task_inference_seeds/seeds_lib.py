"""New fold assignments for the nested evaluation of the ask rules. Declared in docs/findings_log.md (F37, 2026-11-10)
before the run; post hoc for the paper (review of 2026-11-09, major comment 1).

The paper's nested evaluation uses one fold assignment, KFold(5, shuffle, seed 0). This run repeats it on other
assignments. For every fold seed S in config.json:

  design "nested": for each test fold j in {0, 1, 2}, every arm is fitted on the other two of folds 0-2 and tested on
                   fold j. The asserts of nested_lib.splits_for check that no row of folds 3-4 is used.
  design "eval":   for each test fold j in {3, 4}, every arm is fitted on the other four folds and tested on fold j,
                   as in the frozen run.

That is 875 fits per seed. Only the fold assignment changes: the tables, the X preparation, the arms, the gradient
boosting (random_state 0) and the metrics are those of nested_lib.py, a byte-for-byte copy of
colab/task_inference_nested/nested_lib.py (checked by scripts/build_seeds_bundle.py).

Seed 0 is a check, not a result: nested fold 0 and evaluation fold 3 of seed 0 must reproduce the frozen fits
(frozen_seed0.csv). No labels are in this bundle; it only produces losses. TIA's probabilities do not depend on the
folds and are not touched here.

Per fit: results/runs/seed<S>/<id>__<arm>__<design><fold>.json. Finished fits are skipped, so an interrupted session
resumes. `summarize` writes results/runs_long_seeds.csv, results/seed_status.csv and results/check_seed0.csv.
"""
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nested_lib as nl


def verify_data(base):
    """Every table in the bundle must have the sha256 recorded in targets.csv (catches a broken upload)."""
    _, targets = nl.load_bundle(base)
    bad = [r.instance_id for r in targets.itertuples()
           if hashlib.sha256((Path(base) / r.data_file).read_bytes()).hexdigest() != r.sha256]
    if bad:
        raise RuntimeError(f"tables do not match targets.csv: {bad}. Upload the data folder again")
    return len(targets)


def planned(cfg, fits=None):
    """The (design, test fold) pairs of a run: all of them, or the ones named in `fits`."""
    every = [("nested", j) for j in cfg["dev_folds"]] + [("eval", j) for j in cfg["eval_folds"]]
    if fits is None:
        return every
    wanted = [tuple(f) for f in fits]
    assert set(wanted) <= set(every), wanted
    return wanted


def fit_rows(splits, cfg, plan):
    """(design, test fold, train rows, test rows) for each planned fit."""
    nested = {j: (tr, te) for j, tr, te in nl.splits_for("nested", splits, cfg)}
    for design, j in plan:
        tr, te = nested[j] if design == "nested" else splits[j]
        assert not np.intersect1d(tr, te).size, "a test row in the training set"
        yield design, j, tr, te


def run(base, results, seed, only=None, fits=None, retry_errors=False, log=print):
    """Fit every arm of every target for one fold seed, cheapest targets first. Finished fits are skipped."""
    cfg, targets = nl.load_bundle(base)
    results = Path(results)
    nl.check_sklearn(cfg, results)
    out = results / "runs" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    nl.write_env(results)
    plan = planned(cfg, fits)
    t_start = time.time()

    def done(p):
        return p.exists() and not (retry_errors and json.loads(p.read_text()).get("status") == "error")

    for row in targets.sort_values("frozen_fit_seconds").itertuples():
        if only and row.instance_id not in only:
            continue
        arms = row.arms.split("|")
        path = {(d, j, arm): out / f"{row.instance_id}__{arm}__{d}{j}.json" for d, j in plan for arm in arms}
        if all(done(p) for p in path.values()):
            continue
        df, y_raw = nl.load_table(base, row)
        y = nl.numeric_target(y_raw)
        X = nl.prep_hgb_X(df.drop(columns=[row.target_column]))
        splits = nl.make_folds(len(y), cfg["n_folds"], seed)
        held = np.concatenate([splits[f][1] for f in cfg["eval_folds"]])
        for design, j, tr, te in fit_rows(splits, cfg, plan):
            levels = np.unique(y[tr])
            for arm in arms:
                p = path[(design, j, arm)]
                if done(p):
                    continue
                rec = {"seed": seed, "instance_id": row.instance_id, "arm": arm, "task": nl.TASK_OF[arm],
                       "design": design, "fold": j, "k": int(len(np.unique(y))), "n_train": len(tr), "n_test": len(te),
                       "eval_rows_in_train": int(np.intersect1d(tr, held).size) if design == "nested" else None,
                       "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                t0 = time.time()
                try:
                    dec, info = nl.fit_predict_hgb(arm, X.iloc[tr], y[tr], X.iloc[te], cfg["model_seed"], cfg["max_ordinal"])
                    rec.update(status="ok", fit_seconds=round(time.time() - t0, 2), **info,
                               metrics={d: nl.compute_metrics(y[te], v, levels, cfg["max_ordinal"]) for d, v in dec.items()})
                except nl.Skip as e:
                    rec.update(status="skipped", message=str(e), fit_seconds=round(time.time() - t0, 2))
                except Exception as e:  # recorded, not retried silently
                    rec.update(status="error", message=repr(e), traceback=traceback.format_exc(),
                               fit_seconds=round(time.time() - t0, 2))
                nl._atomic(p, lambda t: t.write_text(json.dumps(rec, indent=1, default=str)))
                log(f"seed {seed} {p.stem}: {rec['status']} ({rec['fit_seconds']}s, "
                    f"{(time.time() - t_start) / 60:.1f} min so far)"
                    + (f"  {rec.get('message', '')[:120]}" if rec["status"] != "ok" else ""))


def run_seeds(base, results, seeds=None, log=print):
    """Every fold seed of config.json, in order, so that an interrupted run leaves complete early seeds."""
    cfg, _ = nl.load_bundle(base)
    for seed in (cfg["fold_seeds"] if seeds is None else seeds):
        log(f"=== fold seed {seed} ===")
        run(base, results, seed, log=log)


def run_check(base, results, only=None, log=print):
    """The fits of seed 0 that are compared with the frozen run."""
    cfg, _ = nl.load_bundle(base)
    run(base, results, cfg["check_seed"], only=only, fits=cfg["check_fits"], log=log)


def summarize(results, base):
    """runs_long_seeds.csv (one row per seed, target, arm, design, test fold and decision), seed_status.csv, check_seed0.csv."""
    results = Path(results)
    cfg, targets = nl.load_bundle(base)
    recs = [json.loads(p.read_text()) for p in sorted((results / "runs").glob("seed*/*.json"))]
    ok = [r for r in recs if r["status"] == "ok"]
    cols = ["seed", "instance_id", "arm", "task", "design", "fold", "k", "n_train", "n_test", "eval_rows_in_train"]
    long = pd.DataFrame([{**{c: r[c] for c in cols}, "decision": d, **{m: r["metrics"][d][m] for m in nl.METRICS},
                          "fit_seconds": r["fit_seconds"]} for r in ok for d in r["metrics"]],
                        columns=cols + ["decision", *nl.METRICS, "fit_seconds"])
    long.to_csv(results / "runs_long_seeds.csv", index=False)

    n_arms = int(targets.arms.str.split("|").str.len().sum())
    expected = n_arms * (len(cfg["dev_folds"]) + len(cfg["eval_folds"]))
    status = []
    for seed in cfg["fold_seeds"]:
        s = [r for r in recs if r["seed"] == seed]
        n = {st: sum(r["status"] == st for r in s) for st in ("ok", "skipped", "error")}
        status.append({"seed": seed, "expected": expected, **n, "missing": expected - len(s),
                       "complete": len(s) == expected and n["error"] == 0,
                       "fit_minutes": round(sum(r["fit_seconds"] for r in s) / 60, 1)})
    status = pd.DataFrame(status)
    status.to_csv(results / "seed_status.csv", index=False)
    print(status.to_string(index=False))

    native = long[long.decision == "native"]
    print(f"evaluation rows in any nested training set: "
          f"{int(native[native.design == 'nested'].eval_rows_in_train.fillna(0).sum())}")
    frozen = pd.read_csv(Path(base) / "frozen_seed0.csv")
    chk = native[native.seed == cfg["check_seed"]].merge(frozen, on=["design", "instance_id", "arm", "fold"],
                                                         suffixes=("", "_frozen"))
    for m in ("mae", "rmse", "exact_acc", "qwk"):
        chk[f"diff_{m}"] = (chk[m] - chk[f"{m}_frozen"]).abs()
    chk.to_csv(results / "check_seed0.csv", index=False)
    if len(chk):
        d = chk[[c for c in chk.columns if c.startswith("diff_")]]
        same = (d.fillna(0).max(axis=1) < 1e-9)
        print(f"seed 0 against the frozen fits: {int(same.sum())}/{len(chk)} identical, largest difference {d.max().max():.2g}; "
              f"{n_arms * len(cfg['check_fits'])} expected for the full check")
        for r in chk[~same].itertuples():
            print(f"  differs: {r.instance_id} {r.arm} {r.design}{r.fold}")
    for r in recs:
        if r["status"] != "ok":
            print(f"  {r['status']} seed {r['seed']} {r['instance_id']} {r['arm']} {r['design']}{r['fold']}: "
                  f"{r.get('message', '')[:150]}")
    return long, status, chk
