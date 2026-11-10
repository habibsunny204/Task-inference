"""Score TIA and the rule baselines on MLTaskBench (F39, declared in docs/findings_log.md on 2026-11-10 before any run).

    .venv/Scripts/python scripts/score_mltaskbench.py [--results results/systems/mltaskbench]

Reads   <results>/predictions.csv, load_log.csv, report.json (from colab/task_inference_mltaskbench, Kaggle)
        data/external/mltaskbench/targets.csv (labels and the overlap rule; `primary` = the 210 targets kept)
Writes  results/sensitivity/mltaskbench_accuracy.csv   top-1 accuracy per system, condition, target set and scoring
                                                       variant, with 95% bootstrap intervals over targets (B = 4000)
        results/sensitivity/mltaskbench_paired.csv     TIA minus each rule: difference, bootstrap interval, McNemar
        results/sensitivity/mltaskbench_per_label.csv  accuracy per MLTaskBench label
        results/sensitivity/mltaskbench_proposals.csv  how often the top-1 task is ORDINAL or COUNT, per label
        results/sensitivity/mltaskbench_targets.csv    per target, system and condition: answer and correctness

Scoring (F39): BINARY = binary, MULTICLASS = multiclass, COUNT or CONTINUOUS = regression, ORDINAL = multiclass or
regression ("lenient", the declared rule); "strict" counts ORDINAL as wrong. A table that could not be read counts as
wrong for every system. Post hoc.
"""
import argparse
import json
import zlib
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sensitivity"
B = 4000
CONDS = ("C0", "C1", "C2")
SYSTEMS = ["tia_v2", "tia_v1", "rule:autogluon", "rule:sklearn", "rule:heuristic_set", "rule:heuristic_names"]
CORRECT = {"BINARY": {"binary"}, "MULTICLASS": {"multiclass"}, "ORDINAL": {"multiclass", "regression"},
           "COUNT_REGRESSION": {"regression"}, "CONTINUOUS_REGRESSION": {"regression"}}


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def correct(task, label, strict=False):
    if not isinstance(task, str) or (strict and task == "ORDINAL"):
        return False
    return label in CORRECT.get(task, set())


def load(results):
    t = pd.read_csv(ROOT / "data" / "external" / "mltaskbench" / "targets.csv")
    pred = pd.read_csv(results / "predictions.csv")
    log = pd.read_csv(results / "load_log.csv")
    rep = json.loads((results / "report.json").read_text())
    assert rep.get("fidelity_passed") and not rep.get("smoke"), "not a full run that passed the fidelity check"
    assert set(pred.instance_id) <= set(t.instance_id)
    grid = pd.MultiIndex.from_product([t.instance_id, SYSTEMS, CONDS], names=["instance_id", "system", "condition"])
    d = pred.set_index(["instance_id", "system", "condition"]).most_likely_task.reindex(grid).reset_index()
    d = d.merge(t[["instance_id", "dataset_name", "label", "primary"]], on="instance_id")
    d["read_ok"] = d.instance_id.isin(log[log.status == "ok"].instance_id)
    d["lenient"] = [correct(a, b) for a, b in zip(d.most_likely_task, d.label)]
    d["strict"] = [correct(a, b, True) for a, b in zip(d.most_likely_task, d.label)]
    return t, d, log, rep


def boot_mean(v, *key):
    v = np.asarray(v, float)
    rng = rng_for(*key)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "systems" / "mltaskbench"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    results, out = Path(a.results), Path(a.out)
    t, d, log, rep = load(results)
    sets = {"primary": d[d.primary], "all": d}

    acc, paired, lab, prop = [], [], [], []
    for sname, s in sets.items():
        for cond in CONDS:
            sc = s[s.condition == cond]
            for var in ("lenient", "strict"):
                for sysname in SYSTEMS:
                    v = sc[sc.system == sysname].sort_values("instance_id")[var].to_numpy()
                    lo, hi = boot_mean(v, "acc", sname, cond, var, sysname)
                    acc.append({"set": sname, "condition": cond, "variant": var, "system": sysname, "n": len(v),
                                "correct": int(v.sum()), "accuracy": v.mean(), "ci_lo": lo, "ci_hi": hi})
                for tia in ("tia_v2", "tia_v1"):
                    x = sc[sc.system == tia].sort_values("instance_id")[var].to_numpy(float)
                    for rule in SYSTEMS[2:]:
                        y = sc[sc.system == rule].sort_values("instance_id")[var].to_numpy(float)
                        lo, hi = boot_mean(x - y, "paired", sname, cond, var, tia, rule)
                        b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                        paired.append({"set": sname, "condition": cond, "variant": var, "system": tia, "versus": rule,
                                       "n": len(x), "diff": (x - y).mean(), "ci_lo": lo, "ci_hi": hi,
                                       "only_system": b, "only_versus": c, "mcnemar_p": mcnemar_p(b, c)})
            for sysname in SYSTEMS:
                ss = sc[sc.system == sysname]
                for label, g in ss.groupby("label"):
                    lab.append({"set": sname, "condition": cond, "system": sysname, "label": label, "n": len(g),
                                "lenient": g.lenient.mean(), "strict": g.strict.mean()})
                    if sysname.startswith("tia"):
                        vc = g.most_likely_task.value_counts()
                        prop.append({"set": sname, "condition": cond, "system": sysname, "label": label, "n": len(g),
                                     **{f"top1_{k}": int(vc.get(k, 0)) for k in CORRECT},
                                     "share_ordinal_or_count": (vc.get("ORDINAL", 0) + vc.get("COUNT_REGRESSION", 0)) / len(g)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acc).to_csv(out / "mltaskbench_accuracy.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "mltaskbench_paired.csv", index=False)
    pd.DataFrame(lab).to_csv(out / "mltaskbench_per_label.csv", index=False)
    pd.DataFrame(prop).to_csv(out / "mltaskbench_proposals.csv", index=False)
    d.sort_values(["instance_id", "system", "condition"]).to_csv(out / "mltaskbench_targets.csv", index=False)

    A = pd.DataFrame(acc)
    view = A[(A.set == "primary") & (A.variant == "lenient")].pivot(index="system", columns="condition", values="accuracy")
    print(f"read {int((log.status == 'ok').sum())} of {len(log)} tables; versions {rep['versions']}")
    print("\nprimary (n = %d), lenient, top-1 accuracy:\n" % d[d.primary].instance_id.nunique()
          + view.reindex(SYSTEMS).round(3).to_string())
    P = pd.DataFrame(paired)
    print("\nTIA-v2 minus the AutoGluon rule (primary, lenient):\n" + P[(P.set == "primary") & (P.variant == "lenient")
          & (P.system == "tia_v2") & (P.versus == "rule:autogluon")][["condition", "diff", "ci_lo", "ci_hi", "only_system",
                                                                      "only_versus", "mcnemar_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
