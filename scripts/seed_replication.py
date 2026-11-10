"""New fold seeds for the nested evaluation of the ask rules (post hoc; declared in docs/findings_log.md, F37, 2026-11-10,
before any fit of a new seed existed). Fits come from colab/task_inference_seeds; nothing is fitted here.

    .venv/Scripts/python scripts/seed_replication.py --check   # seed 0 from the frozen fits must reproduce ask_bayes.csv
    .venv/Scripts/python scripts/seed_replication.py           # scores results/downstream/hgb_seed_fits/

Reuses the functions of policy_regret.py, sensitivity_analyses.py and review_analyses.py, so a new seed is scored by the same
code as seed 0. Only complete seeds count (seed_status.csv).

Writes to results/sensitivity/:
ask_seeds_curves.csv   costly targets left after k = 0..28 questions, per seed, system, condition, measure and policy (seed 0 included)
ask_seeds_summary.csv  confidence minus ER at 6, 11 and 17 questions and in the mean over k = 0..17: the value for seed 0, and over
                       the new seeds the mean, minimum, maximum and the number of seeds favouring each rule. The tables are the same in
                       every seed, so this is a spread, not a sample: no interval or p-value across seeds.
"""
import argparse
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import policy_regret as pr  # noqa: E402
import review_analyses as ra  # noqa: E402
import sensitivity_analyses as sa  # noqa: E402

RUNS = ROOT / "results" / "downstream" / "hgb_seed_fits"
OUT = ROOT / "results" / "sensitivity"
FROZEN = OUT / "ask_bayes.csv"
warnings.filterwarnings("ignore", message="An input array is constant")  # the constant predictors of the normalised regret


@contextmanager
def fold_seed(seed):
    """sa.constant_losses under this fold assignment (the function always uses seed 0)."""
    make = sa.kt.make_folds
    sa.kt.make_folds = lambda n, k, _ignored: make(n, k, seed)
    try:
        yield
    finally:
        sa.kt.make_folds = make


def matrices(nested, ev, seed):
    """Decision and realized regret matrices of one fold assignment. As in review_analyses.setup, with SELECT = 'decision':
    the CONTINUOUS formulation is chosen on the nested fits of this same assignment."""
    Ld = pr.tables_from_runs(nested)
    keep = pr.nested_runs
    pr.nested_runs = lambda: nested
    try:
        Le = pr.tables_from_runs(ev, pr.formulation_choice("decision"))
    finally:
        pr.nested_runs = keep
    D = {i: pr.regret_matrix(L) for i, L in Ld.items()}
    with fold_seed(seed):
        L0 = sa.constant_losses(set(D) | set(Le))
    real = {"rel": {i: pr.regret_matrix(L) for i, L in Le.items()},
            "norm": {i: sa.regret_matrix_norm(L, L0[i]) for i, L in Le.items()}}
    return D, real


def curves(seed, ref, D, real):
    rows = []
    for system in ra.SYSTEMS:
        pred = ra.probabilities(system)
        for cond, g in pred.groupby("condition", sort=False):
            d = ra.frame(g, ref, D, real)
            o = ra.orders(d)
            for meas in ("rel", "norm"):
                a_cost = (d[f"act_{meas}"] > ra.TAU).to_numpy()
                b_cost = (d[f"bayes_{meas}"] > ra.TAU).to_numpy()
                policies = {"act_conf": ra.left_curve(o["confidence"], a_cost, 28),
                            "act_er": ra.left_curve(o["expected_regret"], a_cost, 28),
                            "bayes_er": ra.left_curve(o["bayes_er"], b_cost, 28),
                            "bayes_conf": ra.left_curve(o["confidence"], b_cost, 28)}
                for pol, curve in policies.items():
                    rows += [{"seed": seed, "system": system, "condition": cond, "measure": meas, "policy": pol, "k": k,
                              "costly_left": int(v)} for k, v in enumerate(curve)]
    return pd.DataFrame(rows)


def check_seed0(ref):
    """The frozen seed-0 fits must give exactly the curves of the HGB evaluation in ask_bayes.csv."""
    nested = pr.nested_runs()
    nested = nested[nested.decision == "native"]
    D, real = matrices(nested, pr.fold_runs([3, 4], "hgb"), 0)
    cur = curves(0, ref, D, real)
    frozen = pd.read_csv(FROZEN)
    frozen = frozen[frozen["eval"] == "hgb"].drop(columns="eval")
    keys = ["system", "condition", "measure", "policy", "k"]
    m = cur.merge(frozen, on=keys, suffixes=("", "_frozen"), how="outer", indicator=True)
    assert (m._merge == "both").all(), m[m._merge != "both"].head()
    bad = m[m.costly_left != m.costly_left_frozen]
    assert bad.empty, f"seed 0 does not reproduce ask_bayes.csv:\n{bad.head(20)}"
    print(f"seed 0 reproduces {FROZEN.name} exactly on {len(m)} rows")
    return cur


def summary(cur):
    rows = []
    two = cur[cur.policy.isin(["act_conf", "act_er"])]
    for (system, cond, meas), g in two.groupby(["system", "condition", "measure"], sort=False):
        w = g.pivot_table(index=["seed", "k"], columns="policy", values="costly_left")
        items = {f"k={k}": w.xs(k, level="k") for k in ra.KS}
        items[f"mean_k0_{ra.K_AUC}"] = w[w.index.get_level_values("k") <= ra.K_AUC].groupby(level="seed").mean()
        for label, t in items.items():
            diff = t.act_conf - t.act_er  # positive: the expected-regret rule leaves fewer costly targets
            new = diff[diff.index != 0]
            rows.append({"system": system, "condition": cond, "measure": meas, "summary": label,
                         "seed0_conf": t.act_conf.get(0), "seed0_er": t.act_er.get(0), "seed0_diff": diff.get(0),
                         "n_new_seeds": len(new), "mean_conf": t.act_conf[t.index != 0].mean(),
                         "mean_er": t.act_er[t.index != 0].mean(), "mean_diff": new.mean(), "min_diff": new.min(),
                         "max_diff": new.max(), "seeds_er_fewer": int((new > 0).sum()), "seeds_tied": int((new == 0).sum()),
                         "seeds_conf_fewer": int((new < 0).sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only reproduce seed 0 from the frozen fits")
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--allow-incomplete", action="store_true", help="smoke tests only: score seeds that are not complete")
    args = ap.parse_args()
    inst, _ = sa.load()
    ref = inst.set_index("instance_id")
    cur0 = check_seed0(ref)
    if args.check:
        return
    status = pd.read_csv(args.runs / "seed_status.csv")
    use = status.seed.tolist() if args.allow_incomplete else status[status.complete].seed.tolist()
    left = sorted(set(status.seed) - set(use))
    print(f"seeds scored: {use}" + (f"; not complete, left out: {left}" if left else ""))
    assert use, "no complete seed to score"
    runs = pd.read_csv(args.runs / "runs_long_seeds.csv")
    runs = runs[runs.decision == "native"]
    frames = [cur0]
    for seed in use:
        g = runs[runs.seed == seed]
        D, real = matrices(g[g.design == "nested"], g[g.design == "eval"], seed)
        frames.append(curves(seed, ref, D, real))
    cur = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cur.to_csv(args.out / "ask_seeds_curves.csv", index=False)
    summ = summary(cur)
    summ.to_csv(args.out / "ask_seeds_summary.csv", index=False)
    show = summ[(summ.system == "v2:forced") & (summ.measure == "rel") & (summ["summary"] == "k=6")]
    print(show.drop(columns=["system", "measure", "summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
