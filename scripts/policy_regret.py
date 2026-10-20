"""Experiment 4 (H5), offline: downstream regret of each system's task choice on gold, and the cost-aware ask rule.

Definitions fixed 2026-10-18, before any number was computed (the method itself is frozen, F22).

Inputs: results/downstream/hgb_fits/runs_long.csv (5-fold HGB fits of every applicable formulation per target; no labels used),
        benchmark/reference_instances.csv (frozen), results/scores/reference_scores_long.csv (every system's answers).
Scope:  the 35 gold targets with 2+ formulations fitted. Binary and text targets admit one formulation, so no downstream cost.

Loss of task a on target x under utility u: L[a, u] = mean over folds of the best formulation of a
(u ∈ RMSE, MAE, 1 − accuracy, 1 − QWK; accuracy and QWK only with ≤ 30 levels). Only CONTINUOUS has two formulations
(squared and absolute loss); on the evaluation folds, which one is "best" is decided without them (see below).
Each task t is judged on its own utility u_t: MULTICLASS → 1 − accuracy, ORDINAL → 1 − QWK,
CONTINUOUS → RMSE, COUNT → MAE (as in compare_human.py).
Regret of choosing a when the intended task is t: R(a | t) = (L[a, u_t] − L[t, u_t]) / max(L[t, u_t], 1e-3).
A task that was not fitted for this target (e.g. ORDINAL with > 30 levels) is a pipeline failure: R is set to the target's largest regret.

Intended task: the C2 reference task for single-task gold. For set-valued gold, regret is the mean of R(a | t) over the human set
(uniform: the intent is unknown, and humans could not resolve it).
Policies: act on the system's most likely task, or ask when its answer flags ambiguity (set of 2+ tasks). Asking costs one
question and gives the intended task (regret 0; for set-valued gold the user states their intent).

Cost-aware ask (planned in the scope, label-free): expected regret of acting
  ER(x) = Σ_t p(t | x) · R(a* | t),  a* = argmax p,
with p the v2 ensemble's calibrated probabilities (system v2:forced) and R computed from the target's own fits.
Compared with confidence-based asking (1 − max p) at the same ask rates. Targets without fits have ER = 0.

Fold split (added after the first run, before any interpretation; a correctness fix, not tuning): decisions (ER) use the
fits of folds 0–2 and every realized regret uses folds 3–4, so the ask rule is not scored on the estimates it decided from.
Every system is also scored acting always on its most likely task (`forced_*`), because heuristic_names and Qwen flag almost
every numeric target.

    .venv/Scripts/python scripts/policy_regret.py            # realized regret from the HGB fits
    .venv/Scripts/python scripts/policy_regret.py --eval ag  # realized regret from the AutoGluon fits (colab/task_inference_e2e)
    .venv/Scripts/python scripts/policy_regret.py --eval ag_argmax  # sensitivity, see below
writes results/downstream/policy_{hgb,ag,ag_argmax}/{policy_regret.md, policy_regret_long.csv, ask_curves.csv}
Decisions (the cost-aware rule) always use the HGB fits of folds 0–2.

Sensitivity (post-hoc, added 2026-10-19 after the AutoGluon results): `ag_argmax` scores the AutoGluon ORDINAL arm by
the argmax of its class distribution instead of the pre-registered median (both are stored in every run file). On
childrenAnemia (I030) the argmax is perfect, but every median error is exactly one level (flat probabilities), so the
ORDINAL arm loses 0.29 QWK. `ag` stays the primary analysis.

Nested decision fits (post hoc, declared 2026-10-19 after the fourth external reading): the HGB fits of folds 0–2 were
trained on the other four folds, which include the evaluation folds 3–4. When results/downstream/hgb_nested_fits/
exists (colab/task_inference_nested), the ask curves are also computed with decisions from fits trained on folds 0–2
only (each fold tested by a model trained on the other two). ask_curves.csv has a `decide` column: `shared` (the
original fits) or `nested`. Realized regrets are unchanged.

Formulation choice on the evaluation folds (post hoc, declared 2026-10-20 after the fifth external reading): until then
the realized regret took, for each target and utility, the better CONTINUOUS formulation on folds 3–4 themselves, an
optimistic choice. `SELECT` now fixes it without folds 3–4: `decision` (reported) takes the formulation with the lower
loss on the nested decision fits of folds 0–2, ties to the formulation whose loss matches the utility; `fixed` takes
that matching formulation (squared for RMSE, accuracy and QWK, absolute for MAE); `eval` is the original choice, kept
for the sensitivity analysis. Every other task has one formulation, so only CONTINUOUS in the HGB fits is affected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AG_RUNS = ROOT / "results" / "downstream" / "autogluon_fits" / "runs_long_ag.csv"
NESTED_RUNS = ROOT / "results" / "downstream" / "hgb_nested_fits" / "runs_long_nested.csv"
DECIDES = ["shared"] + (["nested"] if NESTED_RUNS.exists() else [])  # decision designs available
TASKS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]
OWN = {"MULTICLASS": "exact_acc", "ORDINAL": "qwk", "CONTINUOUS_REGRESSION": "rmse", "COUNT_REGRESSION": "mae"}
HIGHER = {"exact_acc": True, "qwk": True, "rmse": False, "mae": False}
SYSTEMS = {"autogluon": "AutoGluon rule", "sklearn": "sklearn rule", "heuristic_names": "heuristic set + names",
           "llm:Qwen_Qwen2_5_7B_Instruct": "Qwen2.5-7B", "method:forced": "method v1 (forced)",
           "method:conformal_a05": "method v1 conformal α=0.05", "v2:forced": "v2 ensemble (forced)",
           "v2:ask10": "v2 ask budget 10%", "v2:ask20": "v2 ask budget 20%", "v2:ask30": "v2 ask budget 30%",
           "sortinghat": "SortingHat", "sortinghat:forced": "SortingHat (forced)",
           "llm:Qwen_Qwen3_14B": "Qwen3-14B", "llm:microsoft_phi_4": "Phi-4"}
NEVER_ASK = {"autogluon", "sklearn", "method:forced", "v2:forced", "sortinghat:forced"}
SELECT = "decision"  # how the evaluation folds pick a task's formulation when it has several (see the docstring)
FIXED_ARM = {"rmse": "regression_squared", "mae": "regression_absolute", "exact_acc": "regression_squared",
             "qwk": "regression_squared"}  # the formulation whose training loss matches the utility; squared otherwise


def fold_runs(folds, source="hgb"):
    if source == "ag_argmax":
        recs = [json.loads(p.read_text()) for p in sorted((AG_RUNS.parent / "runs").glob("*.json"))]
        runs = pd.DataFrame([{**{k: r[k] for k in ("instance_id", "group", "arm", "task", "fold")},
                              **(r["metrics_all_decisions"]["native"] if r["task"] == "ORDINAL" else r["metrics"])}
                             for r in recs if r["status"] == "ok"])
    else:
        runs = pd.read_csv(AG_RUNS if source == "ag" else ROOT / "results" / "downstream" / "hgb_fits" / "runs_long.csv")
    return runs[(runs.group == "pilot") & runs.fold.isin(folds)]


def loss_tables(folds, source="hgb"):
    """Loss tables with each task's best formulation on these folds (decision folds 0-2; or `eval_tables`' `eval`)."""
    return tables_from_runs(fold_runs(folds, source))


def formulation_choice(select=SELECT):
    """choose(instance_id, task, utility, arms) -> the arm that scores a task with several formulations."""
    if select == "fixed":
        return lambda iid, task, u, arms: FIXED_ARM[u]
    runs = nested_runs()
    dm = runs[runs.decision == "native"].groupby(["instance_id", "task", "arm"])[list(HIGHER)].mean()

    def choose(iid, task, u, arms):
        loss = dm.loc[(iid, task), u].reindex(arms)
        loss = 1 - loss if HIGHER[u] else loss
        best = list(loss.index[loss == loss.min()])  # empty when the utility does not apply (all NaN)
        return FIXED_ARM[u] if not best or FIXED_ARM[u] in best else sorted(best)[0]
    return choose


def eval_tables(source="hgb", select=None):
    """Loss tables of the evaluation folds 3-4. A task with several formulations (CONTINUOUS in the HGB fits) is
    scored with the formulation `select` (default SELECT) picks without folds 3-4 (`decision`, `fixed`), or the best on
    them (`eval`)."""
    select = select or SELECT
    return tables_from_runs(fold_runs([3, 4], source), None if select == "eval" else formulation_choice(select))


def nested_runs():
    """Nested decision fits (colab/task_inference_nested): one row per target, arm, test fold (0-2) and decision."""
    return pd.read_csv(NESTED_RUNS)


def decision_tables(decide="shared"):
    """Loss tables the cost-aware rule decides from: HGB test folds 0-2, trained on the other four folds (`shared`,
    the original design) or on the other two of folds 0-2 only (`nested`)."""
    if decide == "shared":
        return loss_tables([0, 1, 2])
    runs = nested_runs()
    out = tables_from_runs(runs[runs.decision == "native"])
    shared = loss_tables([0, 1, 2])
    assert set(out) == set(shared) and all(set(out[i].index) == set(shared[i].index) for i in out), \
        "the nested fits must cover the same targets and tasks as the original fits"
    return out


def tables_from_runs(runs, choose=None):
    """Per target: the fold-mean loss of each task under each utility. A task with several formulations takes the one
    `choose` names, or by default the best one on these runs."""
    means = runs.groupby(["instance_id", "task", "arm"])[list(HIGHER)].mean()
    out = {}
    for iid, g in means.groupby(level=0):
        if g.index.get_level_values(1).nunique() < 2:  # one task: no downstream cost (dropped below as before)
            continue
        tab = {}
        for task, gt in g.groupby(level=1):
            gt = gt.droplevel([0, 1])  # index: arm
            tab[task] = {}
            for u in HIGHER:
                loss = 1 - gt[u] if HIGHER[u] else gt[u]
                if choose is None or len(loss) == 1:
                    tab[task][u] = loss.min()
                else:
                    arm = choose(iid, task, u, list(loss.index))
                    assert arm in loss.index, (iid, task, u, arm)
                    tab[task][u] = loss[arm]
        out[iid] = pd.DataFrame(tab).T  # rows: task, cols: utility (loss scale)
    return {k: v for k, v in out.items() if len(v) >= 2}


def regret_matrix(L):
    """R[a][t] for the five tasks; unfitted chosen tasks get the largest finite regret of the target."""
    R = pd.DataFrame(np.nan, index=TASKS, columns=TASKS)
    for t in TASKS:
        if t not in L.index or t not in OWN or pd.isna(L.at[t, OWN[t]]):
            continue
        base = max(L.at[t, OWN[t]], 1e-3)
        for a in TASKS:
            if a in L.index and pd.notna(L.at[a, OWN[t]]):
                R.at[a, t] = (L.at[a, OWN[t]] - L.at[t, OWN[t]]) / base
    worst = np.nanmax(R.to_numpy()) if np.isfinite(R.to_numpy()).any() else 0.0
    for t in R.columns:
        if R[t].notna().any():
            R[t] = R[t].fillna(worst)
    return R


def intended_regret(R, ref_set, a):
    ts = [t for t in ref_set if R[t].notna().any()]
    return float(np.mean([R.at[a, t] for t in ts])) if ts else np.nan


def main(source="hgb"):
    OUT = ROOT / "results" / "downstream" / f"policy_{source}"
    Rs = {iid: regret_matrix(L) for iid, L in eval_tables(source).items()}  # evaluation
    Rds = {dz: {iid: regret_matrix(L) for iid, L in decision_tables(dz).items()} for dz in DECIDES}  # decisions
    inst = pd.read_csv(ROOT / "benchmark" / "reference_instances.csv").set_index("instance_id")
    sc = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    sc = sc[sc.system.isin(SYSTEMS) & sc.instance_id.isin(Rs)]
    rows = []
    for r in sc.itertuples():
        R = Rs[r.instance_id]
        ref = inst.at[r.instance_id, "set_C2"].split("|")
        act = intended_regret(R, ref, r.most_likely_task) if r.most_likely_task in TASKS else np.nanmax(R.to_numpy())
        ask = bool(r.flag) and r.system not in NEVER_ASK
        rows.append({"instance_id": r.instance_id, "system": r.system, "condition": r.condition,
                     "kind": inst.at[r.instance_id, "c2_reference_kind"], "chosen": r.most_likely_task,
                     "regret_if_act": act, "asked": ask, "regret": 0.0 if ask else act})
    long = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT / "policy_regret_long.csv", index=False)

    def summ(g):
        return pd.Series({"n": len(g), "forced_mean_regret": g.regret_if_act.mean(),
                          "forced_share_gt_10pct": (g.regret_if_act > 0.10).mean(),
                          "ask_rate": g.asked.mean(), "mean_regret": g.regret.mean(),
                          "median_regret": g.regret.median(), "share_regret_gt_10pct": (g.regret > 0.10).mean(),
                          "single_mean_regret": g[g.kind == "SINGLE"].regret.mean(),
                          "set_mean_regret": g[g.kind != "SINGLE"].regret.mean()})
    tab = long.groupby(["system", "condition"]).apply(summ, include_groups=False).reset_index()
    tab["system"] = pd.Categorical(tab.system.map(SYSTEMS), list(SYSTEMS.values()), ordered=True)
    tab = tab.sort_values(["condition", "system"])

    # ask curves for the v2 ensemble: confidence vs expected regret, at matched ask rates
    v2 = pd.read_csv(ROOT / "results" / "systems" / "tia_v2" / "predictions.csv")
    v2 = v2[(v2.system == "v2:forced") & v2.instance_id.str.startswith("I")]
    curves = []
    for (cond, g), (decide, Rd) in ((cg, dr) for cg in v2.groupby("condition") for dr in Rds.items()):
        recs = []
        for r in g.itertuples():
            p = np.array([getattr(r, f"p_{t}") for t in TASKS])
            a = TASKS[int(p.argmax())]
            if r.instance_id in Rs and r.instance_id in Rd:
                D = Rd[r.instance_id]
                er = float(sum(p[j] * (D.at[a, t] if D[t].notna().any() else 0.0) for j, t in enumerate(TASKS)))
                act = intended_regret(Rs[r.instance_id], inst.at[r.instance_id, "set_C2"].split("|"), a)
            else:
                er, act = 0.0, 0.0
            recs.append({"instance_id": r.instance_id, "conf": p.max(), "er": er, "act": act})
        d = pd.DataFrame(recs)
        n = len(d)
        for rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            k = int(round(rate * n))
            for rule, order in (("confidence", d.conf.argsort().to_numpy()), ("expected_regret", (-d.er).argsort().to_numpy())):
                asked = np.zeros(n, bool)
                asked[order[:k]] = True
                realized = np.where(asked, 0.0, d.act.fillna(0.0))
                curves.append({"decide": decide, "condition": cond, "rule": rule, "ask_rate": k / n,
                               "mean_regret": realized.mean(), "share_regret_gt_10pct": (realized > 0.10).mean()})
    cur = pd.DataFrame(curves)
    cur.to_csv(OUT / "ask_curves.csv", index=False)
    piv = cur.pivot_table(index=["decide", "condition", "ask_rate"], columns="rule", values="mean_regret").round(3)

    lines = [f"# Experiment 4: downstream regret of task choices on gold (realized regret from {({'hgb': 'HGB', 'ag': 'AutoGluon', 'ag_argmax': 'AutoGluon, ORDINAL arm by argmax (sensitivity)'})[source]} fits)", "",
             f"{len(Rs)} gold targets with 2+ formulations fitted. Regret is relative "
             "loss vs the intended task on that task's own utility, measured on folds 3–4; set-valued gold averages over the "
             f"human set. The CONTINUOUS formulation of each target and utility is chosen by `{SELECT}` (see the "
             "docstring). Asking gives regret 0 at the cost of one question. `forced_*`: the system always acts on its most "
             "likely task. The cost-aware rule decides from HGB test folds 0–2 (`shared`: fits trained on the other four folds; "
             "`nested`: fits trained on the other two of folds 0–2 only).", "",
             "## Policies", "", tab.set_index(["condition", "system"]).to_markdown(floatfmt=".3f"), "",
             "## v2 ensemble: whom to ask (mean realized regret at matched ask rates, all 56 gold targets; no-fit targets have regret 0)",
             "", piv.to_markdown(), ""]
    (OUT / "policy_regret.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["hgb", "ag", "ag_argmax"], default="hgb")
    main(ap.parse_args().eval)
