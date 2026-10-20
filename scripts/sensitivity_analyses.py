"""Post-hoc sensitivity analyses for the paper (declared in docs/findings_log.md on 2026-10-19, after an external
pre-submission review). Frozen outputs only; nothing is re-fitted. Every number is written to results/sensitivity/.

    .venv/Scripts/python scripts/sensitivity_analyses.py

1. top1_family.csv       top-1 on the 34 single-task targets with a family-level bootstrap (27 families), and without
                         the five mfeat targets (one family of handwritten-digit tables).
2. paired.csv            paired differences TIA - other system: won/lost, exact McNemar p, Newcombe (1998, method 10)
                         95% interval, and a family-level bootstrap 95% interval; all 34 targets and without mfeat.
3. regret_thresholds.csv share of the 35 multi-formulation targets whose relative regret exceeds tau (always acting),
                         tau in 0, 2, 5, 10, 20, 50, 100%; the median and the mean with regret capped at 1.
4. regret_normalised*.csv normalised regret: (L(a,u_t) - L(t,u_t)) / (L0(u_t) - L(t,u_t)), the share of the intended
                         formulation's gain over a constant predictor that the chosen task loses. L0 is the loss of a
                         constant fitted on the training folds (mean for RMSE, median for MAE, majority class for
                         accuracy; a constant has QWK 0), on the same folds 3-4 as the realized regret. Unfitted chosen
                         tasks get the target's largest value, as in policy_regret.py.
5. ask_robust.csv        costly targets left after k questions (confidence vs expected regret, TIA-v2), for several
                         thresholds and both regret measures. The ask rule is unchanged (relative regret, HGB folds 0-2).
6. costly_confusions.csv every costly choice (relative regret > 10%) of the AutoGluon rule, TIA-v1, TIA-v2 and the 14B
                         LLMs, with the intended task's own utility under the intended and the chosen formulation.
7. documentation.csv     RQ1 and C2 ambiguity metrics with and without the six targets that have no documentation.
8. ordinal_subtype.csv   top-1 on the nine ORDINAL references, split into originally ordinal and discretised regression.
9. count_answers.csv     how often each system answers COUNT, and how often COUNT is in the human C2 set.
10. robustness_*.csv     only when results/systems/tia_robustness/ exists (colab/task_inference_robustness): the COUNT
                         diagnostic on out-of-fold training predictions and the fidelity report.

Added after a second external reading (declared in docs/findings_log.md, 2026-10-19; frozen outputs only):
11. ask_robust.csv       also rule `expected_regret_norm`: ER from normalised regret estimates on HGB folds 0-2 (the
                         constant predictor fitted on the training part of folds 0-2), so the rule estimates the measure
                         it is scored on.
12. ask_plugin.csv       cost-aware vs confidence-based asking with other probability sources (TIA-v2, Phi-4,
                         Qwen3-14B) in Equation 3: a* is the system's most likely task (argmax p if it answers NONE),
                         p renormalised to sum to 1. LLM confidences take few distinct values, so every count is the
                         expected number of costly targets left under uniformly random tie-breaking (exact).
13. documentation_flips.csv every target on which TIA-v1 or TIA-v2 changes its most likely task between conditions, with
                         the human C2 set, the HGB regret per condition and the documentation lexicon cues that fire.
14. count_targets.csv    answers of TIA and the 14B LLMs on the targets with COUNT in the human C2 set.

B2 proxy (declared in docs/findings_log.md, 2026-10-19; only when results/downstream/hgb_proxy_fits/ exists, returned
from colab/task_inference_proxy):
15. ask_robust.csv       also rules `expected_regret_proxy_{median,argmax}` and `expected_regret_norm_proxy_{...}`: ER
                         from HGB folds 0-2 with the ORDINAL row replaced by the refitted multiclass HGB under
                         AutoGluon's decision rule (median, or argmax for the argmax evaluation).
    proxy_fidelity.csv   argmax metrics of the refit against the frozen multiclass run.

Fourth external reading (declared in docs/findings_log.md, 2026-10-19, before the code was written):
16. clipped regret       the relative and normalised regret with negative entries set to zero before averaging over a
                         human set (an excess-loss measure). Realized: measures `rel_clip`, `norm_clip` in
                         ask_robust.csv and `share_rel_clip`, `share_norm_clip` in regret_thresholds.csv. Decision:
                         rules `expected_regret_clip`, `expected_regret_norm_clip` (ER from clipped estimates).
    er_scores.csv        TIA-v2 per target: confidence, ER under every rule and the realized regret under every
                         measure (per evaluation, decision design and condition), for the per-target statements.
    regret_sign.csv      how many off-diagonal regret entries are negative (decision and evaluation tables), and on how
                         many targets the ER of TIA-v2's choice is negative.
17. nested decisions     when results/downstream/hgb_nested_fits/ exists (colab/task_inference_nested), every rule is
                         also computed from decision fits trained on folds 0-2 only (the proxy from the nested
                         multiclass fits, the constant predictor of the normalised estimates fitted on the same rows).
                         ask_robust.csv and ask_plugin.csv have a `decide` column: `shared` (the original fits of folds
                         0-2, trained on the other four folds) or `nested`.

Fifth external reading (declared in docs/findings_log.md, 2026-10-20, before any number of it was computed):
18. formulation choice   every realized regret now scores CONTINUOUS (HGB: squared or absolute loss) with the formulation
                         chosen without folds 3-4 (policy_regret.SELECT = "decision": lower loss on the nested decision
                         fits). formulation_choice.csv: per target and utility, the formulation under `decision`,
                         `fixed` (the loss that matches the utility; squared for accuracy and QWK) and `eval` (the
                         original choice on folds 3-4), with the losses on both fold sets. formulation_counts.csv:
                         costly choices (relative regret > 10%) under each choice, always acting (every system) and
                         after k questions (TIA-v2, confidence and ER, both decision designs).
19. ask_grid.csv         costly targets left after six questions on a 0.5-point grid of thresholds from 5% to 50%
                         (TIA-v2, confidence and ER, every evaluation and decision design), from er_scores.csv: the
                         record of the "no more costly choices from 5% to 50%" statement (F33), which was a check outside
                         the scripts until then.
"""
import json
import sys
import zlib
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "colab" / "task_inference_killtest"))
import killtest_lib as kt  # noqa: E402
import policy_regret as pr  # noqa: E402

OUT = ROOT / "results" / "sensitivity"
CONDS = ["C0", "C1", "C2"]
TASKS = pr.TASKS
B = 4000
Z95 = 1.959963984540054
MFEAT = "mfeat"
EVALS = ["hgb", "ag", "ag_argmax"]
TAUS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
MAIN = ["autogluon", "sortinghat:forced", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4",
        "method:forced", "v2:forced"]
COSTLY_SYS = ["autogluon", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4", "method:forced", "v2:forced"]
PLUGIN_SYS = ["v2:forced", "llm:microsoft_phi_4", "llm:Qwen_Qwen3_14B"]
K_ASK = (0, 3, 6, 11, 17, 28)
PROXY = ROOT / "results" / "downstream" / "hgb_proxy_fits"
PROXY_RULES = {"proxy_median": "median", "proxy_argmax": "native"}  # rule suffix -> decision of the refitted model


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def wilson(x, n, z=Z95):
    p, d = x / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(a, b, c, d, z=Z95):
    """Newcombe (1998) method 10: interval for p1 - p2 of paired proportions (a both right, b only 1, c only 2)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if den == 0 else (a * d - b * c) / sqrt(den)
    lo = sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return p1 - p2, p1 - p2 - lo, p1 - p2 + hi


def cluster_boot(values, groups, *key):
    """95% percentile interval of a mean, resampling whole groups (ratio of sums)."""
    g = pd.Series(np.asarray(values, float)).groupby(np.asarray(groups))
    sums, ns = g.sum().to_numpy(), g.size().to_numpy()
    idx = rng_for(*key).integers(0, len(sums), (B, len(sums)))
    return tuple(np.percentile(sums[idx].sum(1) / ns[idx].sum(1), [2.5, 97.5]))


# ---------------------------------------------------------------------------------------------- inputs
def load():
    pool = pd.read_csv(ROOT / "benchmark" / "silver" / "pool_silver.csv")[["candidate_id", "family"]]
    inst = pd.read_csv(ROOT / "benchmark" / "reference_instances.csv").merge(pool, on="candidate_id", how="left")
    key = pd.read_csv(ROOT / "benchmark" / "annotation_key.csv").drop_duplicates("instance_id")
    inst = inst.merge(key[["instance_id", "origin", "has_documentation"]], on="instance_id", how="left")
    assert inst.family.notna().all(), "every benchmark target needs a family"
    long = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    long = long[long.group == "pilot"].merge(inst[["instance_id", "family", "origin", "has_documentation"]],
                                             on="instance_id", how="left")
    return inst, long


# ---------------------------------------------------------------------------------------------- 1-2: top-1
def top1_tables(long):
    s = long[long.single.astype(bool)].copy()
    s["top1"] = s.top1.astype(float)
    rows = []
    for (system, cond), g in s.groupby(["system", "condition"]):
        lo, hi = cluster_boot(g.top1, g.family, system, cond, "top1")
        nm = g[g.family != MFEAT]
        rows.append({"system": system, "condition": cond, "n": len(g), "n_families": g.family.nunique(),
                     "top1": g.top1.mean(), "family_ci_lo": lo, "family_ci_hi": hi,
                     "n_no_mfeat": len(nm), "top1_no_mfeat": nm.top1.mean()})
    top = pd.DataFrame(rows)
    pairs = [(a, b) for a in ("method:forced", "v2:forced") for b in MAIN if b not in ("method:forced", "v2:forced")]
    rows = []
    for cond in CONDS:
        piv = s[s.condition == cond].pivot_table(index="instance_id", columns="system", values="top1", aggfunc="first")
        fam = s[s.condition == cond].drop_duplicates("instance_id").set_index("instance_id").family
        for a, b in pairs:
            for subset in ("all", "no_mfeat"):
                ids = piv.index if subset == "all" else piv.index[fam.loc[piv.index] != MFEAT]
                x, y = piv.loc[ids, a].astype(bool), piv.loc[ids, b].astype(bool)
                n11, n10, n01, n00 = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())
                d, lo, hi = newcombe(n11, n10, n01, n00)
                flo, fhi = cluster_boot(x.astype(float) - y.astype(float), fam.loc[ids], a, b, cond, subset, "diff")
                rows.append({"system": a, "other": b, "condition": cond, "subset": subset, "n": len(ids),
                             "won": n10, "lost": n01, "diff": d, "newcombe_lo": lo, "newcombe_hi": hi,
                             "family_boot_lo": flo, "family_boot_hi": fhi, "mcnemar_p": mcnemar_p(n10, n01)})
    return top, pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 3-6: regret
def constant_losses(ids, folds=(3, 4), nested=False):
    """Loss of a constant predictor fitted on the training folds, averaged over the given folds (same splits as the fits).
    nested: the training rows of test fold j are the other folds in `folds` (as in colab/task_inference_nested)."""
    from instances import pilot_instances
    out = {}
    for inst in pilot_instances():
        if inst.instance_id not in ids:
            continue
        y = kt.numeric_target(inst.df[inst.target].dropna())
        per = []
        splits = kt.make_folds(len(y), 5, 0)
        for fold, (tr, te) in enumerate(splits):
            if fold not in folds:
                continue
            if nested:
                tr = np.sort(np.concatenate([splits[f][1] for f in folds if f != fold]))
            levels = np.unique(y[tr])
            vals, cnt = np.unique(y[tr], return_counts=True)
            mode = vals[cnt.argmax()]
            m_mean = kt.compute_metrics(y[te], np.full(len(te), y[tr].mean()), levels, pr_max_levels())
            m_med = kt.compute_metrics(y[te], np.full(len(te), np.median(y[tr])), levels, pr_max_levels())
            m_mode = kt.compute_metrics(y[te], np.full(len(te), mode), levels, pr_max_levels())
            per.append({"rmse": m_mean["rmse"], "mae": m_med["mae"], "exact_acc": 1 - m_mode["exact_acc"],
                        "qwk": 1 - (m_mode["qwk"] if np.isfinite(m_mode["qwk"]) else 0.0) if len(levels) <= pr_max_levels() else np.nan,
                        "n_test": len(te)})
        out[inst.instance_id] = pd.DataFrame(per).mean()
    return out


def pr_max_levels():
    return 30  # run_downstream_ambiguity.MAX_ORDINAL: accuracy and QWK only with at most 30 levels


def regret_matrix_norm(L, L0):
    R = pd.DataFrame(np.nan, index=TASKS, columns=TASKS)
    for t in TASKS:
        if t not in L.index or t not in pr.OWN or pd.isna(L.at[t, pr.OWN[t]]):
            continue
        u = pr.OWN[t]
        gain = L0[u] - L.at[t, u]
        if not np.isfinite(gain) or gain <= 0:
            continue  # the intended formulation does not beat a constant: no gain to lose
        for a in TASKS:
            if a in L.index and pd.notna(L.at[a, u]):
                R.at[a, t] = (L.at[a, u] - L.at[t, u]) / gain
    worst = np.nanmax(R.to_numpy()) if np.isfinite(R.to_numpy()).any() else 0.0
    for t in R.columns:
        if R[t].notna().any():
            R[t] = R[t].fillna(worst)
    return R


def load_proxy(base=None):
    """Refitted multiclass HGB (colab/task_inference_proxy), or None before the run; with its fidelity table."""
    base = base or PROXY
    if not (base / "runs_long_proxy.csv").exists():
        return None, None
    runs = pd.read_csv(base / "runs_long_proxy.csv")
    frozen = pd.read_csv(ROOT / "results" / "downstream" / "hgb_fits" / "runs_long.csv")
    frozen = frozen[(frozen.group == "pilot") & (frozen.arm == "multiclass")]
    fid = runs[runs.decision == "native"].merge(frozen, on=["instance_id", "fold"], suffixes=("", "_frozen"))
    for m in ("mae", "rmse", "exact_acc", "qwk"):
        fid[f"diff_{m}"] = (fid[m] - fid[f"{m}_frozen"]).abs()
    return runs, fid[["instance_id", "fold"] + [c for c in fid.columns if c.startswith("diff_")]]


def proxy_loss_tables(Ld, runs, decision):
    """Folds 0-2 loss tables with the ORDINAL row replaced by the refitted multiclass HGB under `decision`."""
    r = runs[(runs.decision == decision) & runs.fold.isin([0, 1, 2])]
    n = r.groupby("instance_id").fold.nunique()
    out = {}
    for i, L in Ld.items():
        L = L.copy()
        if "ORDINAL" in L.index:
            assert n.get(i, 0) == 3, f"{i}: the proxy run needs folds 0-2"
            m = r[r.instance_id == i][list(pr.HIGHER)].mean()
            L.loc["ORDINAL", list(pr.HIGHER)] = [(1 - m[u]) if pr.HIGHER[u] else m[u] for u in pr.HIGHER]
        out[i] = L
    return out


def expected_regret(p, a, D):
    return float(sum(p[j] * (D.at[a, t] if D[t].notna().any() else 0.0) for j, t in enumerate(TASKS)))


def expected_left(score, costly, k):
    """Expected number of costly targets left when asking on the k highest scores, ties broken uniformly at random."""
    score, costly = np.asarray(score, float), np.asarray(costly, bool)
    left = 0.0
    for s in score[costly]:
        above, tie = int((score > s).sum()), int((score == s).sum())
        p_ask = 0.0 if above >= k else 1.0 if above + tie <= k else (k - above) / tie
        left += 1.0 - p_ask
    return left


def clip(Rs):
    """Excess loss (item 16): negative regrets, where the chosen formulation does better than the intended one, set to 0."""
    return {i: R.clip(lower=0) for i, R in Rs.items()}


def decision_matrices(decide):
    """Every regret table the ask rules decide from, under one decision design (item 17)."""
    Ld = pr.decision_tables(decide)
    L0d = constant_losses(set(Ld), (0, 1, 2), nested=decide == "nested")
    D = {"expected_regret": {i: pr.regret_matrix(L) for i, L in Ld.items()},
         "expected_regret_norm": {i: regret_matrix_norm(L, L0d[i]) for i, L in Ld.items()}}
    D["expected_regret_clip"] = clip(D["expected_regret"])
    D["expected_regret_norm_clip"] = clip(D["expected_regret_norm"])
    # B2: proxy decision tables (none before the Colab run); the nested design uses its own multiclass fits
    if decide == "nested":
        proxy = pr.nested_runs()
        proxy = proxy[proxy.arm == "multiclass"]
    else:
        proxy, _ = load_proxy()
    if proxy is not None:
        for name, decision in PROXY_RULES.items():
            Lp = proxy_loss_tables(Ld, proxy, decision)
            D[f"expected_regret_{name}"] = {i: pr.regret_matrix(L) for i, L in Lp.items()}
            D[f"expected_regret_norm_{name}"] = {i: regret_matrix_norm(L, L0d[i]) for i, L in Lp.items()}
    return D


def regret_analyses(inst):
    ref = inst.set_index("instance_id")
    sc = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    sc = sc[sc.group == "pilot"]
    DEC = {dz: decision_matrices(dz) for dz in pr.DECIDES}
    L0 = constant_losses(set(DEC["shared"]["expected_regret"]))
    v2 = pd.read_csv(ROOT / "results" / "systems" / "tia_v2" / "predictions.csv")
    v2 = v2[(v2.system == "v2:forced") & v2.instance_id.str.startswith("I")]
    thr_rows, norm_rows, ask_rows, costly_rows, gains, plug_rows, er_rows = [], [], [], [], [], [], []
    sign_rows = [{"tables": f"decision ({dz})", **negative_entries(D["expected_regret"])} for dz, D in DEC.items()]
    for ev in EVALS:
        Le = pr.eval_tables(ev)
        Rs = {i: pr.regret_matrix(L) for i, L in Le.items()}
        Rn = {i: regret_matrix_norm(L, L0[i]) for i, L in Le.items()}
        REAL = {"rel": Rs, "norm": Rn, "rel_clip": clip(Rs), "norm_clip": clip(Rn)}  # realized measures (16: clipped)
        sign_rows.append({"tables": f"evaluation ({ev})", **negative_entries(Rs)})
        for i, L in Le.items():
            for t in L.index:
                if t in pr.OWN:
                    u = pr.OWN[t]
                    gains.append({"eval": ev, "instance_id": i, "task": t, "utility": u, "loss_intended": L.at[t, u],
                                  "loss_constant": L0[i][u], "gain": L0[i][u] - L.at[t, u]})
        acts = []
        for r in sc[sc.instance_id.isin(Rs)].itertuples():
            refset = ref.at[r.instance_id, "set_C2"].split("|")
            vals = {}
            for m, Rm in REAL.items():
                R = Rm[r.instance_id]
                vals[f"regret_{m}"] = (pr.intended_regret(R, refset, r.most_likely_task) if r.most_likely_task in TASKS
                                       else np.nanmax(R.to_numpy()))
            acts.append({"eval": ev, "system": r.system, "condition": r.condition, "instance_id": r.instance_id,
                         "chosen": r.most_likely_task, "intended": ref.at[r.instance_id, "set_C2"], **vals})
        acts = pd.DataFrame(acts)
        norm_rows.append(acts)
        for (system, cond), g in acts.groupby(["system", "condition"]):
            for tau in TAUS:
                thr_rows.append({"eval": ev, "system": system, "condition": cond, "tau": tau, "n": len(g),
                                 **{f"share_{m}": (g[f"regret_{m}"] > tau).mean() for m in REAL}})
        # costly confusions (relative regret > 10%), with the intended task's own utility values
        for r in acts[acts.system.isin(COSTLY_SYS) & (acts.regret_rel > 0.10)].itertuples():
            L = Le[r.instance_id]
            for t in r.intended.split("|"):
                if t in L.index and t in pr.OWN:
                    u = pr.OWN[t]
                    costly_rows.append({"eval": ev, "system": r.system, "condition": r.condition,
                                        "instance_id": r.instance_id, "intended_set": r.intended, "intended": t,
                                        "chosen": r.chosen, "utility": u, "loss_intended": L.at[t, u],
                                        "loss_chosen": L.at[r.chosen, u] if r.chosen in L.index else np.nan,
                                        "loss_constant": L0[r.instance_id][u], "regret_rel": r.regret_rel,
                                        "regret_norm": r.regret_norm})
        # ask curves: the frozen rule is `expected_regret` (relative regret from HGB folds 0-2); realized under every
        # measure. expected_regret_norm (post hoc, second reading): the same rule on normalised estimates; *_clip (16):
        # on clipped estimates; *_proxy_* (15): from the AutoGluon-like proxy. One block per decision design (17).
        for (cond, g), (decide, D) in ((cg, dd) for cg in v2.groupby("condition") for dd in DEC.items()):
            recs = []
            for r in g.itertuples():
                p = np.array([getattr(r, f"p_{t}") for t in TASKS])
                a = TASKS[int(p.argmax())]
                rec = {"conf": p.max(), **{m: 0.0 for m in REAL}, **{rule: 0.0 for rule in D}}
                if r.instance_id in Rs and r.instance_id in D["expected_regret"]:
                    rec.update({rule: expected_regret(p, a, Dr[r.instance_id]) for rule, Dr in D.items()})
                    refset = ref.at[r.instance_id, "set_C2"].split("|")
                    for m, Rm in REAL.items():
                        v = pr.intended_regret(Rm[r.instance_id], refset, a)
                        rec[m] = 0.0 if pd.isna(v) else v
                recs.append(rec)
            d = pd.DataFrame(recs)
            er_rows.append(d.assign(eval=ev, decide=decide, condition=cond, instance_id=g.instance_id.to_numpy(),
                                    chosen=[TASKS[int(np.argmax([getattr(r, f"p_{t}") for t in TASKS]))]
                                            for r in g.itertuples()]))
            orders = {"confidence": d.conf.argsort(kind="stable").to_numpy(),
                      **{rule: (-d[rule]).argsort(kind="stable").to_numpy() for rule in D}}
            for k in K_ASK:
                for rule, order in orders.items():
                    asked = np.zeros(len(d), bool)
                    asked[order[:k]] = True
                    for measure in REAL:
                        for tau in (0.05, 0.10, 0.20, 0.50):
                            ask_rows.append({"eval": ev, "decide": decide, "condition": cond, "k": k, "n": len(d),
                                             "rule": rule, "measure": measure, "tau": tau,
                                             "costly_left": int(((~asked) & (d[measure] > tau)).sum())})
        # Equation 3 with other probability sources (post hoc, second reading); expected counts over random tie orders
        for decide, D in DEC.items():
            Rd = D["expected_regret"]
            for system in PLUGIN_SYS:
                for cond, g in sc[sc.system == system].groupby("condition"):
                    recs = []
                    for r in g.itertuples():
                        p = np.array([getattr(r, f"p_{t}") for t in TASKS], float)
                        p = p / p.sum()
                        a = r.most_likely_task if r.most_likely_task in TASKS else TASKS[int(p.argmax())]
                        rel = nrm = er = 0.0
                        if r.instance_id in Rs and r.instance_id in Rd:
                            er = expected_regret(p, a, Rd[r.instance_id])
                            refset = ref.at[r.instance_id, "set_C2"].split("|")
                            rel = pr.intended_regret(Rs[r.instance_id], refset, a)
                            nrm = pr.intended_regret(Rn[r.instance_id], refset, a)
                        recs.append({"conf": p.max(), "er": er, "rel": 0.0 if pd.isna(rel) else rel,
                                     "norm": 0.0 if pd.isna(nrm) else nrm})
                    d = pd.DataFrame(recs)
                    for k in K_ASK:
                        for rule, score in (("confidence", 1 - d.conf), ("expected_regret", d.er)):
                            for measure, tau in [("rel", t) for t in (0.05, 0.10, 0.20, 0.50)] + [("norm", 0.10)]:
                                plug_rows.append({"eval": ev, "decide": decide, "system": system, "condition": cond,
                                                  "k": k, "n": len(d), "rule": rule, "measure": measure, "tau": tau,
                                                  "n_conf_levels": d.conf.nunique(),
                                                  "costly_left": expected_left(score, d[measure] > tau, k)})
    norm_long = pd.concat(norm_rows, ignore_index=True)
    summ = norm_long.groupby(["eval", "system", "condition"]).agg(
        n=("regret_rel", "size"), n_norm_defined=("regret_norm", "count"), median_rel=("regret_rel", "median"),
        mean_rel_capped=("regret_rel", lambda x: x.clip(upper=1).mean()),
        share_rel_gt10=("regret_rel", lambda x: (x > 0.10).mean()),
        median_norm=("regret_norm", "median"), mean_norm_capped=("regret_norm", lambda x: x.clip(upper=1).mean()),
        share_norm_gt10=("regret_norm", lambda x: (x > 0.10).mean()),
        share_norm_gt25=("regret_norm", lambda x: (x > 0.25).mean())).reset_index()
    er = pd.concat(er_rows, ignore_index=True)
    er = er[["eval", "decide", "condition", "instance_id", "chosen", "conf", *[c for c in er.columns if c.startswith(
        "expected_regret")], *REAL]]
    sign = pd.DataFrame(sign_rows)
    neg = er[er["eval"] == "hgb"].groupby(["decide", "condition"]).expected_regret.apply(lambda x: int((x < 0).sum()))
    sign = pd.concat([sign, neg.rename("n_targets_er_negative").reset_index()], ignore_index=True)
    return (pd.DataFrame(thr_rows), norm_long, summ, pd.DataFrame(ask_rows), pd.DataFrame(costly_rows),
            pd.DataFrame(gains), pd.DataFrame(plug_rows), er, sign)


SELECTS = ["decision", "fixed", "eval"]


def formulation_selection(inst):
    """Item 18: which CONTINUOUS formulation each choice rule scores per target and utility, and the costly counts under
    each rule (HGB realized regret; every AutoGluon task has one formulation)."""
    ref = inst.set_index("instance_id")
    cont = "CONTINUOUS_REGRESSION"
    multi = set(pr.eval_tables("hgb", "eval"))  # the 35 multi-formulation targets
    ev = pr.fold_runs([3, 4])
    ev = ev[ev.instance_id.isin(multi) & (ev.task == cont)].groupby(["instance_id", "arm"])[list(pr.HIGHER)].mean()
    nested = pr.nested_runs()
    dm = nested[(nested.decision == "native") & (nested.task == cont)].groupby(["instance_id", "arm"])[
        list(pr.HIGHER)].mean()
    choose = {s: pr.formulation_choice(s) for s in ("decision", "fixed")}
    rows = []
    for iid, g in ev.groupby(level=0):
        g, d = g.droplevel(0), dm.loc[iid]
        for u, higher in pr.HIGHER.items():
            le, ld = (1 - g[u], 1 - d[u]) if higher else (g[u], d[u])
            if le.isna().all():
                continue
            rows.append({"instance_id": iid, "utility": u,
                         **{f"arm_{s}": c(iid, cont, u, list(le.index)) for s, c in choose.items()},
                         "arm_eval": le.idxmin(),
                         **{f"eval_loss_{a.split('_')[1]}": le[a] for a in le.index},
                         **{f"decision_loss_{a.split('_')[1]}": ld[a] for a in ld.index}})
    choice = pd.DataFrame(rows)
    for s in SELECTS:
        choice[f"eval_loss_{s}"] = [r[f"eval_loss_{r[f'arm_{s}'].split('_')[1]}"] for _, r in choice.iterrows()]

    sc = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    sc = sc[(sc.group == "pilot") & sc.system.isin(pr.SYSTEMS)]
    v2 = pd.read_csv(ROOT / "results" / "systems" / "tia_v2" / "predictions.csv")
    v2 = v2[(v2.system == "v2:forced") & v2.instance_id.str.startswith("I")]
    DEC = {dz: {i: pr.regret_matrix(L) for i, L in pr.decision_tables(dz).items()} for dz in pr.DECIDES}
    counts = []
    for s in SELECTS:
        Rs = {i: pr.regret_matrix(L) for i, L in pr.eval_tables("hgb", s).items()}
        for (system, cond), g in sc[sc.instance_id.isin(Rs)].groupby(["system", "condition"]):
            reg = np.array([pr.intended_regret(Rs[r.instance_id], ref.at[r.instance_id, "set_C2"].split("|"),
                                               r.most_likely_task) if r.most_likely_task in TASKS
                            else np.nanmax(Rs[r.instance_id].to_numpy()) for r in g.itertuples()])
            counts.append({"select": s, "policy": "always act", "system": system, "condition": cond, "decide": "",
                           "rule": "", "k": 0, "n": len(g), "costly": int((reg > 0.10).sum()),
                           "median_regret": float(np.nanmedian(reg))})
        for (cond, g), (decide, Rd) in ((cg, dd) for cg in v2.groupby("condition") for dd in DEC.items()):
            recs = []
            for r in g.itertuples():
                p = np.array([getattr(r, f"p_{t}") for t in TASKS])
                a = TASKS[int(p.argmax())]
                er = act = 0.0
                if r.instance_id in Rs and r.instance_id in Rd:
                    er = expected_regret(p, a, Rd[r.instance_id])
                    act = pr.intended_regret(Rs[r.instance_id], ref.at[r.instance_id, "set_C2"].split("|"), a)
                recs.append({"conf": p.max(), "er": er, "act": 0.0 if pd.isna(act) else act})
            d = pd.DataFrame(recs)
            for rule, order in (("confidence", d.conf.argsort(kind="stable").to_numpy()),
                                ("expected_regret", (-d.er).argsort(kind="stable").to_numpy())):
                for k in K_ASK:
                    asked = np.zeros(len(d), bool)
                    asked[order[:k]] = True
                    counts.append({"select": s, "policy": "ask", "system": "v2:forced", "condition": cond,
                                   "decide": decide, "rule": rule, "k": k, "n": len(d),
                                   "costly": int(((~asked) & (d.act > 0.10)).sum()), "median_regret": np.nan})
    return choice, pd.DataFrame(counts)


def ask_grid(er, ask, k=6):
    """Item 19: costly targets (relative regret) left after k questions, on a fine grid of thresholds."""
    grid = np.round(np.arange(0.05, 0.50001, 0.005), 4)
    rows = []
    for (ev, decide, cond), d in er.groupby(["eval", "decide", "condition"], sort=False):  # rows in v2 order
        for rule, order in (("confidence", d.conf.argsort(kind="stable").to_numpy()),
                            ("expected_regret", (-d.expected_regret).argsort(kind="stable").to_numpy())):
            asked = np.zeros(len(d), bool)
            asked[order[:k]] = True
            rows += [{"eval": ev, "decide": decide, "condition": cond, "k": k, "rule": rule, "tau": t,
                      "costly_left": int(((~asked) & (d.rel.to_numpy() > t)).sum())} for t in grid]
    out = pd.DataFrame(rows)
    a = ask[(ask.k == k) & (ask.measure == "rel") & ask.rule.isin(["confidence", "expected_regret"])]
    m = a.merge(out, on=["eval", "decide", "condition", "rule", "tau"], suffixes=("", "_grid"))
    assert len(m) and (m.costly_left == m.costly_left_grid).all(), "the grid must reproduce ask_robust.csv"
    return out


def negative_entries(Rs):
    """Item 16: how often the signed regret is negative (off-diagonal, finite entries of the five-task matrices)."""
    v = [R.to_numpy()[np.isfinite(R.to_numpy()) & ~np.eye(len(TASKS), dtype=bool)] for R in Rs.values()]
    v = np.concatenate(v)
    return {"n_entries": len(v), "n_negative": int((v < 0).sum())}


# ---------------------------------------------------------------------------------------------- 7-9: documentation, ordinal, count
def auroc(score, y):
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def documentation(inst, long):
    doc = inst.has_documentation.astype(bool)
    rows = [{"subset": name, "n": int(m.sum()), "epistemic": int((inst[m].ambiguity_type == "EPISTEMIC").sum()),
             "share_set_valued_C2": float((inst[m].size_C2 > 1).mean()), "mean_set_size_C2": float(inst[m].size_C2.mean()),
             "single_task_C2": int((inst[m].c2_reference_kind == "SINGLE").sum())}
            for name, m in (("all", np.ones(len(inst), bool)), ("documented", doc.to_numpy()), ("undocumented", ~doc.to_numpy()))]
    rq1 = pd.DataFrame(rows)
    sysrows = []
    for system in ["llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4", "ml:logreg", "method:forced", "v2:forced", "v2:ask20",
                   "heuristic_names"]:
        g = long[(long.system == system) & (long.condition == "C2")]
        for name, m in (("all", np.ones(len(g), bool)), ("documented", g.has_documentation.astype(bool).to_numpy())):
            h = g[m]
            amb = h.ref_ambiguous.astype(bool)
            flag = h.flag.astype(bool)
            sysrows.append({"system": system, "subset": name, "n": len(h),
                            "auroc": auroc(1 - h.conf, amb) if h.conf.notna().any() else np.nan,
                            "flag_recall": float(flag[amb].mean()), "flag_specificity": float((~flag[~amb]).mean())})
    return rq1, pd.DataFrame(sysrows)


def ordinal_and_count(long):
    s = long[long.single.astype(bool) & (long.set_C2 == "ORDINAL")].copy()
    s["subtype"] = np.where(s.origin.str.contains("discretised"), "discretised regression", "originally ordinal")
    ordt = s.groupby(["system", "condition", "subtype"]).top1.agg(correct="sum", n="size").reset_index()
    rows = []
    for (system, cond), g in long.groupby(["system", "condition"]):
        said = g[g.most_likely_task == "COUNT_REGRESSION"]
        plaus = g[g.set_C2.str.contains("COUNT")]
        rows.append({"system": system, "condition": cond, "answers_count": len(said),
                     "count_in_human_set": int(said.set_C2.str.contains("COUNT").sum()),
                     "count_plausible_targets": plaus.instance_id.nunique(),
                     "count_top1_on_plausible": int((plaus.most_likely_task == "COUNT_REGRESSION").sum())})
    return ordt, pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 13-14: per-target checks (second reading)
def documentation_flips(long):
    feats = pd.read_csv(ROOT / "results" / "systems" / "tia_v1" / "features.csv").set_index("instance_id")
    cues = [c for c in feats.columns if c.startswith(("d_doc_lex_", "d_win_lex_", "d_title_lex_"))]
    man = pd.read_csv(ROOT / "colab" / "task_inference_sortinghat" / "store" / "manifest.csv").set_index("instance_id")
    pol = pd.read_csv(ROOT / "results" / "downstream" / "policy_hgb" / "policy_regret_long.csv")
    rows = []
    for system in ("method:forced", "v2:forced"):
        g = long[long.system == system]
        ans = g.pivot_table(index="instance_id", columns="condition", values="most_likely_task", aggfunc="first")
        info = g.drop_duplicates("instance_id").set_index("instance_id")
        for iid, a in ans[(ans.C0 != ans.C2) | (ans.C1 != ans.C2)].iterrows():
            refset = info.at[iid, "set_C2"].split("|")
            reg = pol[(pol.system == system) & (pol.instance_id == iid)].set_index("condition").regret_if_act
            rows.append({"system": system, "instance_id": iid, "title": man.at[iid, "title"],
                         "target": man.at[iid, "target_name"], "kind": info.at[iid, "c2_reference_kind"],
                         "set_C2": info.at[iid, "set_C2"], "has_documentation": bool(info.at[iid, "has_documentation"]),
                         **{f"answer_{c}": a[c] for c in CONDS}, **{f"in_set_{c}": a[c] in refset for c in CONDS},
                         **{f"hgb_regret_{c}": reg.get(c, np.nan) for c in CONDS},
                         "doc_cues": ";".join(c.removeprefix("d_") for c in cues if feats.at[iid, c] > 0)})
    return pd.DataFrame(rows)


def count_targets(long):
    g = long[long.set_C2.str.contains("COUNT") & long.system.isin(["method:forced", "v2:forced", "llm:microsoft_phi_4",
                                                                   "llm:Qwen_Qwen3_14B"])]
    out = g.pivot_table(index=["instance_id", "set_C2", "most_likely_C2", "system"], columns="condition",
                        values="most_likely_task", aggfunc="first").reset_index()
    for c in CONDS:
        out[f"in_set_{c}"] = [a in s.split("|") for a, s in zip(out[c], out.set_C2)]
    return out


# ---------------------------------------------------------------------------------------------- 10: Colab robustness run
def robustness():
    base = ROOT / "results" / "systems" / "tia_robustness"
    if not (base / "diagnostics" / "oof_training.csv").exists():
        return None
    oof = pd.read_csv(base / "diagnostics" / "oof_training.csv")
    oof["pred"] = oof[[f"p_{t}" for t in TASKS]].to_numpy().argmax(1)
    oof["pred"] = oof.pred.map(dict(enumerate(TASKS)))
    oof["part"] = np.where(oof.role == "silver_train", "natural", "transformed")
    rows = []
    for (system, cond, part), g in oof.groupby(["system", "condition", "part"]):
        for t in TASKS:
            tp = int(((g.label == t) & (g.pred == t)).sum())
            rows.append({"system": system, "condition": cond, "part": part, "task": t,
                         "n_label": int((g.label == t).sum()), "n_families": g[g.label == t].family.nunique(),
                         "n_pred": int((g.pred == t).sum()), "recall": tp / max(int((g.label == t).sum()), 1),
                         "precision": tp / int((g.pred == t).sum()) if (g.pred == t).any() else np.nan})
    per_class = pd.DataFrame(rows)
    conf = (oof[oof.part == "natural"].groupby(["system", "condition", "label", "pred"]).size()
            .rename("n").reset_index())
    report = json.loads((base / "report.json").read_text())
    fid = pd.DataFrame([{"system": k, **v} for k, v in report["fidelity"].items()]).assign(**report["versions"])
    return per_class, conf, fid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst, long = load()
    top, paired = top1_tables(long)
    top.to_csv(OUT / "top1_family.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    thr, norm_long, summ, ask, costly, gains, plug, er, sign = regret_analyses(inst)
    er.to_csv(OUT / "er_scores.csv", index=False)
    sign.to_csv(OUT / "regret_sign.csv", index=False)
    thr.to_csv(OUT / "regret_thresholds.csv", index=False)
    norm_long.to_csv(OUT / "regret_normalised_long.csv", index=False)
    summ.to_csv(OUT / "regret_normalised.csv", index=False)
    ask.to_csv(OUT / "ask_robust.csv", index=False)
    costly.to_csv(OUT / "costly_confusions.csv", index=False)
    gains.to_csv(OUT / "constant_baseline.csv", index=False)
    plug.to_csv(OUT / "ask_plugin.csv", index=False)
    choice, fcounts = formulation_selection(inst)
    choice.to_csv(OUT / "formulation_choice.csv", index=False)
    fcounts.to_csv(OUT / "formulation_counts.csv", index=False)
    grid = ask_grid(er, ask)
    grid.to_csv(OUT / "ask_grid.csv", index=False)
    _, proxy_fid = load_proxy()
    if proxy_fid is not None:
        proxy_fid.to_csv(OUT / "proxy_fidelity.csv", index=False)
    flips = documentation_flips(long)
    flips.to_csv(OUT / "documentation_flips.csv", index=False)
    cnt_t = count_targets(long)
    cnt_t.to_csv(OUT / "count_targets.csv", index=False)
    rq1, docsys = documentation(inst, long)
    rq1.to_csv(OUT / "documentation.csv", index=False)
    docsys.to_csv(OUT / "documentation_systems.csv", index=False)
    ordt, cnt = ordinal_and_count(long)
    ordt.to_csv(OUT / "ordinal_subtype.csv", index=False)
    cnt.to_csv(OUT / "count_answers.csv", index=False)
    rob = robustness()
    if rob is not None:
        rob[0].to_csv(OUT / "robustness_oof_per_class.csv", index=False)
        rob[1].to_csv(OUT / "robustness_oof_confusion.csv", index=False)
        rob[2].to_csv(OUT / "robustness_fidelity.csv", index=False)

    pd.set_option("display.width", 200)
    key = top[top.system.isin(MAIN)].round(3)
    print("== top-1, family bootstrap and without mfeat ==\n", key.to_string(index=False))
    print("\n== paired differences (TIA-v1) ==\n",
          paired[paired.system == "method:forced"].round(3).to_string(index=False))
    print("\n== share relative regret > tau (always act), C0 / C2 ==")
    t = thr[thr.system.isin(COSTLY_SYS) & thr.condition.isin(["C0", "C2"])]
    print(t.pivot_table(index=["eval", "system", "condition"], columns="tau", values="share_rel").round(2).to_string())
    print("\n== normalised regret summary ==\n",
          summ[summ.system.isin(COSTLY_SYS)].round(3).to_string(index=False))
    a = ask[(ask.k == 6) & ask.measure.isin(["rel", "norm"])].pivot_table(
        index=["decide", "eval", "measure", "tau", "condition"], columns="rule", values="costly_left")
    print("\n== costly targets left after 6 questions (TIA-v2) ==\n", a.to_string())
    print("\n== documentation ==\n", rq1.to_string(index=False), "\n", docsys.round(3).to_string(index=False))
    print("\n== ordinal subtype ==\n",
          ordt[ordt.system.isin(MAIN)].pivot_table(index=["system", "subtype"], columns="condition",
                                                   values="correct").to_string())
    print("\n== constant baseline: smallest gains ==\n", gains[gains["eval"] == "hgb"].nsmallest(6, "gain").round(4).to_string(index=False))
    print("\n== ER on normalised estimates, costly (normalised > 10%) left after 6 questions (TIA-v2) ==\n",
          ask[(ask.k == 6) & (ask.measure == "norm")].pivot_table(index=["decide", "eval", "condition"], columns="rule",
                                                                  values="costly_left").to_string())
    pk = plug[(plug.k.isin([0, 6, 11])) & (plug.measure == "rel") & np.isclose(plug.tau, 0.10)]
    print("\n== Equation 3 with other probabilities: expected costly (relative > 10%) left ==\n",
          pk.pivot_table(index=["decide", "eval", "system", "condition"], columns=["k", "rule"],
                         values="costly_left").round(2).to_string())
    if proxy_fid is not None:
        d = proxy_fid[[c for c in proxy_fid.columns if c.startswith("diff_")]]
        print(f"\n== B2 proxy: fidelity {int((d.fillna(0).max(axis=1) < 1e-9).sum())}/{len(d)} identical, "
              f"largest difference {d.max().max():.2g} ==")
        b2 = ask[(ask.k == 6) & ask["eval"].isin(["ag", "ag_argmax"]) & (
            (ask.measure == "rel") & np.isclose(ask.tau, 0.10) | (ask.measure == "norm") & np.isclose(ask.tau, 0.10))]
        print(b2.pivot_table(index=["decide", "eval", "measure", "condition"], columns="rule",
                             values="costly_left").to_string())
    c16 = ask[ask.k.isin([0, 6]) & np.isclose(ask.tau, 0.10) & ask.rule.isin(
        ["confidence", "expected_regret", "expected_regret_clip", "expected_regret_norm", "expected_regret_norm_clip"])]
    print("\n== 16: clipped regret, costly (> 10%) left after 0 / 6 questions (TIA-v2) ==\n",
          c16.pivot_table(index=["decide", "eval", "measure", "condition"], columns=["k", "rule"],
                          values="costly_left").to_string())
    print(f"\n== 18: CONTINUOUS formulation choice, {len(choice)} target-utility pairs; differs from `eval` ==\n",
          {s: int((choice[f"arm_{s}"] != choice.arm_eval).sum()) for s in ("decision", "fixed")})
    f18 = fcounts[fcounts.system.isin(COSTLY_SYS) & ((fcounts.policy == "always act") | fcounts.k.isin([6]))]
    print(f18.pivot_table(index=["policy", "system", "decide", "rule", "condition"], columns="select",
                          values="costly").to_string())
    w = grid.pivot_table(index=["eval", "decide", "condition", "tau"], columns="rule", values="costly_left").reset_index()
    print("\n== 19: thresholds (5-50%) where ER leaves more / fewer costly choices than confidence at 6 questions ==")
    for key, g in w.groupby(["eval", "decide", "condition"]):
        more, fewer = g[g.expected_regret > g.confidence].tau, g[g.expected_regret < g.confidence].tau
        print(key, "more:", f"{more.min():.3f}-{more.max():.3f}" if len(more) else "never",
              "| fewer:", f"{len(fewer)}/{len(g)}, {fewer.min():.3f}-{fewer.max():.3f}" if len(fewer) else "never")
    print("\n== documentation flips ==\n", flips.drop(columns=["title"]).to_string(index=False))
    print("\n== COUNT targets ==\n", cnt_t.to_string(index=False))
    if rob is not None:
        print("\n== robustness run ==\n", rob[2].to_string(index=False))
        print(rob[0][(rob[0].task == "COUNT_REGRESSION")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
