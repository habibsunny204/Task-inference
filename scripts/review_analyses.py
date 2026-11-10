"""Post-hoc analyses requested by the Frontiers in AI pre-submission review (declared in docs/findings_log.md on 2026-11-09,
before any number was computed). Frozen outputs only: nothing is fitted, inferred or downloaded.

    .venv/Scripts/python scripts/review_analyses.py

Writes to results/sensitivity/:
1. ask_uncertainty.csv     paired bootstrap of (costly choices left by the confidence rule - by the ER rule) at 6, 11, 17
                           questions, of the mean over k = 0..17 of the costly-choice curve, and of the mean normalised regret
                           at 6 questions. Both rules are re-applied in every resample. Resampling unit: target or family
                           (k scaled with the number of targets drawn).
2. ask_per_target.csv      every target that is costly when acting: ranks under both rules, who asks at 6 questions.
3. ask_rotation.csv        the ten assignments of 3 decision and 2 evaluation folds (HGB, shared fits, which include the
                           evaluation rows). A sensitivity check, not a nested estimate.
4. ask_bayes.csv           the Bayes action a_B = argmin_a sum_t p(t) R_hat(a | t) and its expected regret, next to a*;
                           includes TIA-v1 and TIA-v2 (item 5 of the declaration) for k = 0..28.
   ask_bayes_moves.csv     the targets on which a_B differs from a*.
5. interval_metrics.csv    95% bootstrap intervals over targets for AUROC, Brier, exact set match and Jaccard.
   brier_paired.csv        paired Brier differences TIA-v2 minus every other system.
6. favourite_weights.csv   regret over a set-valued reference with a named favourite weighted at 2/3.
7. noisy_user.csv          imperfect users: favourite-answering, and wrong answers with probability p.
8. decision_fit_time.csv   wall-clock seconds of the nested decision fits per target.
9. regret_sign_fitted.csv  off-diagonal regret entries, separating fitted formulations from filled ones.
   formulations.csv        the formulations each of the 56 targets admits.
"""
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import policy_regret as pr  # noqa: E402
import sensitivity_analyses as sa  # noqa: E402

OUT = ROOT / "results" / "sensitivity"
TASKS, CONDS = pr.TASKS, ["C0", "C1", "C2"]
EVALS = ["hgb", "ag", "ag_argmax"]
SYSTEMS = {"v2:forced": ROOT / "results/systems/tia_v2/predictions.csv",
           "method:forced": ROOT / "results/systems/tia_v1/predictions.csv"}
B = 4000
KS = (6, 11, 17)
K_AUC = 17
N = 56
TAU = 0.10


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def probabilities(system):
    d = pd.read_csv(SYSTEMS[system])
    d = d[(d.system == system) & d.instance_id.str.startswith("I")]
    assert d.groupby("condition").size().eq(N).all(), system
    return d


def bayes_action(p, D):
    """argmin over the tasks with a formulation of the expected signed regret; ties go to the most probable task."""
    fitted = [a for a in TASKS if a in D.index and D.loc[a].notna().any() and (D[a].notna().any())]
    if not fitted:
        return None, 0.0
    value = {a: sa.expected_regret(p, a, D) for a in fitted}
    best = min(value.values())
    tied = [a for a in fitted if abs(value[a] - best) < 1e-12]
    return max(tied, key=lambda a: p[TASKS.index(a)]), best


def setup():
    inst, _ = sa.load()
    ref = inst.set_index("instance_id")
    Ld = pr.decision_tables("nested")
    D = {i: pr.regret_matrix(L) for i, L in Ld.items()}
    L0 = sa.constant_losses(set(D))
    real = {}
    for ev in EVALS:
        Le = pr.eval_tables(ev)
        real[ev] = {"rel": {i: pr.regret_matrix(L) for i, L in Le.items()},
                    "norm": {i: sa.regret_matrix_norm(L, L0[i]) for i, L in Le.items()}}
    return inst, ref, Ld, D, real


def frame(g, ref, D, real_ev):
    """One row per target for one system, evaluation and condition; targets without fits have ER 0 and regret 0."""
    rows = []
    for r in g.itertuples():
        p = np.array([getattr(r, f"p_{t}") for t in TASKS], float)
        a = TASKS[int(p.argmax())]
        rec = {"instance_id": r.instance_id, "conf": p.max(), "a_star": a, "er": 0.0, "act_rel": 0.0, "act_norm": 0.0,
               "a_bayes": a, "er_bayes": 0.0, "bayes_rel": 0.0, "bayes_norm": 0.0}
        if r.instance_id in D and r.instance_id in real_ev["rel"]:
            Di = D[r.instance_id]
            refset = ref.at[r.instance_id, "set_C2"].split("|")
            ab, ev_b = bayes_action(p, Di)
            rec.update({"er": sa.expected_regret(p, a, Di), "a_bayes": ab, "er_bayes": ev_b})
            for m in ("rel", "norm"):
                for name, act in (("act", a), ("bayes", ab)):
                    v = pr.intended_regret(real_ev[m][r.instance_id], refset, act)
                    rec[f"{name}_{m}"] = 0.0 if pd.isna(v) else v
        rows.append(rec)
    return pd.DataFrame(rows)


def left_curve(order, costly, kmax):
    """Costly targets left after asking the first k of `order`, for k = 0..kmax."""
    c = np.asarray(costly, bool)[order]
    return c.sum() - np.r_[0, c.cumsum()][:kmax + 1]


def orders(d):
    return {"confidence": d.conf.to_numpy().argsort(kind="stable"),
            "expected_regret": (-d.er.to_numpy()).argsort(kind="stable"),
            "bayes_er": (-d.er_bayes.to_numpy()).argsort(kind="stable")}


# ---------------------------------------------------------------------------------------------- 1, 4, 5: rules
def rule_tables(inst, ref, D, real):
    fam = inst.set_index("instance_id").family
    curves, unc, per, moves = [], [], [], []
    for system in SYSTEMS:
        pred = probabilities(system)
        for ev in EVALS:
            for cond, g in pred.groupby("condition", sort=False):
                d = frame(g, ref, D, real[ev])
                d["family"] = d.instance_id.map(fam)
                o = orders(d)
                if ev == "hgb":  # the Bayes action does not depend on the evaluation
                    moves += [{"system": system, "condition": cond, "instance_id": r.instance_id, "a_star": r.a_star,
                               "a_bayes": r.a_bayes, "er": r.er, "er_bayes": r.er_bayes} for r in d.itertuples()
                              if r.a_star != r.a_bayes]
                for meas in ("rel", "norm"):
                    a_cost = (d[f"act_{meas}"] > TAU).to_numpy()
                    b_cost = (d[f"bayes_{meas}"] > TAU).to_numpy()
                    policies = {"act_conf": left_curve(o["confidence"], a_cost, 28),
                                "act_er": left_curve(o["expected_regret"], a_cost, 28),
                                "bayes_er": left_curve(o["bayes_er"], b_cost, 28),
                                "bayes_conf": left_curve(o["confidence"], b_cost, 28)}
                    for pol, curve in policies.items():
                        curves += [{"system": system, "eval": ev, "condition": cond, "measure": meas, "policy": pol,
                                    "k": k, "costly_left": int(v)} for k, v in enumerate(curve)]
                if system == "v2:forced":
                    per += per_target(d, ev, cond, ref, o)
                unc += bootstrap(d, system, ev, cond)
    return pd.DataFrame(curves), pd.DataFrame(unc), pd.DataFrame(per), pd.DataFrame(moves)


def per_target(d, ev, cond, ref, o):
    rk_c, rk_e = np.empty(N, int), np.empty(N, int)
    rk_c[o["confidence"]] = np.arange(1, N + 1)
    rk_e[o["expected_regret"]] = np.arange(1, N + 1)
    rows = []
    for i, r in enumerate(d.itertuples()):
        if r.act_rel > TAU:
            rows.append({"system": "v2:forced", "eval": ev, "condition": cond, "instance_id": r.instance_id,
                         "dataset": ref.at[r.instance_id, "dataset_name"], "target": ref.at[r.instance_id, "target_column"],
                         "set_C2": ref.at[r.instance_id, "set_C2"], "chosen": r.a_star, "conf": r.conf, "er": r.er,
                         "regret_rel": r.act_rel, "regret_norm": r.act_norm, "conf_rank": int(rk_c[i]), "er_rank": int(rk_e[i]),
                         "asked_conf_k6": bool(rk_c[i] <= 6), "asked_er_k6": bool(rk_e[i] <= 6)})
    return rows


def bootstrap(d, system, ev, cond):
    """Paired bootstrap of the difference confidence - ER. Positive: the ER rule leaves fewer."""
    conf, er = d.conf.to_numpy(), d.er.to_numpy()
    cost = (d.act_rel > TAU).to_numpy()
    nreg = d.act_norm.to_numpy()
    fams = d.family.to_numpy()
    uniq = np.unique(fams)
    members = {f: np.flatnonzero(fams == f) for f in uniq}
    rows = []
    for unit in ("target", "family"):
        rng = rng_for(system, ev, cond, unit)
        res = {f"d{k}": np.zeros(B) for k in KS}
        res["auc"], res["norm6"] = np.zeros(B), np.zeros(B)
        for b in range(B):
            if unit == "target":
                idx = rng.integers(0, N, N)
            else:
                idx = np.concatenate([members[f] for f in rng.choice(uniq, len(uniq))])
            scale = len(idx) / N
            oc, oe = conf[idx].argsort(kind="stable"), (-er[idx]).argsort(kind="stable")
            kmax = max(int(round(K_AUC * scale)), 1)
            lc, le = left_curve(oc, cost[idx], kmax), left_curve(oe, cost[idx], kmax)
            for k in KS:
                j = min(int(round(k * scale)), kmax)
                res[f"d{k}"][b] = lc[j] - le[j]
            res["auc"][b] = lc.mean() - le.mean()
            k6 = int(round(6 * scale))
            nc, ne = nreg[idx].copy(), nreg[idx].copy()
            nc[oc[:k6]], ne[oe[:k6]] = 0.0, 0.0
            res["norm6"][b] = nc.mean() - ne.mean()
        o_c, o_e = conf.argsort(kind="stable"), (-er).argsort(kind="stable")
        c_full, e_full = left_curve(o_c, cost, K_AUC), left_curve(o_e, cost, K_AUC)
        for stat, values in res.items():
            if stat.startswith("d"):
                k = int(stat[1:])
                point = c_full[k] - e_full[k]
            elif stat == "auc":
                point = c_full.mean() - e_full.mean()
            else:
                nc, ne = nreg.copy(), nreg.copy()
                nc[o_c[:6]], ne[o_e[:6]] = 0.0, 0.0
                point = nc.mean() - ne.mean()
            rows.append({"system": system, "eval": ev, "condition": cond, "unit": unit, "stat": stat, "point": float(point),
                         "ci_lo": float(np.percentile(values, 2.5)), "ci_hi": float(np.percentile(values, 97.5)),
                         "share_er_better": float((values > 0).mean()), "share_tie": float((values == 0).mean()),
                         "share_conf_better": float((values < 0).mean())})
    return rows


# ---------------------------------------------------------------------------------------------- 3: fold rotation
def rotation(ref):
    rows = []
    for dec in itertools.combinations(range(5), 3):
        evf = tuple(sorted(set(range(5)) - set(dec)))
        dm_runs = pr.fold_runs(list(dec))
        dm = dm_runs.groupby(["instance_id", "task", "arm"])[list(pr.HIGHER)].mean()

        def choose(iid, task, u, arms):
            loss = dm.loc[(iid, task), u].reindex(arms)
            loss = 1 - loss if pr.HIGHER[u] else loss
            best = list(loss.index[loss == loss.min()])
            return pr.FIXED_ARM[u] if not best or pr.FIXED_ARM[u] in best else sorted(best)[0]

        Rd = {i: pr.regret_matrix(L) for i, L in pr.tables_from_runs(dm_runs).items()}
        Rs = {i: pr.regret_matrix(L) for i, L in pr.tables_from_runs(pr.fold_runs(list(evf)), choose).items()}
        for system in SYSTEMS:
            for cond, g in probabilities(system).groupby("condition", sort=False):
                d = frame(g, ref, Rd, {"rel": Rs, "norm": Rs})
                o = orders(d)
                a_cost, b_cost = (d.act_rel > TAU).to_numpy(), (d.bayes_rel > TAU).to_numpy()
                cc, ce = left_curve(o["confidence"], a_cost, 17), left_curve(o["expected_regret"], a_cost, 17)
                cb = left_curve(o["bayes_er"], b_cost, 17)
                for k in (0, 6, 11, 17):
                    rows.append({"decision_folds": "+".join(map(str, dec)), "eval_folds": "+".join(map(str, evf)),
                                 "system": system, "condition": cond, "k": k, "conf": int(cc[k]), "er": int(ce[k]),
                                 "bayes_er": int(cb[k])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 5: metric intervals
def metric_intervals():
    long = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    long = long[long.group == "pilot"]
    sysnames = ["sortinghat:forced", "llm:microsoft_Phi_3_5_mini_instruct", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4", "ml:logreg",
                "ml:rf", "method:forced", "v2:forced", "heuristic_set", "heuristic_names", "sortinghat",
                "method:conformal_a10", "v2:conformal_a05", "v2:ask10", "v2:ask20", "v2:ask30"]
    rows, paired = [], []
    for (system, cond), g in long[long.system.isin(sysnames)].groupby(["system", "condition"]):
        g = g.drop_duplicates("instance_id").set_index("instance_id")
        rng = rng_for("interval", system, cond)
        idx = rng.integers(0, len(g), (B, len(g)))
        for metric, col in (("exact_set", "exact_set"), ("jaccard", "jaccard")):
            v = g[col].astype(float).to_numpy()
            b = v[idx].mean(1)
            rows.append({"system": system, "condition": cond, "metric": metric, "n": len(v), "point": v.mean(),
                         "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5)})
        if g.conf.notna().any():
            amb, score = g.ref_ambiguous.astype(bool).to_numpy(), (1 - g.conf).fillna(1).to_numpy()
            bs = np.array([sa.auroc(score[i], amb[i]) for i in idx[:2000]])
            rows.append({"system": system, "condition": cond, "metric": "auroc", "n": len(g),
                         "point": sa.auroc(score, amb), "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5)})
            s = g[g.single.astype(bool) & g.brier.notna()]
            if len(s):
                v = s.brier.to_numpy()
                b = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
                rows.append({"system": system, "condition": cond, "metric": "brier", "n": len(v), "point": v.mean(),
                             "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5)})
    for cond in CONDS:
        piv = long[(long.condition == cond) & long.single.astype(bool)].pivot_table(index="instance_id", columns="system",
                                                                                    values="brier", aggfunc="first")
        for other in ("llm:microsoft_Phi_3_5_mini_instruct", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4",
                      "ml:rf", "ml:logreg"):
            d = (piv["v2:forced"] - piv[other]).dropna().to_numpy()
            rng = rng_for("brierdiff", other, cond)
            b = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            paired.append({"condition": cond, "other": other, "n": len(d), "diff_v2_minus_other": d.mean(),
                           "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(paired)


# ---------------------------------------------------------------------------------------------- 6-7: intended tasks and users
def weighted_regret(R, refset, fav, a):
    ts = [t for t in refset if R[t].notna().any()]
    if not ts:
        return np.nan
    if fav in ts and len(ts) > 1:
        w = np.array([2 / 3 if t == fav else (1 / 3) / (len(ts) - 1) for t in ts])
    else:
        w = np.ones(len(ts)) / len(ts)
    return float(sum(wi * R.at[a, t] for wi, t in zip(w, ts)))


def favourite_weights(ref, D, real):
    fav = ref.most_likely_C2
    sc = pd.read_csv(ROOT / "results" / "scores" / "reference_scores_long.csv")
    sc = sc[(sc.group == "pilot") & sc.system.isin(sa.MAIN)]
    rows = []
    for ev in ("hgb", "ag"):
        Rs = real[ev]["rel"]
        for (system, cond), g in sc[sc.instance_id.isin(Rs)].groupby(["system", "condition"]):
            for name, fn in (("uniform", lambda R, i, a: pr.intended_regret(R, ref.at[i, "set_C2"].split("|"), a)),
                             ("favourite_2_3", lambda R, i, a: weighted_regret(R, ref.at[i, "set_C2"].split("|"), fav[i], a))):
                v = [fn(Rs[r.instance_id], r.instance_id, r.most_likely_task) if r.most_likely_task in TASKS
                     else np.nanmax(Rs[r.instance_id].to_numpy()) for r in g.itertuples()]
                rows.append({"eval": ev, "system": system, "condition": cond, "weights": name, "policy": "always act", "k": 0,
                             "n": len(g), "costly": int((np.array(v) > TAU).sum())})
        for cond, g in probabilities("v2:forced").groupby("condition", sort=False):
            d = frame(g, ref, D, real[ev])
            acts = {"uniform": d.act_rel.to_numpy()}
            w = []
            for r in d.itertuples():
                if r.instance_id in Rs:
                    v = weighted_regret(Rs[r.instance_id], ref.at[r.instance_id, "set_C2"].split("|"), fav[r.instance_id], r.a_star)
                    w.append(0.0 if pd.isna(v) else v)
                else:
                    w.append(0.0)
            acts["favourite_2_3"] = np.array(w)
            o = orders(d)
            for name, act in acts.items():
                for rule in ("confidence", "expected_regret"):
                    rows.append({"eval": ev, "system": "v2:forced", "condition": cond, "weights": name, "policy": rule, "k": 6,
                                 "n": N, "costly": int(left_curve(o[rule], act > TAU, 6)[6])})
                rows.append({"eval": ev, "system": "v2:forced", "condition": cond, "weights": name, "policy": "never ask", "k": 0,
                             "n": N, "costly": int((act > TAU).sum())})
    return pd.DataFrame(rows)


def noisy_users(ref, D, real):
    rows = []
    for ev in ("hgb", "ag"):
        Rs = real[ev]["rel"]
        for cond, g in probabilities("v2:forced").groupby("condition", sort=False):
            d = frame(g, ref, D, real[ev])
            o = orders(d)
            q, favc = [], []
            for r in d.itertuples():
                qi, fi = 0.0, 0.0
                if r.instance_id in Rs:
                    R, s = Rs[r.instance_id], ref.at[r.instance_id, "set_C2"].split("|")
                    fitted = [a for a in TASKS if R.loc[a].notna().any() and R[a].notna().any()]
                    ts = [t for t in s if R[t].notna().any()]
                    vals = [float(R.at[b, t] > TAU) for t in ts for b in fitted if b != t]
                    qi = float(np.mean(vals)) if vals else 0.0
                    fav = ref.at[r.instance_id, "most_likely_C2"]
                    if fav in ts and len(ts) > 1:
                        fi = float(pr.intended_regret(R, s, fav) > TAU)
                q.append(qi)
                favc.append(fi)
            q, favc = np.array(q), np.array(favc)
            act = (d.act_rel > TAU).to_numpy()
            for rule in ("confidence", "expected_regret"):
                asked = np.zeros(N, bool)
                asked[o[rule][:6]] = True
                base = float((act & ~asked).sum())
                for p in (0.0, 0.1, 0.2, 0.3):
                    rows.append({"eval": ev, "condition": cond, "rule": rule, "user": f"wrong_p{p:.1f}", "k": 6,
                                 "costly_left": base + p * q[asked].sum()})
                rows.append({"eval": ev, "condition": cond, "rule": rule, "user": "favourite", "k": 6,
                             "costly_left": base + favc[asked].sum()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- 8-9: cost and counts
def fit_times():
    runs = pr.nested_runs()
    r = runs.drop_duplicates(["instance_id", "arm", "fold"])
    per = r.groupby("instance_id").agg(n_formulations=("arm", "nunique"), n_train=("n_train", "max"),
                                       decision_fit_seconds=("fit_seconds", "sum")).reset_index()
    ag = pd.read_csv(pr.AG_RUNS)
    ag = ag[(ag.group == "pilot")].groupby("instance_id").agg(ag_fits=("fit_seconds", "size"),
                                                              ag_seconds_per_fit=("fit_seconds", "mean")).reset_index()
    return per.merge(ag, on="instance_id", how="left")


def signed_counts(inst, Ld, D):
    rows, forms = [], []
    n_fit, n_neg, n_fill, n_fill_neg = 0, 0, 0, 0
    for i, R in D.items():
        L = Ld[i]
        for a in TASKS:
            for t in TASKS:
                if a == t or pd.isna(R.at[a, t]):
                    continue
                fitted = a in L.index and pd.notna(L.at[a, pr.OWN[t]])
                if fitted:
                    n_fit += 1
                    n_neg += int(R.at[a, t] < 0)
                else:
                    n_fill += 1
                    n_fill_neg += int(R.at[a, t] < 0)
    rows.append({"tables": "decision (nested)", "n_entries_fitted": n_fit, "n_negative_fitted": n_neg,
                 "n_entries_filled": n_fill, "n_negative_filled": n_fill_neg})
    ref = inst.set_index("instance_id")
    for i in inst.instance_id:
        forms.append({"instance_id": i, "dataset": ref.at[i, "dataset_name"], "target": ref.at[i, "target_column"],
                      "set_C2": ref.at[i, "set_C2"],
                      "formulations": "|".join(t for t in TASKS if i in Ld and t in Ld[i].index),
                      "multi_formulation": i in Ld})
    return pd.DataFrame(rows), pd.DataFrame(forms)


def main():
    inst, ref, Ld, D, real = setup()
    curves, unc, per, moves = rule_tables(inst, ref, D, real)
    moves.to_csv(OUT / "ask_bayes_moves.csv", index=False)
    # item 1 must reproduce the frozen ask table for TIA-v2
    frozen = pd.read_csv(OUT / "ask_robust.csv")
    frozen = frozen[(frozen.decide == "nested") & (frozen.measure == "rel") & np.isclose(frozen.tau, 0.10) & frozen.k.isin([0, 6, 11, 17])
                    & frozen.rule.isin(["confidence", "expected_regret"])]
    pol = {"confidence": "act_conf", "expected_regret": "act_er"}
    for r in frozen.itertuples():
        got = curves[(curves.system == "v2:forced") & (curves["eval"] == r.eval) & (curves.condition == r.condition)
                     & (curves.measure == "rel") & (curves.policy == pol[r.rule]) & (curves.k == r.k)].costly_left.iloc[0]
        assert got == r.costly_left, ("frozen ask table not reproduced", r)
    curves.to_csv(OUT / "ask_bayes.csv", index=False)
    unc.to_csv(OUT / "ask_uncertainty.csv", index=False)
    per.to_csv(OUT / "ask_per_target.csv", index=False)
    rotation(ref).to_csv(OUT / "ask_rotation.csv", index=False)
    im, bp = metric_intervals()
    im.to_csv(OUT / "interval_metrics.csv", index=False)
    bp.to_csv(OUT / "brier_paired.csv", index=False)
    favourite_weights(ref, D, real).to_csv(OUT / "favourite_weights.csv", index=False)
    noisy_users(ref, D, real).to_csv(OUT / "noisy_user.csv", index=False)
    fit_times().to_csv(OUT / "decision_fit_time.csv", index=False)
    sign, forms = signed_counts(inst, Ld, D)
    sign.to_csv(OUT / "regret_sign_fitted.csv", index=False)
    forms.to_csv(OUT / "formulations.csv", index=False)
    print("frozen ask table reproduced; files written to", OUT)


if __name__ == "__main__":
    main()
