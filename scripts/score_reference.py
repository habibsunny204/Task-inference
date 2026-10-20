"""Score all baselines on the 56 pilot instances against the frozen human reference (GO-2).

Metrics fixed 2026-10-05, after the reference was frozen and before any score was seen.
Unparseable LLM answers count as wrong and as not flagging ambiguity.

Intended task (vs the C2 reference), only on the 34 instances whose C2 reference is one task:
  top1        most likely task equals the reference task
  covers      reference task is inside the predicted set
Same-condition sets (vs the reference under the same condition), all 56:
  exact_set   predicted set equals the reference set
  jaccard     mean Jaccard between predicted and reference sets
  keeps_ref   reference set is a subset of the predicted set (nothing defensible dropped)
  set_size    mean predicted set size (valid answers)
Ambiguity detection (reference set under the same condition has 2+ tasks), all 56:
  flag_recall / flag_specificity   predicted set has 2+ tasks, against that
  auroc_amb   LLMs only: AUROC of 1 - max probability
  commits     share answered with a task instead of NONE (no system has a NONE output except the LLMs)
Calibration (LLMs, the 34 single-task instances, vs the C2 reference task):
  conf        mean max probability (renormalised over the 5 tasks)
  gap         conf - top1 (positive = overconfident)
  brier       multi-class Brier score against the one-hot reference task
  p_ref_set   all 56, same condition: mean probability mass on the reference set
Selective prediction (the 34 single-task instances):
  risk/coverage when committing only if the predicted set has one task (all systems with sets)
  aurc        LLMs only: area under the risk-coverage curve, ranking by max probability
By protocol ambiguity type: top1_lenient = most likely task is in the C2 reference set.
Cost: mean seconds and tokens per prompt (LLMs; rules take milliseconds).
Bootstrap 95% CIs over instances (2000 resamples) for top1 and jaccard. Each (system, condition, metric) cell has its own
seed (crc32 of its name; fix of 2026-10-19), so adding a system does not move the other systems' intervals.

    .venv/Scripts/python scripts/score_reference.py
writes results/scores/reference_scores.{csv,md}, reference_scores_long.csv, reference_confusion_C2.csv
"""
import hashlib
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scores"
SYS = ROOT / "results" / "systems"
BENCH = ROOT / "benchmark"
LABELS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]
PCOLS = [f"p_{t}" for t in LABELS]
SYSTEMS = ["autogluon", "sklearn", "heuristic_set", "heuristic_names",
           "llm:Qwen_Qwen2_5_7B_Instruct", "llm:microsoft_Phi_3_5_mini_instruct"]


def check_frozen():
    for line in (BENCH / "reference_labels.sha256").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            digest, name = line.split()
            if hashlib.sha256((BENCH / name).read_bytes()).hexdigest() != digest:
                raise SystemExit(f"{name} changed since it was frozen")


def load_predictions():
    rules = pd.read_csv(SYS / "rules" / "predictions.csv")
    rules = rules.rename(columns={"baseline": "system"}).assign(parse_ok=True)
    llm = pd.read_csv(SYS / "llm" / "predictions.csv").rename(columns={"model": "system"})
    llm["system"] = "llm:" + llm.system
    parts = [rules, llm]
    strong = SYS / "llm_strong" / "predictions.csv"
    if strong.exists():  # colab/task_inference_llm_strong (Qwen3-14B, Phi-4), same format as the first LLM baseline
        s = pd.read_csv(strong).rename(columns={"model": "system"})
        parts.append(s.assign(system="llm:" + s.system))
    for method in sorted(SYS.glob("tia_*/predictions.csv")) + sorted(SYS.glob("sortinghat/predictions.csv")):  # Colab TIA runs (v1, v2, robustness), SortingHat
        parts.append(pd.read_csv(method))
    pred = pd.concat(parts, ignore_index=True)
    return pred[pred.group == "pilot"]


def to_set(s):
    return frozenset(x for x in str(s).split("|") if x and x != "nan")


def auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def aurc(conf, correct):
    order = np.argsort(-np.asarray(conf), kind="stable")
    err = 1 - np.asarray(correct, float)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def boot_ci(values, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    v = np.asarray(values, float)
    means = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return np.percentile(means, [2.5, 97.5])


def score(pred, ref, inst):
    ref_c = ref.rename(columns={"reference_set": "ref_set", "reference_most_likely": "ref_ml"})[
        ["instance_id", "condition", "ref_set", "ref_ml"]]
    m = pred.merge(ref_c, on=["instance_id", "condition"], how="right", validate="many_to_one")
    m = m.merge(inst[["instance_id", "set_C2", "most_likely_C2", "ambiguity_type", "c2_reference_kind"]],
                on="instance_id")
    sets = m.plausible_task_set.apply(to_set)
    rs, r2 = m.ref_set.apply(to_set), m.set_C2.apply(to_set)
    ok = m.parse_ok.fillna(False).astype(bool) & (sets.str.len() > 0)
    single = m.c2_reference_kind == "SINGLE"
    ref_task = m.set_C2.where(single)
    m["valid"] = ok
    m["single"] = single
    m["top1"] = (ok & (m.most_likely_task == ref_task)).where(single)
    m["covers"] = pd.Series([o and t in s if isinstance(t, str) else np.nan
                             for o, t, s in zip(ok, ref_task, sets)], index=m.index)
    m["top1_lenient"] = ok & pd.Series([t in r for t, r in zip(m.most_likely_task, r2)], index=m.index)
    m["exact_set"] = ok & (sets == rs)
    m["jaccard"] = [len(s & r) / len(s | r) if o else 0.0 for s, r, o in zip(sets, rs, ok)]
    m["keeps_ref"] = ok & pd.Series([r <= s for r, s in zip(rs, sets)], index=m.index)
    m["set_size"] = sets.str.len().where(ok)
    m["ref_ambiguous"] = rs.str.len() > 1
    m["flag"] = ok & (sets.str.len() > 1)
    m["commits"] = ok & (m.most_likely_task != "NONE")
    m["commit_if_single"] = ok & (sets.str.len() == 1)
    p = m.reindex(columns=PCOLS).astype(float)
    p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0)
    valid_p = ok & p.notna().all(axis=1)
    m["conf"] = p.max(axis=1).where(valid_p)
    m["p_ref_set"] = pd.Series([sum(row[f"p_{t}"] for t in r) for (_, row), r in zip(p.iterrows(), rs)],
                               index=m.index).where(valid_p)
    onehot = np.array([[t == x for x in LABELS] if isinstance(t, str) else [np.nan] * 5 for t in ref_task], float)
    m["brier"] = pd.Series(((p.to_numpy() - onehot) ** 2).sum(axis=1), index=m.index).where(valid_p & single)
    return m


def summarise(m):
    rows = []
    for (sys_, cond), g in m.groupby(["system", "condition"]):
        s = g[g.single]
        is_llm = sys_.startswith("llm:")
        has_p = g.conf.notna().any()  # LLMs and the Colab method/ML systems report probabilities
        amb = g.ref_ambiguous
        sel = s[s.commit_if_single]
        conf = s.conf.fillna(0)  # an unparseable answer ranks last
        r = {"system": sys_, "condition": cond, "valid": g.valid.mean(),
             "top1": s.top1.mean(), "top1_ci": "[{:.2f}, {:.2f}]".format(*boot_ci(s.top1.astype(float), f"{sys_}|{cond}|top1")),
             "covers": s.covers.astype(float).mean(),
             "exact_set": g.exact_set.mean(), "jaccard": g.jaccard.mean(),
             "jaccard_ci": "[{:.2f}, {:.2f}]".format(*boot_ci(g.jaccard, f"{sys_}|{cond}|jaccard")),
             "keeps_ref": g.keeps_ref.mean(), "set_size": g.set_size.mean(),
             "flag_recall": g.flag[amb].mean(), "flag_specificity": 1 - g.flag[~amb].mean(),
             "commits": g.commits.mean(),
             "sel_coverage": s.commit_if_single.mean(),
             "sel_risk": 1 - sel.top1.astype(float).mean() if len(sel) else np.nan}
        if has_p:
            r.update({"auroc_amb": auroc((1 - g.conf).fillna(1), amb), "conf": s.conf.mean(),
                      "gap": s.conf.mean() - s.top1.mean(), "brier": s.brier.mean(),
                      "p_ref_set": g.p_ref_set.mean(), "aurc": aurc(conf, s.top1.astype(float))})
        if is_llm:
            r.update({"seconds": g.seconds.mean(), "tokens": (g.tokens_in + g.tokens_out).mean()})
        rows.append(r)
    out = pd.DataFrame(rows)
    order = SYSTEMS + sorted(set(out.system) - set(SYSTEMS))
    out["system"] = pd.Categorical(out.system, order, ordered=True)
    return out.sort_values(["system", "condition"]).reset_index(drop=True)


def short(name):
    return {"llm:Qwen_Qwen2_5_7B_Instruct": "Qwen2.5-7B", "llm:microsoft_Phi_3_5_mini_instruct": "Phi-3.5-mini",
            "llm:Qwen_Qwen3_14B": "Qwen3-14B", "llm:microsoft_phi_4": "Phi-4"}.get(name, name)


def main():
    check_frozen()
    ref = pd.read_csv(BENCH / "reference_labels.csv", dtype=str, keep_default_na=False)
    inst = pd.read_csv(BENCH / "reference_instances.csv", dtype=str, keep_default_na=False)
    pred = load_predictions()
    m = pd.concat([score(g, ref, inst).assign(system=s) for s, g in pred.groupby("system")], ignore_index=True)
    summ = summarise(m)
    by_type = (m.groupby(["system", "condition", "ambiguity_type"]).top1_lenient.mean().unstack(["ambiguity_type"])
               .round(2))
    c2 = m[(m.condition == "C2") & m.single]
    confusion = pd.crosstab([c2.system, c2.set_C2], c2.most_likely_task.fillna("INVALID"))
    per_task = c2.groupby(["set_C2", "system"]).top1.mean().unstack().round(2)
    per_task.insert(0, "n", c2[c2.system == SYSTEMS[0]].set_C2.value_counts())

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "reference_scores_long.csv", index=False)
    summ.to_csv(OUT / "reference_scores.csv", index=False)
    confusion.to_csv(OUT / "reference_confusion_C2.csv")

    disp = summ.assign(system=summ.system.astype(str).map(short)).set_index(["system", "condition"])
    fmt = lambda d: d.to_markdown(floatfmt=".2f")  # noqa: E731
    counts = inst.c2_reference_kind.value_counts().to_dict()
    lines = ["# Baselines vs the human reference (56 pilot instances)", "",
             f"Reference: `benchmark/reference_labels.csv` (frozen, hash checked). C2 kinds: {counts}. "
             "Single-task metrics use the 34 SINGLE instances. Unparseable LLM answers count as wrong. "
             "C0 includes 3 groups of identical packets (mfeat ×5, census ×2, abalone ×2).", "",
             "## Intended task (34 single-task instances, vs the C2 reference)", "",
             fmt(disp[["valid", "top1", "top1_ci", "covers"]]), "",
             "## Sets vs the human set under the same condition (all 56)", "",
             fmt(disp[["exact_set", "jaccard", "jaccard_ci", "keeps_ref", "set_size"]]), "",
             "## Ambiguity detection and commitment (all 56; ambiguous = human set has 2+ tasks)", "",
             fmt(disp[["flag_recall", "flag_specificity", "commits", "auroc_amb"]]), "",
             "## Calibration and selective prediction", "",
             "`sel_*`: commit only when the predicted set has one task (34 single-task instances). "
             "`aurc`: rank by max probability (lower is better).", "",
             fmt(disp[["conf", "gap", "brier", "p_ref_set", "sel_coverage", "sel_risk", "aurc"]]), "",
             "## Top-1 (lenient: most likely task in the C2 reference set) by protocol ambiguity type", "",
             fmt(by_type.rename(index=short, level=0)), "",
             "## C2 top-1 by reference task (single-task instances)", "",
             fmt(per_task.rename(columns=short)), "",
             "## Cost per prompt (LLMs, Colab T4, 4-bit)", "",
             fmt(disp[["seconds", "tokens"]].dropna()), ""]
    (OUT / "reference_scores.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
