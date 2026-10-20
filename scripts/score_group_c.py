"""Score all baselines on Group C, whose expected tasks are known by construction (no human labels).

Metrics per system and condition (unparseable LLM answers count as wrong):
  top1        most likely task is in the expected set
  top1_strict most likely task equals the expected most likely task
  covers      expected most likely task is inside the predicted set
  exact_set   predicted set equals the expected set
  jaccard     mean Jaccard between predicted and expected sets
  set_size    mean predicted set size (valid answers)
  p_expected  LLMs only: mean probability mass on the expected set

    .venv/Scripts/python scripts/score_group_c.py
writes results/scores/group_c_scores.{csv,md}
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scores"
SYS = ROOT / "results" / "systems"
LABELS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]


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
    return pd.concat(parts, ignore_index=True)


def score(pred, key):
    m = pred.merge(key, on="instance_id")
    sets = m.plausible_task_set.fillna("").str.split("|").apply(lambda s: {x for x in s if x})
    exp = m.expected_task_set.str.split("|").apply(set)
    ok = m.parse_ok.astype(bool) & (sets.str.len() > 0)
    m["top1"] = ok & m.apply(lambda r: r.most_likely_task in set(r.expected_task_set.split("|")), axis=1)
    m["top1_strict"] = ok & (m.most_likely_task == m.expected_most_likely)
    m["covers"] = ok & pd.Series([e in s for e, s in zip(m.expected_most_likely, sets)], index=m.index)
    m["exact_set"] = ok & pd.Series([s == e for s, e in zip(sets, exp)], index=m.index)
    m["jaccard"] = [len(s & e) / len(s | e) if o else 0.0 for s, e, o in zip(sets, exp, ok)]
    m["set_size"] = sets.str.len().where(ok)
    if "p_BINARY" in m:
        m["p_expected"] = [sum(r[f"p_{t}"] for t in e) if o and pd.notna(r["p_BINARY"]) else float("nan")
                           for (_, r), e, o in zip(m.iterrows(), exp, ok)]
    m["valid"] = ok
    return m


def main():
    key = pd.read_csv(ROOT / "benchmark" / "group_c" / "key.csv")[
        ["instance_id", "transformation_type", "expected_task_set", "expected_most_likely"]]
    pred = load_predictions()
    m = score(pred[pred.instance_id.str.startswith("G")], key)
    cols = ["valid", "top1", "top1_strict", "covers", "exact_set", "jaccard", "set_size", "p_expected"]
    overall = m.groupby(["system", "condition"])[cols].mean().round(2)
    by_type = (m[m.condition == "C1"].groupby(["system", "transformation_type"]).top1.mean()
               .unstack().round(2))
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "group_c_scores_long.csv", index=False)
    overall.to_csv(OUT / "group_c_scores.csv")
    lines = ["# Group C scores (20 transformed targets, expected task known by construction)", "",
             "Unparseable LLM answers count as wrong. n = 20 per cell, so these are descriptive.", "",
             "## By system and condition", "", overall.to_markdown(), "",
             "## Top-1 by transformation (C1: values + table)", "", by_type.to_markdown(), ""]
    (OUT / "group_c_scores.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
