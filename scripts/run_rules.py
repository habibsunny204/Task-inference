"""Rule baselines for task inference. Frozen 2026-09-26, before any annotation was seen.

  autogluon        replica of AutoGluon's infer_problem_type (scripts/build_candidate_pool.py)
  sklearn          sklearn.utils.multiclass.type_of_target
  heuristic_set    set-valued rule on the target values (cardinality, integer-ness, sign, zeros)
  heuristic_names  heuristic_set + keywords in the target column name (needs C1 or C2)

The first three read only the target values, so their output is the same under C0, C1 and C2.
Thresholds and keyword lists were fixed a priori and are not tuned on any labels.

    .venv/Scripts/python scripts/run_rules.py
writes results/systems/rules/predictions.csv (+ predictions.sha256)
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_candidate_pool import rule_autogluon, rule_sklearn  # noqa: E402
from instances import all_instances  # noqa: E402

OUT = ROOT / "results" / "systems" / "rules"
CONDITIONS = ("C0", "C1", "C2")
B, M, O, CNT, CONT = "BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"
AG_MAP = {"binary": B, "multiclass": M, "regression": CONT}
SK_MAP = {"binary": B, "multiclass": M, "continuous": CONT}

# heuristic_set thresholds (a priori)
FEW_LEVELS = 10       # up to this many integer levels: categorical-looking
MID_LEVELS = 30       # up to this many: ordinal scale still plausible
# heuristic_names keywords (a priori), matched against tokens of the target column name
KEYWORDS = {
    CNT: {"count", "counts", "number", "num", "rings", "flares", "frequency", "freq", "visits",
          "sessions", "times", "cases", "events", "calls"},
    O: {"quality", "grade", "score", "rating", "rank", "level", "severity", "stage", "satisfaction",
        "utility", "g3"},
    M: {"class", "type", "category", "species", "label", "kind", "method", "genre", "group"},
    CONT: {"price", "cost", "amount", "value", "weight", "area", "strength", "load", "temperature",
           "income", "wage", "salary", "fat", "medv", "lc50", "duration", "usage", "volume"},
}


def as_numeric(y):
    if is_numeric_dtype(y):
        return y.astype(float)
    v = pd.to_numeric(y, errors="coerce")
    return v.astype(float) if v.notna().all() else None


def profile(y):
    y = y.dropna()
    num = as_numeric(y)
    p = {"k": int(y.nunique()), "numeric": num is not None}
    if num is not None:
        p.update(integer=bool(np.all(np.mod(num, 1) == 0)), min=float(num.min()),
                 zero_is_mode=bool(num.mode().iloc[0] == 0) if len(num) else False)
    return p


def heuristic_set(y):
    p = profile(y)
    if p["k"] == 2:
        return {B}, B
    if not p["numeric"]:
        return {M}, M
    if not p["integer"]:
        return {CONT}, CONT
    nonneg, countlike = p["min"] >= 0, p["min"] == 0 and p["zero_is_mode"]
    if p["k"] <= FEW_LEVELS:
        s = {M, O} | ({CNT} if nonneg else set())
        return s, CNT if countlike else O
    if p["k"] <= MID_LEVELS:
        s = {O, CONT} | ({CNT} if nonneg else set())
        return s, CNT if countlike else O
    s = {CONT} | ({CNT} if nonneg else set())
    return s, CNT if countlike else CONT


def compatible(task, p):
    k = p["k"]
    if task == B:
        return k == 2
    if task in (M, O):
        return k > 2 and (not p["numeric"] or k <= (100 if task == M else MID_LEVELS))
    if task == CNT:
        return p["numeric"] and p.get("integer") and p["min"] >= 0
    return p["numeric"]


def tokens(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def heuristic_names(y, target_name):
    s, ml = heuristic_set(y)
    p = profile(y)
    hits = [t for t, words in KEYWORDS.items() if tokens(target_name) & words]
    hits = [t for t in hits if compatible(t, p)]
    if len(hits) == 1:
        return s | {hits[0]}, hits[0]
    return s, ml


def main():
    rows = []
    for inst in all_instances():
        y = inst.df[inst.target]
        ag = AG_MAP.get(rule_autogluon(y), "ERROR")
        sk_raw = rule_sklearn(y)
        sk = SK_MAP.get(sk_raw, "ERROR")
        hs, hs_ml = heuristic_set(y)
        hn, hn_ml = heuristic_names(y, inst.target)
        for c in CONDITIONS:
            out = [("autogluon", {ag}, ag, rule_autogluon(y)), ("sklearn", {sk}, sk, sk_raw),
                   ("heuristic_set", hs, hs_ml, "")]
            out.append(("heuristic_names", hn, hn_ml, "") if c != "C0" else ("heuristic_names", hs, hs_ml, "no name at C0"))
            for name, s, ml, raw in out:
                rows.append({"instance_id": inst.instance_id, "group": inst.group, "condition": c,
                             "baseline": name, "plausible_task_set": "|".join(sorted(s)),
                             "most_likely_task": ml, "raw_output": raw})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictions.csv", index=False)
    h = hashlib.sha256((OUT / "predictions.csv").read_bytes()).hexdigest()
    (OUT / "predictions.sha256").write_text(h + "  predictions.csv\n", encoding="utf-8")
    print(df.groupby(["group", "baseline", "condition"]).size().unstack().to_string())
    print("sha256", h)


if __name__ == "__main__":
    main()
