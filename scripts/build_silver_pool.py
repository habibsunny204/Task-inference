"""Silver labels for the non-gold candidate pool, from source metadata only (no annotator, no LLM).

Rules are fixed in benchmark/silver/RULES.md before any check against gold. This script only reads
benchmark/candidate_pool.csv (metadata, seconds). Tables are fetched later in the Colab data bundle.

    .venv/Scripts/python scripts/build_silver_pool.py
writes benchmark/silver/pool_silver.csv     every pool candidate: family, silver label or exclusion reason
       benchmark/silver/silver_vs_gold.md   the same rules applied to the 56 gold candidates vs the frozen reference
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "silver"

# Same underlying source table or source collection, matched by dataset identity (not by task).
FAMILY = {
    "abalone": "abalone", "car": "car_evaluation", "eucalyptus": "eucalyptus",
    "wine-quality-white": "wine_quality", "winequalityred": "wine_quality", "white_wine": "wine_quality",
    "red_wine": "wine_quality", "wine_quality": "wine_quality",
    "concretestrength": "concrete", "concrete_compressive_strength": "concrete",
    "housing": "boston_housing", "boston": "boston_housing",
    "studentperformance": "student_performance", "student_performance_por": "student_performance",
    "california_housing": "california_housing", "calhousing": "california_housing",
    "kc1": "nasa_defect", "kc2": "nasa_defect", "pc1": "nasa_defect", "pc3": "nasa_defect",
    "pc4": "nasa_defect", "jm1": "nasa_defect",
    "ozone-level-8hr": "ozone", "ozone_level": "ozone",
    "cpu_activity": "comp_activ", "computer1": "comp_activ", "computer2": "comp_activ",
    "kings_county": "kc_house", "house_sales": "kc_house",
    "brazilian_houses": "brazilian_houses", "satellite": "satimage", "satimage": "satimage",
    "balance-scale": "balance_scale", "balancescale": "balance_scale",
    "forest_fires": "forest_fires", "forestfires": "forest_fires",
    "australian": "credit_australian", "credit-approval": "credit_australian",
    "census1": "census", "census2": "census",
}
PREFIX_FAMILY = [("mfeat-", "mfeat"), ("buoys", "buoys"), ("contaminant-detection", "contaminant_detection")]
# Keep one row per duplicated (family, target): the curated regression suite over AMLB.
ORIGIN_PRIORITY = ["OpenML-CTR23", "AMLB-regression+OpenML-CTR23", "OpenML-CC18", "AMLB-classification+OpenML-CC18",
                   "AMLB-regression", "AMLB-classification", "TOC-UCO (originally-ordinal)",
                   "TOC-UCO (discretised-regression)", "OpenML-multitype", "OpenML-CC18+OpenML-multitype"]


def family(name):
    n = str(name).strip().lower()
    for pre, fam in PREFIX_FAMILY:
        if n.startswith(pre):
            return fam
    return FAMILY.get(n, re.sub(r"[^a-z0-9]+", "_", n).strip("_"))


def silver_rule(r):
    """(label, rule id) from the source metadata, or (None, reason)."""
    o = r.origin
    if "multitype" in o:
        return None, "R5 registered as both classification and regression (F1): no source label"
    if o.startswith("TOC-UCO (originally-ordinal)"):
        return "ORDINAL", "R3 TOC-UCO originally-ordinal repository"
    if o.startswith("TOC-UCO (discretised-regression)"):
        return "ORDINAL", "R4 TOC-UCO discretised regression: ordered interval index by construction"
    if "CTR23" in o or "regression" in o:
        return "CONTINUOUS_REGRESSION", "R2 curated regression suite (OpenML-CTR23 / AMLB regression)"
    if "CC18" in o or "classification" in o:
        if pd.isna(r.n_unique_target):
            return None, "R1 class count unknown"
        k = int(r.n_unique_target)
        return ("BINARY" if k == 2 else "MULTICLASS"), f"R1 curated classification suite, {k} classes"
    return None, f"no rule for origin {o}"


def main():
    pool = pd.read_csv(ROOT / "benchmark" / "candidate_pool.csv")
    key = pd.read_csv(ROOT / "benchmark" / "annotation_key.csv")
    pool["family"] = pool.dataset_name.map(family)
    pool["gold"] = pool.candidate_id.isin(key.candidate_id)
    lab = pool.apply(silver_rule, axis=1, result_type="expand")
    pool["silver_label"], pool["rule"] = lab[0], lab[1]
    gold_fams = set(pool[pool.gold].family)

    pool["status"] = "silver"
    pool.loc[pool.gold, "status"] = "gold"
    pool.loc[~pool.gold & pool.skip_reason.notna(), "status"] = "excluded: not loadable (" + pool.skip_reason.astype(str) + ")"
    pool.loc[~pool.gold & pool.local_file.isna() & pool.skip_reason.isna(), "status"] = "excluded: no table in pool"
    pool.loc[(pool.status == "silver") & pool.family.isin(gold_fams), "status"] = "excluded: same family as a gold table"
    pool.loc[(pool.status == "silver") & pool.silver_label.isna(), "status"] = "excluded: " + pool.rule.astype(str)
    s = pool[pool.status == "silver"].copy()
    s["prio"] = s.origin.map({o: i for i, o in enumerate(ORIGIN_PRIORITY)})
    dup = s.sort_values("prio").duplicated(["family", "target_column"], keep="first")
    pool.loc[dup[dup].index, "status"] = "excluded: duplicate of a kept table"

    cols = ["candidate_id", "did", "dataset_name", "target_column", "origin", "family", "status", "silver_label", "rule",
            "n_samples", "n_features", "n_unique_target", "has_description"]
    OUT.mkdir(parents=True, exist_ok=True)
    pool[cols].sort_values(["status", "family"]).to_csv(OUT / "pool_silver.csv", index=False)

    kept = pool[pool.status == "silver"]
    print(f"silver: {len(kept)} targets in {kept.family.nunique()} families")
    print(kept.silver_label.value_counts().to_string())
    print(pool[~pool.gold].status.str.split(":").str[0:2].str.join(":").str[:60].value_counts().to_string())

    # one-off check of the frozen rules on the gold candidates
    ref = pd.read_csv(ROOT / "benchmark" / "reference_instances.csv")
    g = pool[pool.gold].merge(key[["candidate_id", "instance_id"]], on="candidate_id").merge(ref, on="instance_id")
    g = g[g.silver_label.notna()]
    g["in_set"] = [lbl in s.split("|") for lbl, s in zip(g.silver_label, g.set_C2)]
    g["exact"] = g.silver_label == g.set_C2
    single = g[g.c2_reference_kind == "SINGLE"]
    by_rule = g.groupby(g.rule.str[:2]).agg(n=("in_set", "size"), in_C2_set=("in_set", "mean"), equals_C2=("exact", "mean"))
    miss = g[~g.in_set][["instance_id", "dataset_name_x", "target_column_x", "rule", "silver_label", "set_C2"]]
    lines = ["# Silver rules vs the frozen gold reference (one-off check, rules fixed beforehand)", "",
             f"Gold candidates with a silver label: {len(g)} of 56 (multitype targets have none).", "",
             f"- silver label inside the gold C2 set: {g.in_set.mean():.2f}",
             f"- silver label equals a single-task gold reference: {single.exact.mean():.2f} (n = {len(single)})",
             f"- gold C2 reference is set-valued where silver gives one label: {(g.c2_reference_kind != 'SINGLE').mean():.2f}", "",
             by_rule.to_markdown(floatfmt=".2f"), "", "## Silver label outside the gold C2 set", "",
             miss.to_markdown(index=False), ""]
    (OUT / "silver_vs_gold.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
