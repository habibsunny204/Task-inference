"""Summarise benchmark/candidate_pool.csv and draft a 60-instance pilot shortlist.

Rule outputs are used ONLY to find interesting candidates, never as labels.
Outputs: docs/reports/pool_summary.md, benchmark/pilot_shortlist.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pool = pd.read_csv(ROOT / "benchmark" / "candidate_pool.csv", low_memory=False)
ok = pool[pool.get("skip_reason").isna()] if "skip_reason" in pool else pool
om = ok[ok["source"] == "OpenML"].copy()
tc = ok[ok["source"] == "TOC-UCO"].copy()


def as_kind(rule):
    return {"binary": "classification", "multiclass": "classification",
            "regression": "regression"}.get(rule, rule)


single = om[om["openml_task_type"].isin(["Supervised Classification", "Supervised Regression"])].copy()
single["registered"] = single["openml_task_type"].map(
    {"Supervised Classification": "classification", "Supervised Regression": "regression"})
single["ag_csv_kind"] = single["rule_autogluon_csv"].map(as_kind)
single["ag_typed_kind"] = single["rule_autogluon_typed"].map(as_kind)

lines = ["# Candidate pool summary", "",
         f"- profiled candidates: {len(ok)} (OpenML {len(om)}, TOC-UCO {len(tc)}); "
         f"skipped: {len(pool) - len(ok)}", ""]

lines += ["## 1. Does the file format leak the task?", "",
          "AutoGluon rule on OpenML's typed view (ARFF nominal -> category) vs. the same values "
          "re-read from plain CSV:", ""]
flip = single[single["ag_typed_kind"] != single["ag_csv_kind"]]
lines += [f"- decision changes for {len(flip)} / {len(single)} targets "
          f"({100 * len(flip) / max(1, len(single)):.1f}%) once type metadata is removed.", ""]
if len(flip):
    lines += ["| did | dataset | target | typed | csv | registered |", "|---|---|---|---|---|---|"]
    for r in flip.head(40).itertuples():
        lines.append(f"| {r.did} | {r.dataset_name} | {r.target_column} | {r.rule_autogluon_typed} | "
                     f"{r.rule_autogluon_csv} | {r.registered} |")
    lines.append("")

lines += ["## 2. Rule vs registered task type (CSV view)", ""]
dis = single[single["ag_csv_kind"] != single["registered"]]
ct = pd.crosstab(single["registered"], single["rule_autogluon_csv"])
lines += [ct.to_markdown(), "", f"- disagreements: {len(dis)} / {len(single)}", ""]

lines += ["## 3. Integer-valued regression targets (count / ordinal suspects)", ""]
intreg = single[(single["registered"] == "regression") & (single["integer_valued"] == True)]
lines += ["| did | dataset | target | n_unique | min | max | zeros | rule(csv) |", "|---|---|---|---|---|---|---|---|"]
for r in intreg.sort_values("n_unique_target").itertuples():
    lines.append(f"| {r.did} | {r.dataset_name} | {r.target_column} | {r.n_unique_target} | "
                 f"{r.min_target:g} | {r.max_target:g} | {r.zero_count} | {r.rule_autogluon_csv} |")
lines.append("")

lines += ["## 4. Numeric-coded classification targets", ""]
numcls = single[(single["registered"] == "classification") & single["integer_valued"].notna()
                & (single["n_unique_target"] > 2)]
lines += ["| did | dataset | target | n_unique | sample | rule(csv) |", "|---|---|---|---|---|---|"]
for r in numcls.itertuples():
    lines.append(f"| {r.did} | {r.dataset_name} | {r.target_column} | {r.n_unique_target} | "
                 f"{str(r.target_sample)[:40]} | {r.rule_autogluon_csv} |")
lines.append("")

lines += ["## 5. TOC-UCO ordinal targets under the rules", "",
          pd.crosstab(tc["origin"], tc["rule_autogluon_csv"]).to_markdown(), "",
          "No rule baseline can output ORDINAL; every ordinal target is necessarily mislabeled "
          "at the label level. The question for the paper is whether that costs anything downstream.", ""]

# ---- draft pilot shortlist (20 clear / 20 edge / 20 ordinal-values-only) ----
clear = single[(single["ag_csv_kind"] == single["registered"]) & single["has_description"]
               & ~single.index.isin(intreg.index) & ~single.index.isin(numcls.index)]
clear = pd.concat([clear[clear["registered"] == "classification"].sort_values("n_samples").head(12),
                   clear[clear["registered"] == "regression"].sort_values("n_samples").head(8)])
multi = om[om["openml_task_type"].str.contains(r"\+", na=False)]
multi = multi[multi["n_unique_target"] > 2]
edge = pd.concat([intreg.sort_values("n_unique_target").head(8), numcls.head(6),
                  multi.sort_values("n_samples").head(6)]).drop_duplicates("candidate_id").head(20)
toc = pd.concat([tc[tc["origin"].str.contains("originally")].head(12),
                 tc[tc["origin"].str.contains("discretised")].head(8)])
short = pd.concat([clear.assign(pilot_group="A-clear"), edge.assign(pilot_group="B-edge"),
                   toc.assign(pilot_group="C/D-ordinal-values-only")])
cols = ["pilot_group", "candidate_id", "dataset_name", "target_column", "origin", "openml_task_type",
        "n_samples", "n_features", "observed_representation", "target_sample",
        "rule_autogluon_csv", "rule_sklearn_csv", "source_url", "local_file"]
short[cols].to_csv(ROOT / "benchmark" / "pilot_shortlist.csv", index=False)
lines += ["## 6. Draft pilot shortlist", "",
          short["pilot_group"].value_counts().to_markdown(), "",
          "Written to `benchmark/pilot_shortlist.csv`. This is a starting point only — every "
          "row still needs documentation review (E1/E2 evidence) before it enters the benchmark.", ""]

(ROOT / "docs" / "reports").mkdir(parents=True, exist_ok=True)
(ROOT / "docs" / "reports" / "pool_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12]))
