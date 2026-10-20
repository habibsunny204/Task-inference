"""Agreement between two annotators (PROTOCOL.md §6): per condition, raw agreement and Cohen's kappa
on most_likely_task, mean Jaccard and exact match on plausible_task_set, and agreement on the
derived ambiguity type. Run compile_annotations.py on both annotator folders first.

    .venv/Scripts/python scripts/annotation_agreement.py annotation/annotator_1 annotation/annotator_2
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]


def jaccard(a, b):
    a, b = set(a.split("|")), set(b.split("|"))
    return len(a & b) / len(a | b) if a | b else 1.0


def main(dir_a, dir_b):
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    a = pd.read_csv(dir_a / "annotations_compiled.csv")
    b = pd.read_csv(dir_b / "annotations_compiled.csv")
    if "valid" in a:
        a = a[a.valid]
    if "valid" in b:
        b = b[b.valid]
    excl_path = ROOT / "annotation" / "EXCLUSIONS.csv"
    excluded = set(pd.read_csv(excl_path).instance_id) if excl_path.exists() else set()
    a, b = a[~a.instance_id.isin(excluded)], b[~b.instance_id.isin(excluded)]
    m = a.merge(b, on=["instance_id", "condition"], suffixes=("_a", "_b"))
    rows = []
    for c, g in m.groupby("condition"):
        rows.append({
            "condition": c, "n": len(g),
            "most_likely_agreement": (g.most_likely_task_a == g.most_likely_task_b).mean(),
            "most_likely_kappa": cohen_kappa_score(g.most_likely_task_a, g.most_likely_task_b),
            "set_exact_match": (g.plausible_task_set_a == g.plausible_task_set_b).mean(),
            "set_mean_jaccard": g.apply(lambda r: jaccard(r.plausible_task_set_a, r.plausible_task_set_b),
                                        axis=1).mean(),
            "both_ambiguous": ((g.set_size_a > 1) & (g.set_size_b > 1)).mean(),
            "ambiguity_agreement": ((g.set_size_a > 1) == (g.set_size_b > 1)).mean(),
        })
    res = pd.DataFrame(rows)
    da, db = pd.read_csv(dir_a / "derived.csv"), pd.read_csv(dir_b / "derived.csv")
    dm = da.merge(db, on="instance_id", suffixes=("_a", "_b")).dropna(subset=["ambiguity_type_a", "ambiguity_type_b"])
    dm = dm[~dm.instance_id.isin(excluded)]
    out = ROOT / "annotation" / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{dir_a.name}_vs_{dir_b.name}"
    res.to_csv(out / f"{name}.csv", index=False)
    m.to_csv(out / f"{name}_pairs.csv", index=False)
    lines = [f"# Agreement: {dir_a.name} vs {dir_b.name}", "",
             f"Excluded instances (annotation/EXCLUSIONS.csv): {', '.join(sorted(excluded)) or 'none'}.", "",
             res.round(3).to_markdown(index=False), ""]
    if len(dm):
        lines += [f"Ambiguity type (protocol §5) agreement over {len(dm)} instances: "
                  f"{(dm.ambiguity_type_a == dm.ambiguity_type_b).mean():.3f}; "
                  f"kappa {cohen_kappa_score(dm.ambiguity_type_a, dm.ambiguity_type_b):.3f}", "",
                  pd.crosstab(dm.ambiguity_type_a, dm.ambiguity_type_b).to_markdown(), ""]
    (out / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
