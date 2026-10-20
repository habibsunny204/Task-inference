"""Group C: controlled transformations of pilot targets (PROTOCOL.md §7). Frozen 2026-09-27.

Each transformed instance keeps its base candidate, the transformation, its parameters and the
expected task set. The expected task follows from the base target's documented task and the
transformation (it is "known by construction"). It is provisional until the base target's
reference label is adjudicated. Confirmed 2026-10-05 against benchmark/reference_instances.csv:
task-preserving transforms (numcode, round, anon) carry the base's adjudicated C2 set; bin changes the task
by construction (all five bases are single-task CONTINUOUS). Only G018 changed (+ORDINAL). Group C instances are never shown to annotators and never mixed
into headline numbers; they share the data split of their base table.

    .venv/Scripts/python scripts/build_group_c.py
writes benchmark/group_c/{key.csv, data/G###.csv, docs/G###.md, packets/C0|C1|C2/G###.md}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_annotation_packets as bap  # noqa: E402
from instances import pilot_instances  # noqa: E402

OUT = ROOT / "benchmark" / "group_c"
B, M, O, CNT, CONT = "BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"
EUC_ORDER = ["none", "low", "average", "good", "best"]

# (base candidate, transformation, params, expected set, expected most likely, rationale)
SPECS = [
    # numcode: string labels -> integer codes (a raw file with codes and no dictionary)
    ("openml_29", "numcode", {"mapping": "alphabetical"}, {B}, B, "two outcome classes; coding does not change the task"),
    ("openml_15", "numcode", {"mapping": "alphabetical"}, {B}, B, "two outcome classes; coding does not change the task"),
    ("openml_45104", "numcode", {"mapping": "alphabetical"}, {M}, M, "86 unordered plankton taxa; integer codes carry no order"),
    ("openml_188", "numcode", {"mapping": "semantic", "order": EUC_ORDER}, {O, M}, O,
     "utility rating none<low<average<good<best, coded in rating order"),
    ("openml_188", "numcode", {"mapping": "alphabetical"}, {O, M}, O,
     "same rating, coded alphabetically (LabelEncoder): the task is still a rating, the code order is not"),
    # round: continuous measurement rounded to whole numbers
    ("openml_44959", "round", {"decimals": 0}, {CONT}, CONT, "compressive strength (MPa) is a measurement; rounding keeps the task"),
    ("openml_531", "round", {"decimals": 0}, {CONT}, CONT, "median house value is a continuous quantity"),
    ("openml_44960", "round", {"decimals": 0}, {CONT}, CONT, "heating load is a continuous physical quantity"),
    ("openml_44970", "round", {"decimals": 0}, {CONT}, CONT, "LC50 toxicity (-log mol/L) is continuous; rounding leaves ~10 integer levels"),
    ("openml_505", "round", {"decimals": 0}, {CONT}, CONT, "fat content (%) is a continuous measurement"),
    # bin: continuous target discretised into 5 equal-frequency bins (index 0..4), as TOC-UCO does
    ("openml_44959", "bin", {"bins": 5, "strategy": "quantile"}, {O}, O, "ordered interval index of a continuous target"),
    ("openml_44994", "bin", {"bins": 5, "strategy": "quantile"}, {O}, O, "ordered interval index of a continuous target"),
    ("openml_44960", "bin", {"bins": 5, "strategy": "quantile"}, {O}, O, "ordered interval index of a continuous target"),
    ("openml_44970", "bin", {"bins": 5, "strategy": "quantile"}, {O}, O, "ordered interval index of a continuous target"),
    ("openml_505", "bin", {"bins": 5, "strategy": "quantile"}, {O}, O, "ordered interval index of a continuous target"),
    # anon: target renamed to y, features to x_0.. (values untouched)
    ("openml_44994", "anon", {}, {CONT}, CONT, "car price; names removed, values unchanged"),
    ("openml_44962", "anon", {}, {CONT}, CONT, "burned area (ha); names removed, values unchanged"),
    ("openml_44966", "anon", {}, {CNT, O}, CNT,
     "number of C-class flares in 24 h (documented count); names removed; base C2 set is COUNT|ORDINAL"),
    ("openml_40994", "anon", {}, {B}, B, "simulation crash yes/no; names removed"),
    ("openml_1063", "anon", {}, {B}, B, "software module defective yes/no; names removed"),
]
NOTES = {
    "numcode": "Preprocessing note: the target column was integer-encoded.",
    "bin": "Preprocessing note: the original continuous target was discretised into {bins} equal-frequency "
           "intervals; the target column holds the interval index (0 = lowest).",
}


def transform(df, target, kind, params):
    df = df.copy()
    y = df[target]
    if kind == "numcode":
        cats = params.get("order") or sorted(y.dropna().unique())
        df[target] = y.map({c: i + 1 for i, c in enumerate(cats)})
    elif kind == "round":
        df[target] = y.round(params["decimals"]).astype("Int64")
    elif kind == "bin":
        df[target] = pd.qcut(y, q=params["bins"], labels=False, duplicates="drop").astype("Int64")
    elif kind == "anon":
        feats = [c for c in df.columns if c != target]
        df = df.rename(columns={target: "y", **{c: f"x_{i}" for i, c in enumerate(feats)}})
        target = "y"
    return df, target


def main():
    key = pd.read_csv(ROOT / "benchmark" / "annotation_key.csv").set_index("candidate_id")
    bases = {}
    wanted = {s[0] for s in SPECS}
    for inst in pilot_instances():
        cid = key.index[key.instance_id == inst.instance_id][0]
        if cid in wanted:
            bases[cid] = inst
    for sub in ("data", "docs", "packets/C0", "packets/C1", "packets/C2"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (cid, kind, params, exp_set, exp_ml, why) in enumerate(SPECS, start=1):
        gid = f"G{n:03d}"
        base = bases[cid]
        df, target = transform(base.df, base.target, kind, params)
        df.to_csv(OUT / "data" / f"{gid}.csv", index=False)
        doc = base.doc or ""
        if kind in NOTES:
            doc = (doc + "\n\n" if doc else "") + NOTES[kind].format(**params)
        (OUT / "docs" / f"{gid}.md").write_text(doc, encoding="utf-8")
        y = pd.read_csv(OUT / "data" / f"{gid}.csv")[target]
        blocks = {"C0": [("Target values", bap.c0_block(y))]}
        blocks["C1"] = blocks["C0"] + [("Table", bap.c1_block(pd.read_csv(OUT / "data" / f"{gid}.csv"), target))]
        blocks["C2"] = blocks["C1"] + [("Dataset documentation", bap.c2_block(base.title, doc or None))]
        head = {"C0": "values only", "C1": "values + table", "C2": "values + table + documentation"}
        for c, parts in blocks.items():
            text = [f"# Instance {gid} — condition {c} ({head[c]})", ""]
            for h, body in parts:
                text += [f"## {h}", "", body, ""]
            (OUT / "packets" / c / f"{gid}.md").write_text("\n".join(text), encoding="utf-8")
        rows.append({"instance_id": gid, "base_candidate_id": cid, "base_instance_id": base.instance_id,
                     "transformation_type": kind, "parameters": str(params), "random_seed": 0,
                     "expected_task_set": "|".join(sorted(exp_set)), "expected_most_likely": exp_ml,
                     "rationale": why, "expected_basis": "adjudicated base C2 reference + transformation (confirmed 2026-10-05)",
                     "title": base.title, "target_column": target, "n_unique_target": int(y.nunique()),
                     "data_file": f"benchmark/group_c/data/{gid}.csv", "doc_file": f"benchmark/group_c/docs/{gid}.md"})
    pd.DataFrame(rows).to_csv(OUT / "key.csv", index=False)
    print(pd.DataFrame(rows)[["instance_id", "title", "transformation_type", "n_unique_target",
                              "expected_task_set"]].to_string(index=False))


if __name__ == "__main__":
    main()
