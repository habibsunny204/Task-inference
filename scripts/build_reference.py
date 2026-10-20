"""Build the reference labels from the adjudicated sheet plus the agreed rows (PROTOCOL.md §3–§6).

    .venv/Scripts/python scripts/build_reference.py

Reads  annotation/adjudication/adjudication_sheet.csv   (filled adjudicated_* columns)
       annotation/adjudication/agreed.csv
       annotation/adjudication/moderator_notes.csv   (optional) appended to the reason, never changes a label
Writes benchmark/reference_labels.csv      one row per (instance, condition)
       benchmark/reference_instances.csv   one row per instance: sets per condition + protocol §5 ambiguity type
       benchmark/reference_labels.sha256   hash of both files (freeze before scoring baselines)
       annotation/adjudication/reference_report.md

Hard errors (stop, nothing written): missing or duplicate rows, unknown tasks, most likely task not in the
set, NONE with fewer than 2 tasks, C2 row without an evidence level, missing reason, shared C0 group with
different decisions, adjudicated set with a task neither annotator proposed.
Warnings (reported, not fixed): C2 evidence level other than E5 on a set of 2+ tasks, sets that are not
nested C2 ⊆ C1 ⊆ C0.

c2_reference_kind refines the protocol's EPISTEMIC type without replacing it:
  SINGLE            |set(C2)| = 1
  SET_FAVOURED      |set(C2)| > 1 and a most likely task was named
  SET_NO_FAVOURITE  |set(C2)| > 1 and most likely is NONE
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "annotation" / "adjudication"
BENCH = ROOT / "benchmark"
PACKETS = ROOT / "annotation" / "packets"
LABELS = {"BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"}
LEVELS = {"E1", "E2", "E3", "E4", "E5"}
CONDITIONS = ("C0", "C1", "C2")
COLS = ["instance_id", "condition", "adjudicated_set", "adjudicated_most_likely",
        "adjudicated_evidence_level", "adjudication_reason"]


def load():
    sheet = pd.read_csv(ADJ / "adjudication_sheet.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    agreed = pd.read_csv(ADJ / "agreed.csv", dtype=str, keep_default_na=False)
    sheet = sheet.assign(source="adjudicated")
    agreed = agreed.assign(source="agreed", status="AGREED", shared_C0_group="",
                           plausible_task_set_a=agreed.adjudicated_set, plausible_task_set_b=agreed.adjudicated_set)
    keep = COLS + ["source", "status", "shared_C0_group", "plausible_task_set_a", "plausible_task_set_b"]
    df = pd.concat([sheet[keep], agreed[keep]], ignore_index=True)
    for c in COLS:
        df[c] = df[c].str.strip()
    notes = ADJ / "moderator_notes.csv"
    if notes.exists():
        n = pd.read_csv(notes, dtype=str, keep_default_na=False)
        idx = df.set_index(["instance_id", "condition"]).index
        for r in n.itertuples():
            hit = idx == (r.instance_id, r.condition)
            if not hit.any():
                raise SystemExit(f"moderator note for unknown row {r.instance_id}/{r.condition}")
            df.loc[hit, "adjudication_reason"] += " | " + r.note.strip()
    df["adjudicated_set"] = df.adjudicated_set.str.upper().str.split("|").apply(
        lambda s: "|".join(sorted({x.strip() for x in s if x.strip()})))
    df["adjudicated_most_likely"] = df.adjudicated_most_likely.str.upper()
    df["adjudicated_evidence_level"] = df.adjudicated_evidence_level.str.upper()
    return df


def check(df):
    errors, warnings = [], []
    expected = {(p.stem, c) for c in CONDITIONS for p in (PACKETS / c).glob("I*.md")}
    got = list(zip(df.instance_id, df.condition))
    if missing := sorted(expected - set(got)):
        errors.append(f"missing rows: {', '.join(f'{i}/{c}' for i, c in missing)}")
    if dup := sorted({k for k in got if got.count(k) > 1}):
        errors.append(f"duplicate rows: {', '.join(f'{i}/{c}' for i, c in dup)}")
    for r in df.itertuples():
        k = f"{r.instance_id}/{r.condition}"
        s = [x for x in r.adjudicated_set.split("|") if x]
        ml, lvl = r.adjudicated_most_likely, r.adjudicated_evidence_level
        if not s:
            errors.append(f"{k}: empty set")
        if bad := set(s) - LABELS:
            errors.append(f"{k}: unknown tasks {sorted(bad)}")
        if ml == "NONE" and len(s) < 2:
            errors.append(f"{k}: NONE with a set of fewer than 2 tasks")
        elif ml != "NONE" and ml not in s:
            errors.append(f"{k}: most likely {ml!r} not in set")
        if r.condition == "C2" and lvl not in LEVELS:
            errors.append(f"{k}: evidence level {lvl!r} not in {sorted(LEVELS)}")
        if r.condition != "C2" and lvl:
            errors.append(f"{k}: evidence level given outside C2")
        if not r.adjudication_reason:
            errors.append(f"{k}: no adjudication reason")
        proposed = set(r.plausible_task_set_a.split("|")) | set(r.plausible_task_set_b.split("|"))
        if extra := set(s) - proposed:
            errors.append(f"{k}: set adds {sorted(extra)}, proposed by neither annotator")
        if r.condition == "C2" and len(s) > 1 and lvl != "E5":
            warnings.append(f"{k}: level {lvl} on a set of {len(s)} ({r.adjudicated_set}, most likely {ml})")
    for g, d in df[df.shared_C0_group != ""].groupby("shared_C0_group"):
        if d.adjudicated_set.nunique() > 1 or d.adjudicated_most_likely.nunique() > 1:
            errors.append(f"shared C0 group {g}: different decisions")
    return errors, warnings


def per_instance(df):
    key = pd.read_csv(BENCH / "annotation_key.csv", dtype=str).set_index("instance_id")
    rows, nesting = [], []
    for iid, g in df.groupby("instance_id"):
        g = g.set_index("condition")
        sets = {c: set(g.at[c, "adjudicated_set"].split("|")) for c in CONDITIONS}
        n = {c: len(sets[c]) for c in CONDITIONS}
        ml2 = g.at["C2", "adjudicated_most_likely"]
        if not (sets["C2"] <= sets["C1"] <= sets["C0"]):
            nesting.append(f"{iid}: C0={g.at['C0', 'adjudicated_set']} C1={g.at['C1', 'adjudicated_set']} "
                           f"C2={g.at['C2', 'adjudicated_set']}")
        rows.append({
            "instance_id": iid, **key.loc[iid, ["candidate_id", "dataset_name", "target_column", "pilot_group"]],
            **{f"set_{c}": g.at[c, "adjudicated_set"] for c in CONDITIONS},
            **{f"most_likely_{c}": g.at[c, "adjudicated_most_likely"] for c in CONDITIONS},
            **{f"size_{c}": n[c] for c in CONDITIONS},
            "evidence_level_C2": g.at["C2", "adjudicated_evidence_level"],
            "ambiguity_type": "EPISTEMIC" if n["C2"] > 1 else "REPRESENTATIONAL" if n["C0"] > 1 else "NONE",
            "resolvable_from_table": n["C1"] == 1,
            "requires_external_context": n["C1"] > 1 and n["C2"] == 1,
            "c2_reference_kind": "SINGLE" if n["C2"] == 1 else "SET_NO_FAVOURITE" if ml2 == "NONE" else "SET_FAVOURED",
        })
    return pd.DataFrame(rows), nesting


def main():
    df = load()
    errors, warnings = check(df)
    if errors:
        print(f"{len(errors)} errors, nothing written:")
        print("\n".join(f"  {e}" for e in errors))
        raise SystemExit(1)

    labels = (df.rename(columns={"adjudicated_set": "reference_set", "adjudicated_most_likely": "reference_most_likely",
                                 "adjudicated_evidence_level": "evidence_level", "status": "adjudication_status"})
              .assign(set_size=lambda d: d.reference_set.str.count(r"\|") + 1)
              [["instance_id", "condition", "reference_set", "set_size", "reference_most_likely", "evidence_level",
                "source", "adjudication_status", "adjudication_reason"]]
              .sort_values(["instance_id", "condition"]))
    inst, nesting = per_instance(df)
    warnings += [f"not nested {x}" for x in nesting]

    labels.to_csv(BENCH / "reference_labels.csv", index=False)
    inst.to_csv(BENCH / "reference_instances.csv", index=False)
    digest = [f"{hashlib.sha256((BENCH / f).read_bytes()).hexdigest()}  {f}"
              for f in ("reference_labels.csv", "reference_instances.csv")]
    (BENCH / "reference_labels.sha256").write_text("\n".join(digest) + "\n", encoding="utf-8")

    c2 = labels[labels.condition == "C2"]
    tab = lambda s: " | ".join(f"{k} {v}" for k, v in s.value_counts().items())  # noqa: E731
    lines = ["# Reference labels — build report", "",
             f"- rows: {len(labels)} ({inst.shape[0]} instances × 3 conditions); "
             f"adjudicated {int((labels.source == 'adjudicated').sum())}, agreed {int((labels.source == 'agreed').sum())}",
             f"- ambiguity type (protocol §5): {tab(inst.ambiguity_type)}",
             f"- C2 reference kind: {tab(inst.c2_reference_kind)}",
             f"- resolvable from table (|set C1| = 1): {int(inst.resolvable_from_table.sum())}; "
             f"requires external context: {int(inst.requires_external_context.sum())}",
             f"- C2 most likely: {tab(c2.reference_most_likely)}",
             f"- C2 evidence level: {tab(c2.evidence_level)}",
             "- mean set size: " + ", ".join(f"{c} {labels[labels.condition == c].set_size.mean():.2f}"
                                             for c in CONDITIONS),
             "", f"## Warnings ({len(warnings)})", ""] + [f"- {w}" for w in warnings]
    (ADJ / "reference_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
