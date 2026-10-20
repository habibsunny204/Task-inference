"""Validate and compile one annotator's filled sheet (the columns of the recording template, PROTOCOL.md fields).

Reads  <annotator_dir>/annotations.csv   (the returned annotator_N_annotations.csv, saved under this name)
Writes into <annotator_dir>:
  annotations_compiled.csv   one row per (instance, condition), normalised + validity flag
  derived.csv                one row per instance: set sizes per condition + protocol §5 ambiguity type
  validation_report.md       schema violations, missing rows, non-verbatim quotes

    .venv/Scripts/python scripts/compile_annotations.py annotation/annotator_1
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "annotation" / "packets"
LABELS = {"BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"}
LEVELS = {"E1", "E2", "E3", "E5"}
CONDITIONS = ("C0", "C1", "C2")


def norm(s):
    """Compare quotes ignoring whitespace, case and markdown emphasis (*, _, `) in the packet."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", str(s))).strip().lower()


def load(adir):
    df = pd.read_csv(adir / "annotations.csv", dtype=str, encoding="utf-8-sig").fillna("")
    rows = []
    for r in df.itertuples():
        if not r.plausible_task_set.strip():
            continue  # unfilled row; reported as missing
        conf = r.confidence.strip()
        rec = {"instance_id": r.instance_id.strip(), "condition": r.condition.strip(),
               "plausible_task_set": [x.strip().upper() for x in r.plausible_task_set.split("|") if x.strip()],
               "most_likely_task": r.most_likely_task.strip().upper(),
               "confidence": int(float(conf)) if re.fullmatch(r"\d+(\.0+)?", conf) else conf or None,
               "evidence": r.evidence.strip()}
        if rec["condition"] == "C2":
            rec["evidence_level"] = r.evidence_level.strip().upper()
            rec["evidence_quote"] = r.evidence_quote.strip() or None
        rows.append(rec)
    return rows


def validate(rec):
    errs = []
    c = rec["condition"]
    if c not in CONDITIONS:
        errs.append(f"condition {c!r} not in {CONDITIONS}")
    s = rec["plausible_task_set"]
    bad = [x for x in s if x not in LABELS]
    if bad:
        errs.append(f"unknown labels in set: {bad}")
    m = rec["most_likely_task"]
    if m != "NONE" and m not in s:
        errs.append(f"most_likely_task {m!r} not in set and not NONE")
    if m == "NONE" and len(s) < 2:
        errs.append("NONE with a set of fewer than 2 labels")
    if len(s) == 1 and m != s[0]:
        errs.append("single-label set but most_likely_task differs")
    conf = rec["confidence"]
    if not isinstance(conf, int) or not 1 <= conf <= 5:
        errs.append(f"confidence {conf!r} not an integer 1-5")
    if not rec["evidence"]:
        errs.append("evidence empty")
    if c == "C2":
        lvl = rec.get("evidence_level")
        if lvl not in LEVELS:
            errs.append(f"evidence_level {lvl!r} not in {sorted(LEVELS)}")
        if lvl in ("E1", "E2") and not rec.get("evidence_quote"):
            errs.append(f"{lvl} without evidence_quote")
    return errs


def quote_is_verbatim(rec):
    q = rec.get("evidence_quote")
    if not q or rec["condition"] != "C2":
        return None
    packet = PACKETS / "C2" / f"{rec['instance_id']}.md"
    return packet.exists() and norm(q) in norm(packet.read_text(encoding="utf-8"))


def ambiguity(sizes):
    c0, c1, c2 = (sizes.get(c) for c in CONDITIONS)
    if None in (c0, c1, c2):
        return None, None, None
    typ = "EPISTEMIC" if c2 > 1 else "REPRESENTATIONAL" if c0 > 1 else "NONE"
    return typ, c1 == 1, (c1 > 1 and c2 == 1)


def main(adir):
    adir = Path(adir).resolve()
    recs = load(adir)
    expected = {(p.stem, c) for c in CONDITIONS for p in (PACKETS / c).glob("I*.md")}
    rows, problems = [], []
    for r in recs:
        errs = validate(r)
        verb = quote_is_verbatim(r)
        if verb is False:
            errs.append("evidence_quote not found verbatim in the packet")
        if errs:
            problems.append((r["instance_id"], r["condition"], errs))
        rows.append({"instance_id": r["instance_id"], "condition": r["condition"],
                     "plausible_task_set": "|".join(sorted(r["plausible_task_set"])),
                     "set_size": len(r["plausible_task_set"]),
                     "most_likely_task": r["most_likely_task"], "confidence": r["confidence"],
                     "evidence": r["evidence"], "evidence_level": r.get("evidence_level"),
                     "evidence_quote": r.get("evidence_quote"), "quote_verbatim": verb, "valid": not errs})
    df = pd.DataFrame(rows).sort_values(["instance_id", "condition"])
    missing = sorted(expected - set(zip(df.instance_id, df.condition)))
    df.to_csv(adir / "annotations_compiled.csv", index=False)

    der = []
    for iid, g in df[df.valid].groupby("instance_id"):
        sizes = dict(zip(g.condition, g.set_size))
        typ, from_table, ext = ambiguity(sizes)
        sets = dict(zip(g.condition, g.plausible_task_set))
        der.append({"instance_id": iid, **{f"set_{c}": sets.get(c) for c in CONDITIONS},
                    **{f"size_{c}": sizes.get(c) for c in CONDITIONS},
                    "ambiguity_type": typ, "resolvable_from_table": from_table,
                    "requires_external_context": ext})
    pd.DataFrame(der).to_csv(adir / "derived.csv", index=False)

    lines = [f"# Validation report — {adir.name}", "",
             f"- filled rows: {len(df)} (expected {len(expected)})", f"- valid: {int(df.valid.sum())}",
             f"- missing (unfilled) rows: {len(missing)}", f"- rows with problems: {len(problems)}", ""]
    if missing:
        lines += ["## Missing", "", ", ".join(f"{i}/{c}" for i, c in missing), ""]
    if problems:
        lines += ["## Problems", ""] + [f"- {i}/{c}: {'; '.join(e)}" for i, c, e in problems] + [""]
    (adir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "annotation" / "annotator_1")
