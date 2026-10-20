"""Build blind annotation packets for the pilot (benchmark/PROTOCOL.md, conditions C0/C1/C2).

For each pilot target, writes three Markdown packets under annotation/packets/<condition>/<iid>.md,
where <iid> is an anonymous, shuffled instance id (I001, I002, ...). The key that maps ids back to
candidates goes to benchmark/annotation_key.csv, outside the folder annotators read.

Packets contain only what the condition allows:
  C0  target values (summary + value counts + random sample), no names
  C1  C0 + real target/feature column names + a few table rows
  C2  C1 + dataset title + available documentation text
No packet contains registered task types, rule/baseline outputs or kill-test results.

    .venv/Scripts/python scripts/build_annotation_packets.py
"""
import io
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotation" / "packets"
KEY = ROOT / "benchmark" / "annotation_key.csv"
SEED = 20260926
N_SAMPLE_VALUES = 30
MAX_LEVELS_SHOWN = 30
MAX_FEATURE_NAMES = 60
MAX_TABLE_COLS = 15
N_TABLE_ROWS = 5
MAX_DOC_CHARS = 5000

TOCUCO_TEXT = ROOT / "data" / "metadata" / "tocuco_sources_extracted.txt"
# Per-dataset paragraphs in the TOC-UCO paper (section 4), located by their headings.
TOCUCO_SECTIONS = {
    "forestfires": "4.1.1. Forestfires.",
    "buoysFlux46026": "4.1.2. BuoysFlux and BuoysHeight datasets.",
    "buoysFlux46059": "4.1.2. BuoysFlux and BuoysHeight datasets.",
    "buoysFlux46069": "4.1.2. BuoysFlux and BuoysHeight datasets.",
    "mammoexp": "4.2.1. Mammoexp.",
    "childrenAnemia": "4.2.2. ChildrenAnemia.",
    "gymExerciseTracking": "4.2.3. GymExerciseTracking.",
    "heartDisease": "4.2.4. HeartDisease.",
    "problematicInternetUsage": "4.2.5. ProblematicInternetUsage.",
    "support": "4.2.6. Support.",
}
TOCUCO_DR_NOTE = ("Repository note (TOC-UCO paper, Fig. 4 caption and section 4.1): datasets in this "
                  "group were built from a regression problem whose continuous output was discretised "
                  "into ordered intervals with k-means; the column `y` holds the interval index.")


def tocuco_paragraph(name):
    heading = TOCUCO_SECTIONS.get(name)
    if not heading:
        return None
    text = TOCUCO_TEXT.read_text(encoding="utf-8")
    start = text.index(heading)
    nxt = re.search(r"\n4\.\d\.\d+\. |\n4\.2\. |\n5\. |\n\d+https?://", text[start + len(heading):])
    end = start + len(heading) + (nxt.start() if nxt else 1500)
    return text[start:end].strip()


def load(row):
    if row.candidate_id.startswith("openml_"):
        d = ROOT / "data" / "raw" / row.candidate_id
        df = pd.read_parquet(d / "data.parquet")
        desc = (d / "description.txt").read_text(encoding="utf-8") if (d / "description.txt").exists() else ""
        # Plain-CSV view of the whole table: annotators see values as a raw file shows them.
        df = pd.read_csv(io.StringIO(df.to_csv(index=False)))
        return df, row.dataset_name, desc.strip() or None
    df = pd.read_csv(ROOT / row.local_file.replace("\\", "/"))
    name = row.dataset_name
    parts = []
    para = tocuco_paragraph(name)
    if para:
        parts.append("Source description (TOC-UCO paper, section 4):\n\n" + para)
    if "discretised-regression" in row.origin:
        parts.append(TOCUCO_DR_NOTE)
    return df, name, "\n\n".join(parts) or None


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


def c0_block(y):
    n = len(y)
    miss = int(y.isna().sum())
    yv = y.dropna()
    lines = [f"- rows: {n}", f"- missing target values: {miss} ({100 * miss / n:.1f}%)",
             f"- type as read from a plain CSV: `{y.dtype}`", f"- distinct values: {yv.nunique()}"]
    if pd.api.types.is_numeric_dtype(yv):
        integer = bool(np.all(np.mod(yv, 1) == 0))
        lines += [f"- min / max: {fmt(yv.min())} / {fmt(yv.max())}",
                  f"- mean / median / std: {fmt(yv.mean())} / {fmt(yv.median())} / {fmt(yv.std())}",
                  f"- all values are whole numbers: {'yes' if integer else 'no'}",
                  f"- zeros: {int((yv == 0).sum())}; negative values: {int((yv < 0).sum())}"]
    vc = yv.value_counts()
    if len(vc) <= MAX_LEVELS_SHOWN:
        try:
            vc = vc.sort_index()
            title = "All distinct values (sorted by value) with counts"
        except TypeError:
            title = "All distinct values with counts"
    else:
        vc = vc.head(MAX_LEVELS_SHOWN)
        title = f"The {MAX_LEVELS_SHOWN} most frequent values with counts"
    lines += ["", f"**{title}:**", "", "| value | count |", "|---|---|"]
    lines += [f"| {fmt(k)} | {int(c)} |" for k, c in vc.items()]
    sample = yv.sample(min(N_SAMPLE_VALUES, len(yv)), random_state=0)
    lines += ["", f"**Random sample of {len(sample)} values (in file order of sampling):**", "",
              ", ".join(fmt(v) for v in sample)]
    return "\n".join(lines)


def c1_block(df, target):
    feats = [c for c in df.columns if c != target]
    shown = feats[:MAX_FEATURE_NAMES]
    more = f" … and {len(feats) - len(shown)} more" if len(feats) > len(shown) else ""
    cols = [target] + feats[: MAX_TABLE_COLS - 1]
    rows = df[cols].sample(min(N_TABLE_ROWS, len(df)), random_state=1)
    table = rows.to_csv(index=False).strip()
    return "\n".join([
        f"- target column name: `{target}`",
        f"- number of feature columns: {len(feats)}",
        f"- feature column names: {', '.join(f'`{c}`' for c in shown)}{more}",
        "", f"**{len(rows)} random rows (target first; at most {MAX_TABLE_COLS} columns), as CSV:**", "",
        "```", table, "```"])


def c2_block(title, doc):
    if doc and len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS] + "\n\n[... documentation truncated ...]"
    return "\n".join([f"- dataset title: `{title}`", "", "**Available documentation:**", "",
                      doc if doc else "_No documentation is available for this dataset._"])


def main():
    pilot = pd.read_csv(ROOT / "benchmark" / "pilot_shortlist.csv")
    order = list(range(len(pilot)))
    random.Random(SEED).shuffle(order)
    iids = {pilot.candidate_id[i]: f"I{k + 1:03d}" for k, i in enumerate(order)}
    for c in ("C0", "C1", "C2"):
        (OUT / c).mkdir(parents=True, exist_ok=True)
    key = []
    for row in pilot.itertuples():
        iid = iids[row.candidate_id]
        df, title, doc = load(row)
        target = row.target_column
        y = df[target]
        head = {"C0": "values only", "C1": "values + table", "C2": "values + table + documentation"}
        blocks = {"C0": [("Target values", c0_block(y))]}
        blocks["C1"] = blocks["C0"] + [("Table", c1_block(df, target))]
        blocks["C2"] = blocks["C1"] + [("Dataset documentation", c2_block(title, doc))]
        for c, parts in blocks.items():
            text = [f"# Instance {iid} — condition {c} ({head[c]})", ""]
            for h, body in parts:
                text += [f"## {h}", "", body, ""]
            (OUT / c / f"{iid}.md").write_text("\n".join(text), encoding="utf-8")
        key.append({"instance_id": iid, "candidate_id": row.candidate_id, "dataset_name": row.dataset_name,
                    "target_column": target, "pilot_group": row.pilot_group, "origin": row.origin,
                    "has_documentation": bool(doc)})
    pd.DataFrame(key).sort_values("instance_id").to_csv(KEY, index=False)
    print(f"wrote {len(key)} instances x 3 conditions to {OUT}; key -> {KEY}")


if __name__ == "__main__":
    main()
