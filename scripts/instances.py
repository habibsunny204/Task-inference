"""Shared loader for benchmark instances, used by baselines and downstream experiments.

pilot    56 targets from benchmark/pilot_shortlist.csv, ids from benchmark/annotation_key.csv
group_c  controlled transformations from benchmark/group_c/key.csv (scripts/build_group_c.py)

Each instance gives the table as a plain-CSV view, its target column, title and documentation:
exactly what the annotation packets are built from.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_annotation_packets as bap  # noqa: E402


@dataclass
class Instance:
    instance_id: str
    group: str            # "pilot" or "group_c"
    df: pd.DataFrame      # table as read from plain CSV
    target: str
    title: str
    doc: str | None


def pilot_instances():
    key = pd.read_csv(ROOT / "benchmark" / "annotation_key.csv")
    pilot = pd.read_csv(ROOT / "benchmark" / "pilot_shortlist.csv").set_index("candidate_id")
    for k in key.itertuples():
        row = pilot.loc[k.candidate_id]
        row = pd.Series({**row.to_dict(), "candidate_id": k.candidate_id})
        df, title, doc = bap.load(row)
        yield Instance(k.instance_id, "pilot", df, k.target_column, title, doc)


def group_c_instances():
    path = ROOT / "benchmark" / "group_c" / "key.csv"
    if not path.exists():
        return
    for k in pd.read_csv(path).itertuples():
        df = pd.read_csv(ROOT / k.data_file)
        doc_path = ROOT / k.doc_file if isinstance(k.doc_file, str) else None
        doc = doc_path.read_text(encoding="utf-8") if doc_path and doc_path.exists() else None
        yield Instance(k.instance_id, "group_c", df, k.target_column, k.title, doc)


def all_instances(groups=("pilot", "group_c")):
    if "pilot" in groups:
        yield from pilot_instances()
    if "group_c" in groups:
        yield from group_c_instances()
