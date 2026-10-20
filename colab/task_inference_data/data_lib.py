"""Colab data bundle: build one instance store for training (silver + Group C v2) and testing (gold + Group C v1).

Labels follow benchmark/silver/RULES.md (frozen 2026-10-05). Test instances (gold I###, Group C v1 G###) are
stored **without labels**; the method bundle joins them from the frozen reference files.

Store layout (under STORE):
  meta/<id>.json       title, documentation, target name, feature names and kinds, full-column target statistics,
                       role / source / family / provenance, and (training roles only) label + rule
  targets/<id>.parquet target column as read from a plain CSV (all rows, or a seeded sample of max_target_rows)
  samples/<id>.csv     seeded sample of sample_rows rows, target first, at most sample_cols columns (plain CSV)
  manifest.csv         one row per instance (written by finalize)
  rdatasets_audit.csv  every Rdatasets document scanned: qualifying calls, chosen response, label or skip reason

Every instance is written as soon as it is built; finished ids are skipped on a rerun.
"""
import html
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION")
NOTES = {  # identical to scripts/build_group_c.py (Group C v1)
    "numcode": "Preprocessing note: the target column was integer-encoded.",
    "bin": "Preprocessing note: the original continuous target was discretised into {bins} equal-frequency "
           "intervals; the target column holds the interval index (0 = lowest).",
}
TOCUCO_SECTIONS = {  # identical to scripts/build_annotation_packets.py
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


class Bundle:
    def __init__(self, base, store=None):
        self.base = Path(base)
        self.inp = self.base / "inputs"
        self.cfg = json.loads((self.base / "config.json").read_text())
        self.store = Path(store) if store else self.base / "store"
        self.cache = self.base / "cache"
        for d in ("meta", "targets", "samples"):
            (self.store / d).mkdir(parents=True, exist_ok=True)
        self.pool = pd.read_csv(self.inp / "pool.csv", dtype={"did": str})
        self.gold_key = pd.read_csv(self.inp / "gold_key.csv")

    # ------------------------------------------------------------------ writing
    def has(self, iid):
        return (self.store / "meta" / f"{iid}.json").exists()

    def write(self, iid, df, target, title, doc, info):
        c = self.cfg
        y = df[target]
        n = len(y)
        yv = y.dropna()
        vc = yv.value_counts()
        stored = y if n <= c["max_target_rows"] else y.sample(c["max_target_rows"], random_state=c["seed"]).sort_index()
        out = pd.DataFrame({"y": stored.to_numpy()})
        if out.y.dtype == object or str(out.y.dtype).startswith("str"):
            out["y"] = out.y.astype("string")
        out.to_parquet(self.store / "targets" / f"{iid}.parquet", index=False)
        feats = [f for f in df.columns if f != target]
        cols = [target] + feats[: c["sample_cols"] - 1]
        df[cols].sample(min(c["sample_rows"], n), random_state=c["sample_seed"]).to_csv(
            self.store / "samples" / f"{iid}.csv", index=False)
        meta = {"instance_id": iid, "title": title, "doc": doc or None, "target_name": target,
                "feature_names": feats, "feature_kinds": [kind(df[f]) for f in feats],
                "n_rows": n, "n_features": len(feats), "target_kind": kind(y),
                "n_missing": int(y.isna().sum()), "n_unique": int(yv.nunique()),
                "top_counts": [[str(k), int(v)] for k, v in vc.head(200).items()],
                "target_stored_rows": len(stored), **info}
        (self.store / "meta" / f"{iid}.json").write_text(json.dumps(meta, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ loading
    def openml_table(self, row):
        import openml
        ds = openml.datasets.get_dataset(int(row.did), download_data=True, download_qualities=False,
                                         download_features_meta_data=False)
        X, _, _, _ = ds.get_data(dataset_format="dataframe")
        p = self.inp / "openml_descriptions" / f"{row.candidate_id}.txt"
        desc = p.read_text(encoding="utf-8") if p.exists() else (ds.description or "")
        return plain_csv(X), row.dataset_name, desc.strip() or None

    def tocuco_table(self, row):
        df = pd.read_csv(self.inp / "tocuco" / "data" / f"{row.did}.csv")
        parts = []
        heading = TOCUCO_SECTIONS.get(row.dataset_name)
        if heading:
            text = (self.inp / "tocuco" / "tocuco_sources_extracted.txt").read_text(encoding="utf-8")
            start = text.index(heading)
            nxt = re.search(r"\n4\.\d\.\d+\. |\n4\.2\. |\n5\. |\n\d+https?://", text[start + len(heading):])
            end = start + len(heading) + (nxt.start() if nxt else 1500)
            parts.append("Source description (TOC-UCO paper, section 4):\n\n" + text[start:end].strip())
        if "discretised-regression" in row.origin:
            parts.append(TOCUCO_DR_NOTE)
        return df, row.dataset_name, "\n\n".join(parts) or None

    def pool_table(self, row):
        return self.openml_table(row) if row.candidate_id.startswith("openml_") else self.tocuco_table(row)

    # ------------------------------------------------------------------ test sets (no labels stored)
    def build_gold(self, only=None):
        ids = dict(zip(self.gold_key.candidate_id, self.gold_key.instance_id))
        rows = self.pool[self.pool.status == "gold"]
        for row in rows.itertuples():
            iid = ids[row.candidate_id]
            if (only and iid not in only) or self.has(iid):
                continue
            t0 = time.time()
            df, title, doc = self.pool_table(row)
            self.write(iid, df, row.target_column, title, doc,
                       {"role": "gold_test", "source": row.origin, "family": row.family,
                        "candidate_id": row.candidate_id, "check": integrity(df, row)})
            print(f"{iid} {row.dataset_name}: {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)

    def build_group_c_v1(self):
        key = pd.read_csv(self.inp / "group_c_v1" / "key.csv")
        for k in key.itertuples():
            if self.has(k.instance_id):
                continue
            df = pd.read_csv(self.inp / "group_c_v1" / "data" / f"{k.instance_id}.csv")
            dp = self.inp / "group_c_v1" / "docs" / f"{k.instance_id}.md"
            doc = dp.read_text(encoding="utf-8") if dp.exists() else None
            self.write(k.instance_id, df, k.target_column, k.title, doc,
                       {"role": "groupc1_test", "source": "group_c_v1", "family": k.family,
                        "base_id": k.base_instance_id, "transformation": k.transformation_type,
                        "params": k.parameters})
        print(f"Group C v1: {len(key)} instances", flush=True)

    # ------------------------------------------------------------------ training sets
    def build_silver_pool(self, limit=None):
        rows = self.pool[self.pool.status == "silver"].sort_values("candidate_id").reset_index(drop=True)
        for i, row in enumerate(rows.itertuples()):
            if limit is not None and i >= limit:
                break
            iid = f"S{i + 1:03d}"
            if self.has(iid) and self.has(f"{iid}-anon"):
                continue
            t0 = time.time()
            try:
                df, title, doc = self.pool_table(row)
            except Exception as e:  # noqa: BLE001
                print(f"{iid} {row.dataset_name}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            info = {"role": "silver_train", "source": row.origin, "family": row.family,
                    "candidate_id": row.candidate_id, "label": row.silver_label, "label_rule": row.rule,
                    "check": integrity(df, row)}
            self.write(iid, df, row.target_column, title, doc, info)
            n = self.group_c_v2(iid, df, row.target_column, title, doc, info)
            print(f"{iid} {row.dataset_name} [{row.silver_label}]: {len(df)} rows, +{n} transforms "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def group_c_v2(self, base_id, df, target, title, doc, info):
        """Write the RULES.md T1–T4 transforms of one silver base. Returns the number written."""
        c, label, y = self.cfg, info["label"], df[target]
        made = []
        base = {"role": "groupc2_train", "source": "group_c_v2", "family": info["family"], "base_id": base_id,
                "label_rule": f"{info['label_rule']} + transformation"}
        if label in ("BINARY", "MULTICLASS") and kind(y) in ("string", "bool"):
            cats = sorted(y.dropna().astype(str).unique())
            d = df.copy()
            d[target] = y.astype("string").map({v: i + 1 for i, v in enumerate(cats)}).astype("Int64")
            made.append(("numcode", d, target, add_note(doc, NOTES["numcode"]), label, {"mapping": "alphabetical"}))
        if label == "CONTINUOUS_REGRESSION" and kind(y) in ("int", "float"):
            r = y.round(0)
            if not r.equals(y) and r.nunique() >= c["round_min_levels"]:
                d = df.copy()
                d[target] = r.astype("Int64")
                made.append(("round", d, target, doc, label, {"decimals": 0}))
            prev = None
            for k in c["bins"]:
                b = pd.qcut(y, q=k, labels=False, duplicates="drop")
                if b.nunique() < 3 or (prev is not None and b.equals(prev)):
                    continue
                prev = b
                d = df.copy()
                d[target] = b.astype("Int64")
                made.append((f"bin{k}", d, target, add_note(doc, NOTES["bin"].format(bins=int(b.nunique()))),
                             "ORDINAL", {"bins": k, "actual_bins": int(b.nunique()), "strategy": "quantile"}))
        feats = [f for f in df.columns if f != target]
        d = df.rename(columns={target: "y", **{f: f"x_{i}" for i, f in enumerate(feats)}})
        made.append(("anon", d, "y", doc, label, {}))
        for name, d, tgt, dc, lbl, params in made:
            iid = f"{base_id}-{name}"
            if not self.has(iid):
                self.write(iid, d, tgt, title, dc, {**base, "label": lbl, "transformation": name,
                                                    "params": json.dumps(params)})
        return len(made)

    # ------------------------------------------------------------------ Rdatasets (R6)
    def rdatasets_scan(self, limit=None, packages=None):
        """Parse every documented example in Rdatasets; write rdatasets_audit.csv. Returns the selected rows."""
        c = self.cfg
        idx_path = self.cache / "rdatasets" / "datasets.csv"
        if not idx_path.exists():
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            idx_path.write_bytes(fetch(c["rdatasets_index"]))
        idx = pd.read_csv(idx_path)
        idx = idx[idx.Rows >= c["min_rows_rdatasets"]].sort_values(["Package", "Item"]).reset_index(drop=True)
        if packages:
            idx = idx[idx.Package.isin(packages)]
        if limit:
            idx = idx.head(limit)
        gold_names = set((self.inp / "gold_family_names.txt").read_text().split())
        silver_fams = {norm(f): f for f in self.pool[self.pool.status == "silver"].family}
        audit = []
        for r in idx.itertuples():
            rec = {"package": r.Package, "item": r.Item, "title": r.Title, "rows": r.Rows, "csv": r.CSV, "doc_url": r.Doc}
            try:
                page = cached(self.cache / "rdatasets" / "doc" / f"{r.Package}__{safe(r.Item)}.html", r.Doc,
                              c["request_sleep"]).decode("utf-8", "replace")
            except Exception as e:  # noqa: BLE001
                audit.append({**rec, "status": f"doc fetch failed: {e}"})
                continue
            sec = sections(page)
            calls = model_calls(sec.get("Examples", ""), r.Item)
            rec["calls"] = json.dumps(calls)[:2000]
            rec["doc_text"] = doc_text(sec)
            qual = [x for x in calls if x["label"] and not x["skip"]]
            if norm(r.Item) in gold_names:
                rec["status"] = "excluded: gold family name"
            elif not qual:
                rec["status"] = "no qualifying call"
            else:
                lhs = qual[0]["lhs"]
                labels = {x["label"] for x in qual if x["lhs"] == lhs}
                if len(labels) > 1:
                    rec["status"] = f"excluded: conflicting labels {sorted(labels)}"
                else:
                    rec.update({"status": "candidate", "response": lhs, "label": labels.pop(), "fn": qual[0]["fn"],
                                "family": silver_fams.get(norm(r.Item), "rdata_" + norm(r.Item))})
            audit.append(rec)
        a = pd.DataFrame(audit)
        # duplicates across packages: keep the first (Package, Item) per (family, response)
        cand = a.status == "candidate"
        dup = a[cand].duplicated(["family", "response"], keep="first")
        a.loc[dup[dup].index, "status"] = "excluded: duplicate of an earlier package"
        a.to_csv(self.store / "rdatasets_audit.csv", index=False)
        print(a.status.str.split(":").str[0].value_counts().to_string(), flush=True)
        return a[a.status == "candidate"]

    def build_rdatasets(self, limit=None):
        a = pd.read_csv(self.store / "rdatasets_audit.csv")
        sel = a[a.status == "candidate"].reset_index(drop=True)
        if limit:
            sel = sel.head(limit)
        rows = []
        for i, r in enumerate(sel.itertuples()):
            iid = f"R{i + 1:03d}"
            status = "built"
            try:
                if not (self.has(iid) and self.has(f"{iid}-anon")):
                    df = pd.read_csv(io.BytesIO(cached(self.cache / "rdatasets" / "csv" / f"{r.package}__{safe(r.item)}.csv",
                                                       r.csv, self.cfg["request_sleep"])))
                    df = df.drop(columns=[c for c in ("rownames", "Unnamed: 0") if c in df.columns])
                    status = check_response(df, r.response, r.label, self.cfg["min_rows_rdatasets"])
                    if status == "built":
                        two = df[r.response].dropna().nunique() == 2
                        label = "BINARY" if r.label in ("MULTICLASS", "ORDINAL") and two else r.label
                        info = {"role": "silver_train", "source": f"Rdatasets/{r.package}", "family": r.family,
                                "candidate_id": f"rdatasets_{r.package}_{r.item}", "label": label,
                                "label_rule": f"R6 {r.fn} example in the dataset's documentation"}
                        self.write(iid, df, r.response, r.title, r.doc_text if isinstance(r.doc_text, str) else None, info)
                        n = self.group_c_v2(iid, df, r.response, r.title,
                                            r.doc_text if isinstance(r.doc_text, str) else None, info)
                        print(f"{iid} {r.package}/{r.item} [{label}] {r.response}: {len(df)} rows, +{n}", flush=True)
            except Exception as e:  # noqa: BLE001
                status = f"failed: {type(e).__name__}: {e}"
            rows.append({"instance_id": iid, "package": r.package, "item": r.item, "build_status": status})
            if status != "built":
                print(f"{iid} {r.package}/{r.item}: {status}", flush=True)
        pd.DataFrame(rows).to_csv(self.store / "rdatasets_built.csv", index=False)

    # ------------------------------------------------------------------ finish
    def finalize(self, zip_path=None):
        metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.store / "meta").glob("*.json"))]
        keep = ["instance_id", "role", "source", "family", "candidate_id", "base_id", "transformation", "params",
                "title", "target_name", "n_rows", "n_features", "n_unique", "n_missing", "target_kind",
                "label", "label_rule", "check"]
        man = pd.DataFrame([{k: m.get(k) for k in keep} | {"has_doc": bool(m.get("doc"))} for m in metas])
        test = man.role.str.endswith("_test")
        assert man.loc[test, "label"].isna().all(), "a test instance carries a label"
        man.to_csv(self.store / "manifest.csv", index=False)
        print(man.groupby("role").size().to_string())
        print(man[~test].groupby(["role", "label"]).size().unstack(fill_value=0).to_string())
        bad = man[man.check.fillna("ok") != "ok"]
        if len(bad):
            print("integrity notes:\n" + bad[["instance_id", "title", "check"]].to_string(index=False))
        if zip_path:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in self.store.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(self.store.parent))
            print(f"wrote {zip_path}")
        return man


# ---------------------------------------------------------------------- helpers
def plain_csv(df):
    """Re-read a table from its CSV text: what a raw file shows (F2)."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def kind(s):
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def integrity(df, row):
    notes = []
    if pd.notna(row.n_samples) and len(df) != int(row.n_samples):
        notes.append(f"rows {len(df)} vs pool {int(row.n_samples)}")
    if row.target_column not in df.columns:
        return f"target {row.target_column!r} missing"
    nu = df[row.target_column].dropna().nunique()
    if pd.notna(row.n_unique_target) and nu != int(row.n_unique_target):
        notes.append(f"distinct {nu} vs pool {int(row.n_unique_target)}")
    return "; ".join(notes) or "ok"


def add_note(doc, note):
    return (doc + "\n\n" if doc else "") + note


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def fetch(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "task-inference-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def cached(path, url, sleep):
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    time.sleep(sleep)
    return data


def sections(page):
    out = {}
    parts = re.split(r"<h3[^>]*>(.*?)</h3>", page, flags=re.S)
    for name, body in zip(parts[1::2], parts[2::2]):
        body = re.split(r"</main>|</body>", body)[0]
        text = html.unescape(re.sub(r"<[^>]+>", "", body))
        out[re.sub(r"<[^>]+>", "", name).strip()] = re.sub(r"\n{3,}", "\n\n", text).strip()
    return out


def doc_text(sec):
    """Documentation a model may see: no Usage, no Examples (the label source)."""
    keep = [f"{k}:\n{sec[k]}" for k in ("Description", "Format", "Details", "Source") if sec.get(k)]
    return "\n\n".join(keep) or None


MODEL_FN = re.compile(r"(?<![A-Za-z0-9_.])(?:[A-Za-z0-9.]+::)?"
                      r"(glm\.nb|glmmTMB|glmer|glm|gam|bam|polr|clmm|clm|vglm|vgam|multinom|zeroinfl|hurdle|nbinomial)\s*\(")
FAMILY_TASK = {"poisson": "COUNT_REGRESSION", "quasipoisson": "COUNT_REGRESSION", "negative.binomial": "COUNT_REGRESSION",
               "nbinom1": "COUNT_REGRESSION", "nbinom2": "COUNT_REGRESSION", "genpois": "COUNT_REGRESSION",
               "poissonff": "COUNT_REGRESSION", "negbinomial": "COUNT_REGRESSION",
               "binomial": "BINARY", "quasibinomial": "BINARY", "binomialff": "BINARY",
               "Gamma": "CONTINUOUS_REGRESSION", "inverse.gaussian": "CONTINUOUS_REGRESSION",
               "cumulative": "ORDINAL", "propodds": "ORDINAL", "acat": "ORDINAL", "sratio": "ORDINAL", "cratio": "ORDINAL",
               "multinomial": "MULTICLASS"}
# positions of family= and data= when they are passed without a name
FAMILY_POS = {"glm": 1, "gam": 1, "bam": 1, "vglm": 1, "vgam": 1, "glmer": 2, "glmmTMB": 2}
DATA_POS = {"glm": 2, "gam": 2, "bam": 2, "vglm": 2, "vgam": 2, "glmer": 1, "glmmTMB": 1, "polr": 1, "multinom": 1,
            "glm.nb": 1, "zeroinfl": 1, "hurdle": 1, "nbinomial": 1}
FIXED_TASK = {"glm.nb": "COUNT_REGRESSION", "nbinomial": "COUNT_REGRESSION", "zeroinfl": "COUNT_REGRESSION",
              "hurdle": "COUNT_REGRESSION", "polr": "ORDINAL", "clm": "ORDINAL", "clmm": "ORDINAL", "multinom": "MULTICLASS"}


def split_args(s):
    args, depth, cur, quote = [], 0, "", None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def model_calls(code, item):
    """Every model call in an Examples section with its response, label and skip reason (RULES.md R6)."""
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    out = []
    for m in MODEL_FN.finditer(code):
        fn = m.group(1)
        i, depth = m.end(), 1
        while i < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[i], 0)
            i += 1
        args = split_args(code[m.end():i - 1])
        is_named = [bool(re.match(r"^[A-Za-z.]+\s*=(?!=)", a)) for a in args]
        named = {a.split("=", 1)[0].strip(): a.split("=", 1)[1].strip() for a, n in zip(args, is_named) if n}
        positional = [a for a, n in zip(args, is_named) if not n]
        if "family" not in named and fn in FAMILY_POS and len(positional) > FAMILY_POS[fn]:
            named["family"] = positional[FAMILY_POS[fn]]
        if "data" not in named and fn in DATA_POS and len(positional) > DATA_POS[fn]:
            named["data"] = positional[DATA_POS[fn]]
        formula = named.get("formula") or next((a for a in positional if "~" in a), "")
        lhs = formula.split("~")[0].strip() if "~" in formula else ""
        fam = re.sub(r"[\"'\s]", "", named.get("family", "")).split("(")[0].split("::")[-1]
        label = FIXED_TASK.get(fn) or FAMILY_TASK.get(fam)
        skip = None
        if not re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", lhs or "-"):
            skip = f"response not a plain column: {lhs[:40]!r}"
        elif lhs == "Freq":
            skip = "contingency-table response Freq"
        elif "weights" in named:
            skip = "weights= (aggregated table)"
        elif named.get("data", "").strip() != item:
            skip = f"data= is not this dataset ({named.get('data', '')[:30]!r})"
        elif not label:
            skip = f"no task for {fn} family {fam!r}"
        out.append({"fn": fn, "lhs": lhs, "family": fam, "label": label, "skip": skip})
    return out


def check_response(df, col, label, min_rows):
    if col not in df.columns:
        return f"response {col!r} not in CSV"
    y = df[col].dropna()
    if len(y) < min_rows:
        return f"only {len(y)} non-missing responses"
    if label == "COUNT_REGRESSION":
        if not pd.api.types.is_numeric_dtype(y) or (y < 0).any() or not np.all(np.mod(y, 1) == 0):
            return "count response not non-negative whole numbers"
    if label == "BINARY" and y.nunique() > 2:
        return "binomial response with more than two values"
    return "built"
