"""TIA and the rule baselines on MLTaskBench (external check, post hoc; declared in docs/findings_log.md as F39 on
2026-11-10, before any MLTaskBench table was read).

Stages (each one writes its files as it goes, so a rerun skips finished work):
  1. fetch     the 319 tabular CSV files of MLTaskBench v2 (doi:10.5281/zenodo.21649607, CC BY 4.0), read from dataset.zip
               by HTTP range requests (about 108 MB of 4.2 GB); zip checks every file's CRC. If range requests fail, the
               whole file is downloaded and its md5 checked.
  2. store     one store entry per target, written by the data bundle's Bundle.write (data_lib.py, unchanged): title,
               documentation, target name, feature names and kinds, target statistics, target column, 200-row sample.
  3. features  method_lib.build_features (unchanged) -> features_mltaskbench.csv, with the frozen feature columns.
  4. rules     the four frozen rule baselines (rules_lib.py: source copied unchanged, checked by the builder).
  5. tia       TIA-v1 (HGB) and TIA-v2 (HGB + RF + LR), retrained on the frozen features with robustness_lib.calibrated
               (unchanged), calibrated, argmax. The 76 benchmark and Group C test instances are predicted with them and must
               reproduce v2:forced (TIA-v2) and the robustness re-run rerun:v1 (TIA-v1) to within 1e-9; otherwise the run
               stops before any MLTaskBench prediction is written.

MLTaskBench labels are never read: the file path inside the zip names the task folder, and it is used only to read the file.
Outputs (results/): predictions.csv, load_log.csv, features_mltaskbench.csv, fidelity.csv, report.json.
"""
import hashlib
import json
import platform
import time
import types
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import data_lib
import method_lib as ml
import robustness_lib as rb
import rules_lib as rl

TASKS = ml.TASKS
UA = {"User-Agent": "task-inference-research/1.0"}
BLOCK = 4 << 20


def versions():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def check_sklearn(cfg):
    import sklearn
    if sklearn.__version__ != cfg["sklearn"]:
        raise RuntimeError(f"scikit-learn {sklearn.__version__} is loaded, but TIA was trained with {cfg['sklearn']}. "
                           f"Install it (pip install scikit-learn=={cfg['sklearn']}), restart the kernel and run again.")


def load(work):
    work = Path(work)
    cfg = json.loads((work / "config.json").read_text())
    targets = pd.read_csv(work / "inputs" / "targets.csv", keep_default_na=False, na_values=[""])
    return cfg, targets


# ====================================================================== 1. fetch
class HttpFile:
    """A read-only, seekable view of a remote file through HTTP range requests (4 MB blocks, a few kept)."""

    def __init__(self, url):
        self.pos, self.fetched, self.cache = 0, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.geturl()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo, hi = b * BLOCK, min((b + 1) * BLOCK, self.size) - 1
            for attempt in range(4):
                try:
                    req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        if r.status != 206:
                            raise OSError(f"no range support (HTTP {r.status})")
                        data = r.read()
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    if attempt == 3:
                        raise
                    print(f"  retry block {b} ({e})", flush=True)
                    time.sleep(5 * (attempt + 1))
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[b] = data
            self.fetched += len(data)
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b, o = divmod(self.pos, BLOCK)
            chunk = self._block(b)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def download_whole(url, path, md5_expected):
    h = hashlib.md5()
    tmp = Path(str(path) + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(16 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
    if "md5:" + h.hexdigest() != md5_expected:
        raise IOError("dataset.zip: md5 does not match the Zenodo record")
    tmp.replace(path)


def fetch(work, cfg, targets, only=None):
    """Write each target's CSV to work/tables/<instance_id>.csv. Returns a dict with the mode and bytes read."""
    work = Path(work)
    out = work / "tables"
    out.mkdir(parents=True, exist_ok=True)
    todo = targets[[(only is None or i in only) and not (out / f"{i}.csv").exists() for i in targets.instance_id]]
    info = {"mode": "none needed", "bytes_read": 0, "files": int(len(todo))}
    if not len(todo):
        return info
    t0 = time.time()
    try:
        src = HttpFile(cfg["zip_url"])
        if src.size != cfg["zip_size"]:
            raise IOError(f"dataset.zip has {src.size} bytes, the record says {cfg['zip_size']}")
        z = zipfile.ZipFile(src)
        info["mode"] = "range requests"
    except Exception as e:  # noqa: BLE001
        print(f"range requests failed ({type(e).__name__}: {e}); downloading the whole file (4.2 GB)", flush=True)
        path = work / "dataset.zip"
        if not path.exists():
            download_whole(cfg["zip_url"], path, cfg["zip_md5"])
        z, src = zipfile.ZipFile(path), None
        info["mode"] = "whole file"
    for k, r in enumerate(todo.sort_values("header_offset").itertuples()):
        data = z.read(r.member)  # raises on a CRC mismatch
        assert f"{zlib.crc32(data) & 0xffffffff:08x}" == r.crc32, f"{r.member}: CRC differs from the planned list"
        part = out / f"{r.instance_id}.csv.part"
        part.write_bytes(data)
        part.replace(out / f"{r.instance_id}.csv")
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(todo)} files", flush=True)
    info["bytes_read"] = int(src.fetched) if src is not None else int(cfg["zip_size"])
    info["seconds"] = round(time.time() - t0, 1)
    print(f"fetched {len(todo)} files by {info['mode']} ({info['bytes_read'] / 1e6:.0f} MB, {info['seconds']:.0f} s)",
          flush=True)
    return info


# ====================================================================== 2. store
def _read(path, **kw):
    try:
        return pd.read_csv(path, low_memory=False, **kw), "utf-8"
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1", **kw), "latin-1"


def _clean(name):
    return str(name).strip().strip("\"'").strip()


def load_table(path, target):
    """(table, target column, reader) by the declared order of attempts; raises if the target is not found."""
    tried = []
    for name, kw in (("default", {}), ("sniffed", {"sep": None, "engine": "python"}), ("semicolon", {"sep": ";"})):
        try:
            df, enc = _read(path, **kw)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__}")
            continue
        if target in df.columns:
            return df, target, f"{name}/{enc}"
        cleaned = {}
        for c in df.columns:
            cleaned.setdefault(_clean(c), c)
        if _clean(target) in cleaned:
            return df, cleaned[_clean(target)], f"{name}/{enc}/stripped"
        tried.append(f"{name}: target not among {len(df.columns)} columns")
    raise KeyError("; ".join(tried))


def build_store(work, cfg, targets, only=None):
    """One store entry per readable target (data_lib.Bundle.write, unchanged). Returns the load log."""
    work = Path(work)
    store = work / "store"
    for d in ("meta", "targets", "samples"):
        (store / d).mkdir(parents=True, exist_ok=True)
    writer = types.SimpleNamespace(cfg=cfg["store"], store=store)
    log_path = work / "results" / "load_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    done = set(log.instance_id) if len(log) else set()
    rows = []
    for r in targets.itertuples():
        if (only is not None and r.instance_id not in only) or r.instance_id in done:
            continue
        t0 = time.time()
        rec = {"instance_id": r.instance_id, "dataset_name": r.dataset_name, "target_variable": r.target_variable}
        try:
            df, tcol, reader = load_table(work / "tables" / f"{r.instance_id}.csv", r.target_variable)
            doc = r.dataset_description if isinstance(r.dataset_description, str) and r.dataset_description.strip() else None
            data_lib.Bundle.write(writer, r.instance_id, df, tcol, r.dataset_name, doc,
                                  {"role": "mltaskbench", "source": "MLTaskBench v2", "family": r.dataset_name,
                                   "did": int(r.did)})
            rec.update(status="ok", reader=reader, target_column=tcol, n_rows=len(df), n_columns=df.shape[1],
                       n_unique=int(df[tcol].nunique()), target_kind=data_lib.kind(df[tcol]), has_doc=doc is not None)
        except Exception as e:  # noqa: BLE001
            rec.update(status=f"failed: {type(e).__name__}: {e}"[:300])
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        print(f"{r.instance_id} {r.dataset_name}: {rec['status']} ({rec['seconds']:.1f} s)", flush=True)
        log = pd.concat([log, pd.DataFrame([rec])], ignore_index=True)
        log.to_csv(log_path, index=False)
    ok = log[log.status == "ok"].instance_id if len(log) else pd.Series(dtype=str)
    pd.DataFrame({"instance_id": sorted(ok)}).to_csv(store / "manifest.csv", index=False)
    return log


# ====================================================================== 3. features
def build_features(work, frozen_columns):
    work = Path(work)
    store = ml.Store(work / "store")
    ids = list(store.manifest.instance_id)
    feats, sec = ml.build_features(store, ids)
    extra = sorted(set(feats.columns) - set(frozen_columns))
    feats = feats.reindex(columns=frozen_columns)
    feats.index.name = "instance_id"
    path = work / "results" / "features_mltaskbench.csv"
    feats.to_csv(path)
    print(f"features: {len(feats)} targets, {sec:.2f} s per target; columns not in the frozen set: {extra or 'none'}",
          flush=True)
    return pd.read_csv(path, index_col=0), {"seconds_per_target": sec, "extra_columns": extra}


# ====================================================================== 4. rules
def run_rules(work, log):
    """The four frozen rule baselines on each readable target, in the baseline format of run_rules.py."""
    work = Path(work)
    rows = []
    for r in log[log.status == "ok"].itertuples():
        df, tcol, _ = load_table(work / "tables" / f"{r.instance_id}.csv", r.target_variable)
        y = df[tcol]
        ag_raw, sk_raw = rl.rule_autogluon(y), rl.rule_sklearn(y)
        ag, sk = rl.AG_MAP.get(ag_raw, "ERROR"), rl.SK_MAP.get(sk_raw, "ERROR")
        hs, hs_ml = rl.heuristic_set(y)
        hn, hn_ml = rl.heuristic_names(y, tcol)
        for c in ml.CONDITIONS:
            out = [("autogluon", {ag}, ag, ag_raw), ("sklearn", {sk}, sk, sk_raw), ("heuristic_set", hs, hs_ml, "")]
            out.append(("heuristic_names", hn, hn_ml, "") if c != "C0" else ("heuristic_names", hs, hs_ml, "no name at C0"))
            for name, s, m, raw in out:
                rows.append({"instance_id": r.instance_id, "condition": c, "system": f"rule:{name}",
                             "plausible_task_set": "|".join(sorted(s)), "most_likely_task": m, "raw_output": raw})
    return pd.DataFrame(rows)


# ====================================================================== 5. TIA
def run_tia(work, cfg, mltb, smoke=False):
    """Retrain TIA-v1 and TIA-v2 per condition; check them on the 76 frozen test instances; predict MLTaskBench."""
    work = Path(work)
    man = pd.read_csv(work / "inputs" / "manifest.csv")
    feats = pd.read_csv(work / "inputs" / "features.csv", index_col=0)
    train = man[man.role.isin(["silver_train", "groupc2_train"]) & man.label.isin(TASKS)]
    test = man[man.role.isin(["gold_test", "groupc1_test"])]
    if smoke:  # a reduced grid for a quick check of the code path; never used for results
        train = train.groupby("label").head(12)
    train, test = train.set_index("instance_id"), test.set_index("instance_id")
    y = train.label
    y_idx = y.map(TASKS.index).to_numpy()
    w = ml.sample_weights(train.reset_index())
    groups = train.family.astype(str).to_numpy()
    natural = (train.role == "silver_train").to_numpy()
    seed, nf = cfg["seed"], cfg["n_folds"]
    rows, temps = [], []
    for cond in ml.CONDITIONS:
        t0 = time.time()
        cols = ml.condition_columns(feats.columns, cond)
        Xtr = feats.loc[train.index, cols]
        Xte = pd.concat([feats.loc[test.index, cols], mltb[cols]])
        cache = {}
        for system, learners in (("tia_v1", ["hgb"]), ("tia_v2", ["hgb", "rf", "logreg"])):
            _, pte, T = rb.calibrated(learners, Xtr, Xte, y, y_idx, w, groups, natural, seed, nf, cache)
            temps.append({"condition": cond, "system": system, "temperature": T})
            for iid, p in zip(Xte.index, pte):
                rows.append({"instance_id": iid, "condition": cond, "system": system,
                             "most_likely_task": TASKS[int(p.argmax())], "plausible_task_set": TASKS[int(p.argmax())],
                             **{f"p_{t}": float(v) for t, v in zip(TASKS, p)}})
        print(f"TIA {cond}: done in {time.time() - t0:.0f} s", flush=True)
    pred = pd.DataFrame(rows)
    # fidelity on the 76 frozen test instances
    ref = pd.read_csv(work / "inputs" / "frozen_check.csv")
    pc = [f"p_{t}" for t in TASKS]
    fid = []
    for system, frozen in (("tia_v1", "rerun:v1"), ("tia_v2", "v2:forced")):
        a = pred[(pred.system == system) & pred.instance_id.isin(test.index)].set_index(["instance_id", "condition"])
        b = ref[ref.system == frozen].set_index(["instance_id", "condition"]).loc[a.index]
        fid.append({"system": system, "reference": frozen, "n": int(len(a)),
                    "max_abs_prob_diff": float(np.abs(a[pc].to_numpy() - b[pc].to_numpy()).max()),
                    "argmax_agreement": float((a.most_likely_task == b.most_likely_task).mean())})
    fid = pd.DataFrame(fid)
    passed = bool((fid.max_abs_prob_diff < cfg["fidelity_tol"]).all())
    return pred[~pred.instance_id.isin(test.index)].reset_index(drop=True), fid, pd.DataFrame(temps), passed


# ====================================================================== all stages
def zip_results(work, dest):
    import shutil
    work = Path(work)
    shutil.copy(work / "config.json", work / "results" / "config.json")
    return shutil.make_archive(str(Path(dest).with_suffix("")), "zip", work / "results")


def run(work, dest_zip=None, only=None, smoke=False):
    work = Path(work)
    cfg, targets = load(work)
    if not smoke:
        check_sklearn(cfg)
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)
    report = {"versions": versions(), "smoke": bool(smoke), "only": sorted(only) if only else None,
              "n_targets": int(len(targets) if only is None else len(only))}
    t0 = time.time()
    report["fetch"] = fetch(work, cfg, targets, only)
    log = build_store(work, cfg, targets, only)
    report["read_ok"] = int((log.status == "ok").sum())
    report["read_failed"] = log.loc[log.status != "ok", ["instance_id", "dataset_name", "status"]].to_dict("records")
    frozen_cols = list(pd.read_csv(work / "inputs" / "features.csv", index_col=0, nrows=1).columns)
    mltb, finfo = build_features(work, frozen_cols)
    report["features"] = finfo
    rules = run_rules(work, log)
    if dest_zip:
        rules.to_csv(res / "predictions.csv", index=False)  # rules first, so a stopped run still returns them
        zip_results(work, dest_zip)
    tia, fid, temps, passed = run_tia(work, cfg, mltb, smoke=smoke)
    fid.to_csv(res / "fidelity.csv", index=False)
    temps.to_csv(res / "temperatures.csv", index=False)
    report["fidelity"] = fid.to_dict("records")
    report["fidelity_passed"] = passed
    report["seconds"] = round(time.time() - t0, 1)
    if passed or smoke:
        pred = pd.concat([tia, rules], ignore_index=True)
        pred.to_csv(res / "predictions.csv", index=False)
        report["n_predictions"] = int(len(pred))
    (res / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if dest_zip:
        zip_results(work, dest_zip)
    print(fid.to_string(index=False))
    if not passed and not smoke:
        raise RuntimeError("TIA retrained here does not reproduce the frozen predictions; no MLTaskBench prediction was "
                           "written. Send report.json (in the results zip).")
    summary(work)
    return report


def summary(work):
    """Counts only: the labels are not in the bundle, so accuracy is computed locally (scripts/score_mltaskbench.py)."""
    res = Path(work) / "results"
    log = pd.read_csv(res / "load_log.csv")
    pred = pd.read_csv(res / "predictions.csv")
    print(f"\nread: {int((log.status == 'ok').sum())} of {len(log)} tables; failed: "
          f"{', '.join(log[log.status != 'ok'].dataset_name) or 'none'}")
    tab = pred.groupby(["system", "condition"]).most_likely_task.value_counts().unstack(fill_value=0)
    print("\ntop-1 task counts per system and condition:\n" + tab.to_string())
