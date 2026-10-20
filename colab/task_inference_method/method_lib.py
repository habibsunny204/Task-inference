"""Method bundle: uncertainty-aware task inference on the instance store (no LLM inside; fixed 2026-10-05).

Pipeline
  1. features per information condition (nested; same instance, more evidence):
       C0  value statistics of the target column (+ words inside string target values)
       C1  C0 + target/feature-name lexicon + table-context statistics from the 200-row sample
       C2  C1 + documentation/title lexicon (whole text and a window around the target name)
  2. train on silver_train + groupc2_train (labels from RULES.md), one model per condition.
     Weights balance natural vs transformed instances and the five tasks.
  3. 5-fold CV grouped by family → out-of-fold probabilities → temperature scaling (one T per condition) →
     split-conformal LAC threshold on the calibrated OOF probabilities of the **natural** silver instances, α ∈ cfg.alphas.
  4. refit on all training instances; predict the test instances (gold I###, Group C v1 G###) under C0/C1/C2.
     Output is in the baseline format, so `scripts/score_reference.py` scores it against the frozen reference locally.

Systems written to results/predictions.csv
  method:conformal_a{α}   calibrated HGB + LAC set at level α (set size > 1 = ask the user)
  method:forced           calibrated HGB, argmax only
  ml:logreg, ml:rf        classical-ML baselines on the same features, argmax only
  method:natural_only     ablation: HGB + conformal (primary α) trained without Group C v2

Test labels are never read here.
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import lexicons as LX

TASKS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]
CONDITIONS = ("C0", "C1", "C2")
MAX_DOC_CHARS = 5000  # as in the annotation packets
ANON = re.compile(r"^(x|v|f|var|feature|att|attr|attribute|col|column|a|c)[_\-]?\d+$|^\d+$", re.I)


# ============================================================== store
class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = pd.read_csv(self.path / "manifest.csv")

    def meta(self, iid):
        return json.loads((self.path / "meta" / f"{iid}.json").read_text(encoding="utf-8"))

    def target(self, iid):
        return pd.read_parquet(self.path / "targets" / f"{iid}.parquet").y

    def sample(self, iid):
        return pd.read_csv(self.path / "samples" / f"{iid}.csv")


# ============================================================== text helpers
def tokens(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return re.findall(r"[a-z]+|[0-9]+", s.lower())


def lex_hits(toks, words):
    ws = set(words)
    return sum(t in ws for t in toks)


def phrase_count(text, phrases):
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", text)) for p in phrases)


# ============================================================== features
def value_features(meta, y):
    """C0: what the target values alone show."""
    f = {}
    n = max(meta["n_rows"] - meta["n_missing"], 1)
    k = meta["n_unique"]
    f["log_rows"] = math.log10(meta["n_rows"] + 1)
    f["log_unique"] = math.log10(k + 1)
    f["unique_ratio"] = k / n
    f["missing_frac"] = meta["n_missing"] / max(meta["n_rows"], 1)
    for t in (2, 3, 5, 10, 30, 100):
        f[f"k_le_{t}"] = float(k <= t)
    kind = meta["target_kind"]
    f["is_string"], f["is_bool"] = float(kind == "string"), float(kind == "bool")
    counts = np.array([c for _, c in meta["top_counts"]], float)
    if len(counts) > 1:
        p = counts / counts.sum()
        f["entropy_norm"] = float(-(p * np.log(p)).sum() / np.log(len(p)))
        f["count_cv"] = float(counts.std() / counts.mean())
        f["minority_share"] = float(counts.min() / counts.sum())
        f["majority_share"] = float(counts.max() / counts.sum())
    yv = y.dropna()
    num = pd.to_numeric(yv.astype("string") if kind == "string" else yv.astype(float), errors="coerce")
    f["frac_numeric_parse"] = float(num.notna().mean()) if len(num) else 0.0
    if kind == "string":
        levels = [str(v).lower() for v, _ in meta["top_counts"]]
        ltoks = [tokens(v) for v in levels]
        f["str_mean_len"] = float(np.mean([len(v) for v in levels])) if levels else 0.0
        f["str_levels_with_digits"] = float(np.mean([bool(re.search(r"\d", v)) for v in levels])) if levels else 0.0
        for g, words in LX.VALUE.items():
            f[f"val_lex_{g}"] = float(np.mean([lex_hits(t, words) > 0 for t in ltoks])) if ltoks else 0.0
    x = num.dropna().to_numpy(float) if f["frac_numeric_parse"] > 0.99 else np.array([])
    f["numeric"] = float(len(x) > 0)
    if len(x):
        whole = np.mod(x, 1) == 0
        f["integer_valued"] = float(whole.all())
        f["frac_whole"] = float(whole.mean())
        f["has_negative"] = float((x < 0).any())
        f["frac_zero"] = float((x == 0).mean())
        lo, hi = x.min(), x.max()
        f["min_slog"], f["max_slog"] = float(np.sign(lo) * np.log1p(abs(lo))), float(np.sign(hi) * np.log1p(abs(hi)))
        f["range_log"] = float(np.log1p(hi - lo))
        f["bounded01"] = float(lo >= 0 and hi <= 1)
        f["zero_to_100"] = float(lo >= 0 and hi <= 100 and hi > 10)
        sd, mu = x.std(), x.mean()
        f["cv"] = float(sd / abs(mu)) if mu else 0.0
        f["skew"] = float(stats.skew(x)) if sd > 0 else 0.0
        f["kurt"] = float(stats.kurtosis(x)) if sd > 0 else 0.0
        f["dispersion"] = float(x.var() / mu) if (lo >= 0 and mu > 0) else -1.0
        f["poisson_zero_gap"] = float((x == 0).mean() - math.exp(-mu)) if (lo >= 0 and mu > 0 and mu < 50) else 0.0
        f["frac_mult5"] = float((np.mod(x, 5) == 0).mean())
        f["frac_mult10"] = float((np.mod(x, 10) == 0).mean())
        u = np.unique(x)
        f["contiguous_int"] = float(whole.all() and (hi - lo + 1) == len(u))
        f["starts_at_0"], f["starts_at_1"] = float(lo == 0), float(lo == 1)
        if len(u) > 1:
            d = np.diff(u)
            f["step_regular"] = float(d.std() / d.mean()) if d.mean() else 0.0
            f["grid_steps"] = float(np.log1p((hi - lo) / d.min())) if d.min() > 0 else 0.0
        vc = pd.Series(x).value_counts()
        f["mode_is_min"] = float(vc.idxmax() == lo)
        if 2 < len(u) <= 50:
            byv = vc.sort_index()
            f["freq_value_spearman"] = float(stats.spearmanr(byv.index, byv.values)[0]) if byv.values.std() else 0.0
            f["top_uptick"] = float(byv.values[-1] / byv.values[-2]) if byv.values[-2] else 0.0
    return f


def name_features(meta, sample):
    """C1: names and table context."""
    f = {}
    tt = tokens(meta["target_name"])
    for g, words in LX.NAME.items():
        f[f"name_lex_{g}"] = float(lex_hits(tt, words))
    f["name_anonymous"] = float(bool(ANON.match(str(meta["target_name"]))) or meta["target_name"].lower() in ("y", "target"))
    f["name_len_tokens"] = float(len(tt))
    feats = meta["feature_names"]
    f["log_features"] = math.log10(len(feats) + 1)
    kinds = meta["feature_kinds"]
    if kinds:
        f["feat_frac_numeric"] = float(np.mean([k in ("int", "float") for k in kinds]))
        f["feat_frac_string"] = float(np.mean([k == "string" for k in kinds]))
        f["feat_frac_anonymous"] = float(np.mean([bool(ANON.match(str(c))) for c in feats]))
        ftoks = [tokens(c) for c in feats[:500]]
        for g in ("count", "ordinal", "nominal", "continuous"):
            f[f"feat_lex_{g}"] = float(np.mean([lex_hits(t, LX.NAME[g]) > 0 for t in ftoks]))
    t = meta["target_name"]
    if t in sample.columns:
        yt = pd.to_numeric(sample[t], errors="coerce")
        best = 0.0
        if yt.notna().mean() > 0.9 and yt.std() > 0:
            for c in sample.columns:
                if c == t:
                    continue
                xc = pd.to_numeric(sample[c], errors="coerce")
                if xc.notna().mean() > 0.9 and xc.std() > 0:
                    r = abs(stats.spearmanr(xc, yt, nan_policy="omit")[0])
                    best = max(best, 0.0 if np.isnan(r) else r)
        f["max_abs_spearman_feature"] = best
    return f


def doc_features(meta):
    """C2: title and documentation (truncated as in the packets)."""
    f = {}
    doc = (meta.get("doc") or "")[:MAX_DOC_CHARS].lower()
    title = str(meta.get("title") or "").lower()
    f["has_doc"] = float(bool(doc.strip()))
    f["doc_log_chars"] = math.log10(len(doc) + 1)
    text = title + "\n" + doc
    for g, phrases in LX.DOC.items():
        f[f"doc_lex_{g}"] = math.log1p(phrase_count(text, phrases))
    tname = str(meta["target_name"]).lower()
    win = ""
    if len(tname) > 1:
        for m in re.finditer(re.escape(tname), doc):
            win += doc[max(0, m.start() - 200): m.end() + 200] + "\n"
    f["doc_mentions_target"] = float(bool(win))
    for g in ("count", "ordinal", "nominal", "continuous", "binary"):
        f[f"win_lex_{g}"] = math.log1p(phrase_count(win, LX.DOC[g]))
    tt = tokens(title)
    for g in ("count", "ordinal", "nominal", "continuous", "binary"):
        f[f"title_lex_{g}"] = float(lex_hits(tt, LX.NAME[g]))
    return f


def build_features(store, ids):
    rows, t0 = [], time.time()
    for iid in ids:
        m = store.meta(iid)
        v = value_features(m, store.target(iid))
        nm = name_features(m, store.sample(iid))
        d = doc_features(m)
        rows.append({"instance_id": iid, **{f"v_{k}": x for k, x in v.items()},
                     **{f"n_{k}": x for k, x in nm.items()}, **{f"d_{k}": x for k, x in d.items()}})
    df = pd.DataFrame(rows).set_index("instance_id")
    return df, (time.time() - t0) / max(len(ids), 1)


def condition_columns(cols, cond):
    pre = {"C0": ("v_",), "C1": ("v_", "n_"), "C2": ("v_", "n_", "d_")}[cond]
    return [c for c in cols if c.startswith(pre)]


# ============================================================== models
def make_model(name, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                              l2_regularization=1.0, random_state=seed)
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000))
    raise ValueError(name)


def full_proba(model, X):
    """Probabilities over all five tasks (a class missing from training gets 0)."""
    p = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    out = np.zeros((len(X), len(TASKS)))
    for j, c in enumerate(classes):
        out[:, TASKS.index(c)] = p[:, j]
    return out


def sample_weights(train):
    w = np.where(train.role == "silver_train", 1.0, (train.role == "silver_train").sum() / max((train.role != "silver_train").sum(), 1))
    cls = train.label.map(train.label.value_counts())
    w = w * (len(train) / (len(TASKS) * cls.to_numpy()))
    return w / w.mean()


def fit(name, X, y, w, seed):
    m = make_model(name, seed)
    if name == "hgb":
        m.fit(X, y, sample_weight=w)
    else:
        m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m


def oof_proba(name, X, y, w, groups, seed, n_folds):
    from sklearn.model_selection import GroupKFold
    p = np.zeros((len(X), len(TASKS)))
    for tr, te in GroupKFold(n_splits=min(n_folds, len(set(groups)))).split(X, y, groups):
        p[te] = full_proba(fit(name, X.iloc[tr], y.iloc[tr], w[tr], seed), X.iloc[te])
    return p


def temperature(p, y_idx):
    logp = np.log(np.clip(p, 1e-9, 1))

    def nll(t):
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(q[np.arange(len(y_idx)), y_idx], 1e-12, 1)))
    return float(optimize.minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p, t):
    z = np.log(np.clip(p, 1e-9, 1)) / t
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z) * (p > 0)
    return q / q.sum(axis=1, keepdims=True)


def lac_threshold(p, y_idx, alpha):
    s = 1 - p[np.arange(len(y_idx)), y_idx]
    n = len(s)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def lac_sets(p, qhat):
    sets = []
    for row in p:
        s = [TASKS[j] for j in np.argsort(-row) if 1 - row[j] <= qhat]
        sets.append(s or [TASKS[int(row.argmax())]])
    return sets


# ============================================================== run
def run(store_path, out, cfg, smoke=False):
    out = Path(out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    store = Store(store_path)
    man = store.manifest
    if smoke:
        man = pd.concat([man[man.role.str.endswith("_train")].groupby("label").head(15), man[man.role.str.endswith("_test")].head(6)])
    feats, sec_per_instance = build_features(store, man.instance_id)
    feats.to_csv(out / "features.csv")
    train = man[man.role.isin(["silver_train", "groupc2_train"]) & man.label.isin(TASKS)].set_index("instance_id")
    test = man[man.role.isin(["gold_test", "groupc1_test"])].set_index("instance_id")
    y = train.label
    y_idx = y.map(TASKS.index).to_numpy()
    w = sample_weights(train.reset_index())
    groups = train.family.astype(str).to_numpy()
    natural = (train.role == "silver_train").to_numpy()
    seed, alphas, primary = cfg["seed"], cfg["alphas"], cfg["primary_alpha"]
    preds, diag, calib = [], [], {}
    for cond in CONDITIONS:
        cols = condition_columns(feats.columns, cond)
        Xtr, Xte = feats.loc[train.index, cols], feats.loc[test.index, cols]
        for name in ("hgb", "logreg", "rf"):
            t0 = time.time()
            oof = oof_proba(name, Xtr, y, w, groups, seed, cfg["n_folds"])
            model = fit(name, Xtr, y, w, seed)
            pte = full_proba(model, Xte)
            for part, mask in (("natural", natural), ("transformed", ~natural)):
                if mask.any():
                    pred = oof[mask].argmax(1)
                    diag.append({"condition": cond, "learner": name, "oof_part": part, "n": int(mask.sum()),
                                 "oof_accuracy": float((pred == y_idx[mask]).mean()),
                                 "oof_macro_recall": float(np.mean([(pred[y_idx[mask] == k] == k).mean()
                                                                    for k in np.unique(y_idx[mask])])),
                                 "fit_seconds": time.time() - t0})
            if name != "hgb":
                preds += rows_for(test.index, test, cond, f"ml:{name}", pte, [[TASKS[i]] for i in pte.argmax(1)])
                continue
            T = temperature(oof[natural], y_idx[natural])
            oof_c, pte_c = apply_temperature(oof, T), apply_temperature(pte, T)
            preds += rows_for(test.index, test, cond, "method:forced", pte_c, [[TASKS[i]] for i in pte_c.argmax(1)])
            for a in alphas:
                q = lac_threshold(oof_c[natural], y_idx[natural], a)
                sets_oof = lac_sets(oof_c[natural], q)
                calib[(cond, a)] = {"condition": cond, "alpha": a, "temperature": T, "qhat": q,
                                    "oof_natural_coverage": float(np.mean([TASKS[k] in s for k, s in zip(y_idx[natural], sets_oof)])),
                                    "oof_natural_mean_set": float(np.mean([len(s) for s in sets_oof]))}
                preds += rows_for(test.index, test, cond, f"method:conformal_a{int(round(a * 100)):02d}", pte_c, lac_sets(pte_c, q))
            if cond == "C2" or not smoke:
                imp = permutation_importance_oof(Xtr, y, w, groups, seed, cfg)
                imp.to_csv(out / "diagnostics" / f"importance_{cond}.csv")
        # ablation: no Group C v2 in training
        nat = train.index[natural]
        if len(set(groups[natural])) >= 2:
            oof_n = oof_proba("hgb", Xtr.loc[nat], y.loc[nat], w[natural], groups[natural], seed, cfg["n_folds"])
            mdl = fit("hgb", Xtr.loc[nat], y.loc[nat], w[natural], seed)
            Tn = temperature(oof_n, y_idx[natural])
            qn = lac_threshold(apply_temperature(oof_n, Tn), y_idx[natural], primary)
            pn = apply_temperature(full_proba(mdl, Xte), Tn)
            preds += rows_for(test.index, test, cond, "method:natural_only", pn, lac_sets(pn, qn))
        print(f"{cond}: done", flush=True)
    pred = pd.DataFrame(preds)
    pred["seconds"] = sec_per_instance
    pred.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics" / "oof_training.csv", index=False)
    pd.DataFrame(calib.values()).to_csv(out / "diagnostics" / "conformal_calibration.csv", index=False)
    summary = {"n_train": int(len(train)), "n_train_natural": int(natural.sum()), "n_test": int(len(test)),
               "train_labels": train.label.value_counts().to_dict(), "families": int(len(set(groups))),
               "feature_seconds_per_instance": sec_per_instance, "n_features": {c: len(condition_columns(feats.columns, c)) for c in CONDITIONS}}
    (out / "diagnostics" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(diag).pivot_table(index=["condition", "learner"], columns="oof_part", values="oof_accuracy").round(3))
    print(pd.DataFrame(calib.values()).round(3).to_string(index=False))
    return pred


def rows_for(ids, test, cond, system, p, sets):
    out = []
    for iid, row, s in zip(ids, p, sets):
        out.append({"instance_id": iid, "group": "pilot" if test.at[iid, "role"] == "gold_test" else "group_c",
                    "condition": cond, "system": system, "parse_ok": True,
                    "plausible_task_set": "|".join(sorted(s)), "most_likely_task": TASKS[int(row.argmax())],
                    "ask": len(s) > 1, **{f"p_{t}": float(v) for t, v in zip(TASKS, row)}})
    return out


def permutation_importance_oof(X, y, w, groups, seed, cfg):
    """Mean drop in OOF log-loss when one feature column is shuffled (grouped CV, one repeat)."""
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(seed)
    y_idx = y.map(TASKS.index).to_numpy()
    drops = {c: [] for c in X.columns}
    for tr, te in GroupKFold(n_splits=min(cfg["n_folds"], len(set(groups)))).split(X, y, groups):
        m = fit("hgb", X.iloc[tr], y.iloc[tr], w[tr], seed)
        base = log_loss(full_proba(m, X.iloc[te]), y_idx[te])
        for c in X.columns:
            Xp = X.iloc[te].copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops[c].append(log_loss(full_proba(m, Xp), y_idx[te]) - base)
    return pd.Series({c: float(np.mean(v)) for c, v in drops.items()}, name="logloss_increase").sort_values(ascending=False)


def log_loss(p, y_idx):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, 1))))


# ============================================================== analysis run v1.1 (post-hoc; declared in findings_log 2026-10-13)
def mass_set(row, mass):
    """Smallest set of tasks reaching `mass` calibrated probability, at least two (used when asking)."""
    order = np.argsort(-row)
    out, total = [], 0.0
    for j in order:
        out.append(TASKS[j])
        total += row[j]
        if total >= mass and len(out) >= 2:
            break
    return out


def run_v11(store_path, out, cfg, features_csv=None):
    """(1) H6 source hold-out with the v1 learner; (2) v2 = HGB+RF+LR ensemble with silver-only ask thresholds."""
    out = Path(out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    store = Store(store_path)
    man = store.manifest
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv, index_col=0)
    else:
        feats, _ = build_features(store, man.instance_id)
    train = man[man.role.isin(["silver_train", "groupc2_train"]) & man.label.isin(TASKS)].set_index("instance_id")
    test = man[man.role.isin(["gold_test", "groupc1_test"])].set_index("instance_id")
    src = man.set_index("instance_id").source
    eff = train.source.where(train.role == "silver_train", train.base_id.map(src)).fillna("").astype(str)
    holdouts = {"no_tocuco": ~eff.str.startswith("TOC-UCO"), "no_rdatasets": ~eff.str.startswith("Rdatasets"),
                "no_groupc2": train.role == "silver_train"}
    seed = cfg["seed"]
    y = train.label
    y_idx = y.map(TASKS.index).to_numpy()
    w = sample_weights(train.reset_index())
    groups = train.family.astype(str).to_numpy()
    natural = (train.role == "silver_train").to_numpy()
    preds, diag, calib = [], [], []
    for cond in CONDITIONS:
        cols = condition_columns(feats.columns, cond)
        Xtr, Xte = feats.loc[train.index, cols], feats.loc[test.index, cols]
        for name, mask in holdouts.items():
            tr = train[mask.to_numpy()]
            m = fit("hgb", feats.loc[tr.index, cols], tr.label, sample_weights(tr.reset_index()), seed)
            p = full_proba(m, Xte)
            preds += rows_for(test.index, test, cond, f"heldout:{name}", p, [[TASKS[i]] for i in p.argmax(1)])
            diag.append({"condition": cond, "variant": f"heldout:{name}", "n_train": int(len(tr)),
                         "train_labels": json.dumps(tr.label.value_counts().to_dict())})
        oofs, ptes = [], []
        for name in ("hgb", "rf", "logreg"):
            oofs.append(oof_proba(name, Xtr, y, w, groups, seed, cfg["n_folds"]))
            ptes.append(full_proba(fit(name, Xtr, y, w, seed), Xte))
        oof, pte = np.mean(oofs, axis=0), np.mean(ptes, axis=0)
        T = temperature(oof[natural], y_idx[natural])
        oof_c, pte_c = apply_temperature(oof, T), apply_temperature(pte, T)
        for part, mask in (("natural", natural), ("transformed", ~natural)):
            diag.append({"condition": cond, "variant": f"v2:ensemble OOF {part}", "n_train": int(mask.sum()),
                         "oof_accuracy": float((oof_c[mask].argmax(1) == y_idx[mask]).mean())})
        preds += rows_for(test.index, test, cond, "v2:forced", pte_c, [[TASKS[i]] for i in pte_c.argmax(1)])
        for a in cfg["alphas"]:
            q = lac_threshold(oof_c[natural], y_idx[natural], a)
            preds += rows_for(test.index, test, cond, f"v2:conformal_a{int(round(a * 100)):02d}", pte_c, lac_sets(pte_c, q))
            calib.append({"condition": cond, "rule": f"conformal_a{int(round(a * 100)):02d}", "temperature": T, "threshold": q})
        conf_sil = oof_c[natural].max(1)
        for b in cfg["ask_budgets"]:
            thr = float(np.quantile(conf_sil, b))
            sets = [mass_set(r, cfg["ask_mass"]) if r.max() < thr else [TASKS[int(r.argmax())]] for r in pte_c]
            preds += rows_for(test.index, test, cond, f"v2:ask{int(round(b * 100)):02d}", pte_c, sets)
            calib.append({"condition": cond, "rule": f"ask{int(round(b * 100)):02d}", "temperature": T, "threshold": thr,
                          "silver_ask_rate": float((conf_sil < thr).mean())})
        print(f"{cond}: done", flush=True)
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics" / "variants.csv", index=False)
    pd.DataFrame(calib).to_csv(out / "diagnostics" / "ask_calibration.csv", index=False)
    print(pd.DataFrame(diag).to_string(index=False))
    print(pd.DataFrame(calib).round(3).to_string(index=False))
