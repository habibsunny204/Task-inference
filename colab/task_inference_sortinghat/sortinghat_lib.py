"""SortingHat baseline (Shah et al., SIGMOD 2021), re-trained from the authors' released data and code. Fixed 2026-10-18.

Why re-trained: the released Random Forest pickle needs scikit-learn 0.22 / Python ≤ 3.8, which do not install on Colab.
We re-train the same model on the same data with the same features and grid
(Source-Code/Models/Random_Forest.ipynb in ML-Data-Prep-Zoo/MLFeatureTypeInference):
  features   25 descriptive statistics of the column + character bigram counts of the column name (useStats=1,
             useAttributeName=1, no sample bigrams)
  model      RandomForestClassifier, grid n_estimators ∈ {5,25,50,75,100,500} × max_depth ∈ {5,10,25,50,100,250},
             chosen on a 75/25 split of the shuffled training set (random_state 100)
  deviation  the best configuration is refitted on the whole training set (the notebook kept a model from its last CV fold)
Fidelity check: accuracy on the authors' held-out test set (their paper: 0.9265 for the Random Forest).

Applied to our test instances (gold I###, Group C v1 G###) from the data store; the target column is featurized with the
authors' FeaturizeFile (copied below). The column name given to the model: "" at C0 (no names shown), the real target name at
C1 and C2 (SortingHat reads no documentation, so C2 = C1). Counts are taken from the full column (store metadata), since the
stored target can be a 50k-row sample.

Mapping from SortingHat's 9 feature types to tasks (fixed before any prediction was seen):
  numeric, embedded-number   → CONTINUOUS_REGRESSION (SortingHat has no count or ordinal type)
  categorical                → BINARY if the column has 2 distinct values, else MULTICLASS
  context-specific           → ask: set {CONTINUOUS_REGRESSION, BINARY/MULTICLASS}, most likely NONE (its abstain-like class)
  datetime, sentence, url, list, not-generalizable → no valid task (counted as wrong, like an unparseable LLM answer)
Systems: `sortinghat` (as above) and `sortinghat:forced` (always commits: p(CONT) = p(numeric) + p(embedded) + ½ p(context),
p(CAT) = p(categorical) + ½ p(context), renormalised over these two).

Run order (2026-10-18 fix): NLTK data and the test featurization come first (seconds), so a missing resource or a bad
input fails before the 10–20 min grid search, not after it. Library versions are written to report.json.
"""
import json
import platform
import re
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]
TYPES = ["numeric", "categorical", "datetime", "sentence", "url", "embedded-number", "list", "not-generalizable",
         "context-specific"]
STATS = ["total_vals", "num_nans", "%_nans", "num_of_dist_val", "%_dist_val", "mean", "std_dev", "min_val", "max_val",
         "has_delimiters", "has_url", "has_email", "has_date", "mean_word_count", "std_dev_word_count",
         "mean_stopword_total", "stdev_stopword_total", "mean_char_count", "stdev_char_count", "mean_whitespace_count",
         "stdev_whitespace_count", "mean_delim_count", "stdev_delim_count", "is_list", "is_long_sentence"]


# ============================================================== featurization
# Copied from ML-Data-Prep-Zoo/MLFeatureTypeInference/Library/sortinghat/pylib.py
# Copyright 2020 Vraj Shah, Arun Kumar. Licensed under the Apache License, Version 2.0.
# Changes: pd.Timestamp failures other than ValueError are also caught; new columns are set with .loc (pandas 2);
# the NLTK resources are loaded lazily.
del_reg = re.compile(r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}')
delimeters = re.compile(r"(,|;|\|)")
url_reg = re.compile(r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?")
email_reg = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b")
_NLTK = {}


def _nltk():
    if not _NLTK:
        import nltk
        for res in ("stopwords", "punkt", "punkt_tab"):
            try:
                nltk.download(res, quiet=True)
            except Exception:  # noqa: BLE001
                pass
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        try:
            stop = set(stopwords.words("english"))
            word_tokenize("check the tokenizer")
        except LookupError as e:
            raise RuntimeError("NLTK data (stopwords, punkt_tab) could not be downloaded. The runtime needs internet "
                               "access: reconnect and run all cells again.") from e
        _NLTK["stop"] = stop
        _NLTK["tok"] = word_tokenize
    return _NLTK["stop"], _NLTK["tok"]


def summary_stats(dat, key_s):
    from pandas.api.types import is_numeric_dtype
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        total_val = len(dat[col])
        mean = std_dev = min_val = max_val = 0
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])
            if pd.isnull(mean):
                mean = std_dev = min_val = max_val = 0
            else:
                std_dev = np.std(dat[col])
                min_val = float(np.min(dat[col]))
                max_val = float(np.max(dat[col]))
        b_data.append([total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data


def get_sample(dat, key_s):
    rand = []
    for name in key_s:
        rand_sample = list(pd.unique(dat[name]))[:5]
        while len(rand_sample) < 5:
            rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
        rand.append(rand_sample[:5])
    return rand


def featurize_file(df):
    stop_words, word_tokenize = _nltk()
    keys = list(df.keys())
    stats = summary_stats(df, keys)
    samples = get_sample(df, keys)
    cols = ["Attribute_name", "total_vals", "num_nans", "num_of_dist_val", "mean", "std_dev", "min_val", "max_val",
            "%_dist_val", "%_nans", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
    rows = [[keys[i], *stats[i], stats[i][2] * 100.0 / stats[i][0], stats[i][1] * 100.0 / stats[i][0], *samples[i]]
            for i in range(len(keys))]
    curdf = pd.DataFrame(rows, columns=cols)
    for row in curdf.itertuples():
        curlst = [row[11], row[12], row[13], row[14], row[15]]
        delim_cnt = url_cnt = email_cnt = date_cnt = 0
        chars_totals, word_totals, stopwords, whitespaces = [], [], [], []
        for value in curlst:
            word_totals.append(len(str(value).split(" ")))
            chars_totals.append(len(str(value)))
            whitespaces.append(str(value).count(" "))
            if del_reg.match(str(value)):
                delim_cnt += 1
            if url_reg.match(str(value)):
                url_cnt += 1
            if email_reg.match(str(value)):
                email_cnt += 1
            stopwords.append(len([w for w in word_tokenize(str(value)) if w in stop_words]))
            try:
                pd.Timestamp(value)
                date_cnt += 1
            except Exception:  # noqa: BLE001
                pass
        i = row.Index
        curdf.loc[i, "has_delimiters"] = delim_cnt > 2
        curdf.loc[i, "has_url"] = url_cnt > 2
        curdf.loc[i, "has_email"] = email_cnt > 2
        curdf.loc[i, "has_date"] = date_cnt > 2
        curdf.loc[i, "mean_word_count"] = np.mean(word_totals)
        curdf.loc[i, "std_dev_word_count"] = np.std(word_totals)
        curdf.loc[i, "mean_stopword_total"] = np.mean(stopwords)
        curdf.loc[i, "stdev_stopword_total"] = np.std(stopwords)
        curdf.loc[i, "mean_char_count"] = np.mean(chars_totals)
        curdf.loc[i, "stdev_char_count"] = np.std(chars_totals)
        curdf.loc[i, "mean_whitespace_count"] = np.mean(whitespaces)
        curdf.loc[i, "stdev_whitespace_count"] = np.std(whitespaces)
        curdf.loc[i, "mean_delim_count"] = np.mean(whitespaces)  # sic: the original uses whitespaces here too
        curdf.loc[i, "stdev_delim_count"] = np.std(whitespaces)
        curdf.loc[i, "is_list"] = bool(curdf.loc[i, "has_delimiters"]) and curdf.loc[i, "mean_char_count"] < 100
        curdf.loc[i, "is_long_sentence"] = curdf.loc[i, "mean_word_count"] > 10
    return curdf


# ============================================================== model
def stats_matrix(df):
    m = df[STATS].copy()
    for c in m.columns:
        m[c] = pd.to_numeric(m[c].map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v), errors="coerce")
    return m.fillna(0).astype(float).to_numpy()


def features(df, vec):
    from scipy.sparse import csr_matrix, hstack
    return hstack([csr_matrix(stats_matrix(df)), vec.transform([str(x) for x in df["Attribute_name"]])]).tocsr()


GRID = {"n_estimators": [5, 25, 50, 75, 100, 500], "max_depth": [5, 10, 25, 50, 100, 250]}  # the authors' grid


def train(base, log=print, grid=None):
    grid_spec = grid or GRID
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    inp = Path(base) / "inputs"
    xtrain = pd.read_csv(inp / "data_train.csv").sample(frac=1, random_state=100).reset_index(drop=True)
    xtest = pd.read_csv(inp / "data_test.csv")
    y_train = xtrain.y_act.map(TYPES.index).to_numpy()
    y_test = xtest.y_act.map(TYPES.index).to_numpy()
    vec = CountVectorizer(ngram_range=(2, 2), analyzer="char").fit([str(x) for x in xtrain.Attribute_name])
    X_train, X_test = features(xtrain, vec), features(xtest, vec)
    X_tt, X_val, y_tt, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=100)
    grid, best = [], (-1, None, None)
    for ne in grid_spec["n_estimators"]:
        for md in grid_spec["max_depth"]:
            clf = RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=100, n_jobs=-1).fit(X_tt, y_tt)
            sc = clf.score(X_val, y_val)
            grid.append({"n_estimators": ne, "max_depth": md, "val_accuracy": sc})
            if sc > best[0]:
                best = (sc, ne, md)
            log(f"n_estimators {ne:3d}  max_depth {md:3d}  val {sc:.4f}")
    model = RandomForestClassifier(n_estimators=best[1], max_depth=best[2], random_state=100, n_jobs=-1).fit(X_train, y_train)
    report = {"best_n_estimators": best[1], "best_max_depth": best[2], "best_val_accuracy": best[0],
              "holdout_accuracy": float(model.score(X_test, y_test)), "paper_holdout_accuracy_rf": 0.9265,
              "n_train": int(len(y_train)), "n_test": int(len(y_test)), "n_name_bigrams": int(len(vec.vocabulary_)), "grid": grid}
    log(f"best {best[1]} trees, depth {best[2]}; held-out accuracy {report['holdout_accuracy']:.4f} (paper 0.9265)")
    return model, vec, report


def featurize_store(store):
    """Featurized target column of every test instance under C0, C1, C2 (needs no model, so it runs before training)."""
    store = Path(store)
    man = pd.read_csv(store / "manifest.csv")
    test = man[man.role.isin(["gold_test", "groupc1_test"])]
    if test.empty:
        raise ValueError(f"no gold_test / groupc1_test rows in {store / 'manifest.csv'}")
    rows = []
    for r in test.itertuples():
        meta = json.loads((store / "meta" / f"{r.instance_id}.json").read_text(encoding="utf-8"))
        y = pd.read_parquet(store / "targets" / f"{r.instance_id}.parquet").y
        if not pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            y = y.astype(object).where(y.notna(), np.nan)
        n, miss, k = meta["n_rows"], meta["n_missing"], meta["n_unique"]
        for cond in ("C0", "C1", "C2"):
            name = "" if cond == "C0" else str(meta["target_name"])
            np.random.seed(512)
            f = featurize_file(pd.DataFrame({name: y.to_numpy()}))
            f.loc[0, ["total_vals", "num_nans", "num_of_dist_val"]] = [n, miss, k]
            f.loc[0, "%_dist_val"], f.loc[0, "%_nans"] = k * 100.0 / n, miss * 100.0 / n
            rows.append({"instance_id": r.instance_id, "role": r.role, "condition": cond, "name": name, "k": k, "f": f})
    return rows


def predict_store(rows, model, vec):
    types_rows, preds = [], []
    for row in rows:
        iid, cond, name, k, f = row["instance_id"], row["condition"], row["name"], row["k"], row["f"]
        p = np.zeros(len(TYPES))
        p[model.classes_] = model.predict_proba(features(f, vec))[0]
        top = TYPES[int(p.argmax())]
        types_rows.append({"instance_id": iid, "condition": cond, "name_given": name, "type": top,
                           **{f"p_{t}": float(v) for t, v in zip(TYPES, p)}})
        cat = "BINARY" if k == 2 else "MULTICLASS"
        p_cont = p[TYPES.index("numeric")] + p[TYPES.index("embedded-number")] + 0.5 * p[TYPES.index("context-specific")]
        p_cat = p[TYPES.index("categorical")] + 0.5 * p[TYPES.index("context-specific")]
        tot = p_cont + p_cat
        p_cont, p_cat = (p_cont / tot, p_cat / tot) if tot > 0 else (0.5, 0.5)
        probs = {f"p_{t}": 0.0 for t in TASKS}
        probs["p_CONTINUOUS_REGRESSION"], probs[f"p_{cat}"] = p_cont, p_cat
        group = "pilot" if row["role"] == "gold_test" else "group_c"
        base = {"instance_id": iid, "group": group, "condition": cond, **probs}
        if top in ("numeric", "embedded-number"):
            nat = {"parse_ok": True, "plausible_task_set": "CONTINUOUS_REGRESSION", "most_likely_task": "CONTINUOUS_REGRESSION"}
        elif top == "categorical":
            nat = {"parse_ok": True, "plausible_task_set": cat, "most_likely_task": cat}
        elif top == "context-specific":
            nat = {"parse_ok": True, "plausible_task_set": "|".join(sorted([cat, "CONTINUOUS_REGRESSION"])), "most_likely_task": "NONE"}
        else:
            nat = {"parse_ok": False, "plausible_task_set": "", "most_likely_task": ""}
        ml = "CONTINUOUS_REGRESSION" if p_cont >= p_cat else cat
        preds.append({**base, "system": "sortinghat", **nat, "sortinghat_type": top})
        preds.append({**base, "system": "sortinghat:forced", "parse_ok": True, "plausible_task_set": ml,
                      "most_likely_task": ml, "sortinghat_type": top})
    return pd.DataFrame(preds), pd.DataFrame(types_rows)


def versions():
    v = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "nltk", "pyarrow"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def run(base, store, out, log=print, grid=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _nltk()
    rows = featurize_store(store)
    log(f"featurized {len(rows) // 3} test targets x 3 conditions ({time.time() - t0:.0f} s); training now")
    model, vec, report = train(base, log=log, grid=grid)
    report["versions"] = versions()
    report["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    pred, types = predict_store(rows, model, vec)
    pred.to_csv(out / "predictions.csv", index=False)
    types.to_csv(out / "types.csv", index=False)
    log(types.groupby("condition").type.value_counts().unstack(fill_value=0).to_string())
    return pred, types, report
