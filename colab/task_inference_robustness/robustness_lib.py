"""Robustness run for the paper (post hoc; declared in docs/findings_log.md on 2026-10-19 before running).

Everything uses the frozen features (`inputs/features.csv`, written by the TIA-v1 run) and the frozen training code
(`method_lib.py`, copied unchanged from colab/task_inference_method/). Test labels are never read.

  R1 fidelity      re-run TIA-v1 (HGB + temperature) and TIA-v2 (HGB + RF + LR, + temperature); compare the test
                   probabilities with the frozen predictions (inputs/frozen_test.csv).
  R2 out-of-fold   calibrated out-of-fold probabilities on the 510 training instances (family-grouped 5-fold CV, the
                   same folds as in training), for v1 and v2: per-class recall and precision on derived labels.
  R3 no signature  v1 and v2 re-trained without the features that identify TOC-UCO tables (cfg["signature_features"]).
  R4 no bin        v1 trained without the bin3/bin5/bin7 transformations (numcode, round, anon kept); argmax only, like
                   the other hold-outs.

Systems written to results/predictions.csv: rerun:v1, rerun:v2, nosig:v1, nosig:v2, heldout:no_bin.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import method_lib as ml

TASKS = ml.TASKS


def versions():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def calibrated(learners, Xtr, Xte, y, y_idx, w, groups, natural, seed, n_folds, cache=None):
    """Mean of the learners' OOF and test probabilities, then one temperature on the natural OOF part (as v1 / v2)."""
    oofs, ptes = [], []
    for name in learners:
        if cache is not None and name in cache:
            oof, pte = cache[name]
        else:
            oof = ml.oof_proba(name, Xtr, y, w, groups, seed, n_folds)
            pte = ml.full_proba(ml.fit(name, Xtr, y, w, seed), Xte)
            if cache is not None:
                cache[name] = (oof, pte)
        oofs.append(oof)
        ptes.append(pte)
    oof, pte = np.mean(oofs, axis=0), np.mean(ptes, axis=0)
    T = ml.temperature(oof[natural], y_idx[natural])
    return ml.apply_temperature(oof, T), ml.apply_temperature(pte, T), T


def run(work, out, cfg, smoke=False):
    work, out = Path(work), Path(out)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(work / "inputs" / "manifest.csv")
    feats = pd.read_csv(work / "inputs" / "features.csv", index_col=0)
    train = man[man.role.isin(["silver_train", "groupc2_train"]) & man.label.isin(TASKS)]
    test = man[man.role.isin(["gold_test", "groupc1_test"])]
    if smoke:  # a reduced grid for a quick local check; never used for results
        train = train.groupby("label").head(12)
        test = test.head(6)
    train, test = train.set_index("instance_id"), test.set_index("instance_id")
    src = man.set_index("instance_id").source
    eff_source = train.source.where(train.role == "silver_train", train.base_id.map(src)).fillna("").astype(str)
    y = train.label
    y_idx = y.map(TASKS.index).to_numpy()
    w = ml.sample_weights(train.reset_index())
    groups = train.family.astype(str).to_numpy()
    natural = (train.role == "silver_train").to_numpy()
    seed, nf = cfg["seed"], cfg["n_folds"]
    sig = cfg["signature_features"]
    missing = [c for c in sig if c not in feats.columns]
    assert not missing, f"signature features not in features.csv: {missing}"
    no_bin = ~train.transformation.fillna("").astype(str).str.startswith("bin").to_numpy()

    preds, oof_rows, diag = [], [], []
    for cond in ml.CONDITIONS:
        t0 = time.time()
        cols = ml.condition_columns(feats.columns, cond)
        Xtr, Xte = feats.loc[train.index, cols], feats.loc[test.index, cols]
        cache = {}
        variants = {"rerun:v1": (["hgb"], Xtr, Xte), "rerun:v2": (["hgb", "rf", "logreg"], Xtr, Xte)}
        kept = [c for c in cols if c not in sig]
        variants["nosig:v1"] = (["hgb"], Xtr[kept], Xte[kept])
        variants["nosig:v2"] = (["hgb", "rf", "logreg"], Xtr[kept], Xte[kept])
        nosig_cache = {}
        for system, (learners, A, B) in variants.items():
            c = cache if system.startswith("rerun") else nosig_cache
            oof_c, pte_c, T = calibrated(learners, A, B, y, y_idx, w, groups, natural, seed, nf, c)
            preds += ml.rows_for(test.index, test, cond, system, pte_c, [[TASKS[i]] for i in pte_c.argmax(1)])
            for part, mask in (("natural", natural), ("transformed", ~natural)):
                diag.append({"condition": cond, "system": system, "oof_part": part, "n": int(mask.sum()),
                             "n_features": A.shape[1], "temperature": T,
                             "oof_accuracy": float((oof_c[mask].argmax(1) == y_idx[mask]).mean())})
            for iid, row in zip(train.index, oof_c):
                oof_rows.append({"instance_id": iid, "condition": cond, "system": system,
                                 **{f"p_{t}": float(v) for t, v in zip(TASKS, row)}})
        # R4: no bin transformations (argmax of the uncalibrated model, as the other hold-outs)
        tr = train[no_bin]
        m = ml.fit("hgb", feats.loc[tr.index, cols], tr.label, ml.sample_weights(tr.reset_index()), seed)
        p = ml.full_proba(m, Xte)
        preds += ml.rows_for(test.index, test, cond, "heldout:no_bin", p, [[TASKS[i]] for i in p.argmax(1)])
        diag.append({"condition": cond, "system": "heldout:no_bin", "oof_part": "train", "n": int(len(tr)),
                     "n_features": len(cols), "train_labels": json.dumps(tr.label.value_counts().to_dict())})
        print(f"{cond}: done in {time.time() - t0:.0f} s", flush=True)

    pred = pd.DataFrame(preds)
    pred.to_csv(out / "predictions.csv", index=False)
    oof = pd.DataFrame(oof_rows).merge(
        train[["label", "role", "family", "source", "transformation"]].assign(effective_source=eff_source),
        left_on="instance_id", right_index=True)
    oof.to_csv(out / "diagnostics" / "oof_training.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics" / "variants.csv", index=False)

    # R1: fidelity against the frozen test predictions
    frozen = pd.read_csv(work / "inputs" / "frozen_test.csv")
    fid = {}
    for new, old in (("rerun:v1", "method:forced"), ("rerun:v2", "v2:forced")):
        a = pred[pred.system == new].set_index(["instance_id", "condition"])
        b = frozen[frozen.system == old].set_index(["instance_id", "condition"]).loc[a.index]
        pc = [f"p_{t}" for t in TASKS]
        fid[new] = {"frozen_system": old, "n": int(len(a)),
                    "max_abs_prob_diff": float(np.abs(a[pc].to_numpy() - b[pc].to_numpy()).max()),
                    "argmax_agreement": float((a.most_likely_task == b.most_likely_task).mean())}
    report = {"versions": versions(), "smoke": smoke, "n_train": int(len(train)), "n_test": int(len(test)),
              "signature_features": sig, "fidelity": fid}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(pd.DataFrame(diag).pivot_table(index=["condition", "system"], columns="oof_part", values="oof_accuracy").round(3))
    return report
