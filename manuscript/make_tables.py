"""Write the paper's LaTeX tables and figure data from the frozen score files (no number is typed by hand).

    .venv/Scripts/python manuscript/make_tables.py

Inputs: benchmark/reference_*.csv, annotation/agreement/, results/scores/, results/downstream/, results/sensitivity/,
        results/systems/tia_v2/, colab store manifest.
Outputs: manuscript/tables/*.tex, the figure data manuscript/figures/*.dat, and manuscript/figures/figure_data.tex
(the numbers that the figures print, as macros). The figure layouts are figures/*.tex without the _data suffix;
manuscript/make_figures.py renders them to the image files the manuscript includes.
"""
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "manuscript" / "tables", ROOT / "manuscript" / "figures"
CONDS = ["C0", "C1", "C2"]
CW = {"C0": "Czero", "C1": "Cone", "C2": "Ctwo"}  # conditions in macro names
sys.path.insert(0, str(ROOT / "scripts"))
import policy_regret as pr  # noqa: E402  loss tables and regret matrices of the ask rule (Figure 1)

FIGMAC = {}  # macros for figures/figure_data.tex, written at the end
# Decision design of every result of the cost-aware rule (ask curves, Appendix D ask tables, Figure 1(b)): `shared`, the
# original HGB fits of folds 0-2 (trained on the other four folds), or `nested` (trained on folds 0-2 only,
# colab/task_inference_nested). Declared in docs/findings_log.md (2026-10-19): `nested` becomes the reported design
# once its fits are back; switched 2026-10-20 (F33). The other design is reported in Table D (ask_design).
DECIDE = "nested"


def by_decide(df):
    """Rows of the reported decision design (files written before the nested run have no `decide` column)."""
    return df[df.decide == DECIDE] if "decide" in df.columns else df


def figmac(name, value):
    assert name.isalpha() and name not in FIGMAC, name
    FIGMAC[name] = value


def write_dat(name, df):
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / name, sep=" ", index=False, float_format="%.4f")
    print("wrote", name)


# fixed colour family per system in every figure: TIA (blue), LLMs (orange), rules and SortingHat (aqua)
def family(s):
    return "llm" if s.startswith("llm:") else ("tia" if s.split(":")[0] in ("method", "v2", "ml") else "rule")

NAMES = {
    "autogluon": "AutoGluon rule", "sklearn": "scikit-learn rule", "heuristic_set": "Heuristic set (values)",
    "heuristic_names": "Heuristic set + names", "sortinghat": "SortingHat", "sortinghat:forced": "SortingHat (forced)",
    "llm:Qwen_Qwen2_5_7B_Instruct": "Qwen2.5-7B", "llm:microsoft_Phi_3_5_mini_instruct": "Phi-3.5-mini",
    "llm:Qwen_Qwen3_14B": "Qwen3-14B", "llm:microsoft_phi_4": "Phi-4",
    "ml:logreg": "Logistic regression", "ml:rf": "Random forest", "method:forced": "TIA-v1 (HGB)",
    "v2:forced": "TIA-v2 (ensemble)", "method:conformal_a05": r"TIA-v1 conformal $\alpha{=}0.05$",
    "method:conformal_a10": r"TIA-v1 conformal $\alpha{=}0.10$", "v2:conformal_a05": r"TIA-v2 conformal $\alpha{=}0.05$",
    "v2:conformal_a10": r"TIA-v2 conformal $\alpha{=}0.10$", "v2:ask10": "TIA-v2 ask budget 10\\%",
    "v2:ask20": "TIA-v2 ask budget 20\\%", "v2:ask30": "TIA-v2 ask budget 30\\%",
    "method:natural_only": "TIA-v1 without Group C v2", "heldout:no_tocuco": "TIA-v1 without TOC-UCO sources",
    "heldout:no_groupc2": "TIA-v1 without Group C v2 (uncal.)", "heldout:no_rdatasets": "TIA-v1 without Rdatasets",
}


def f2(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def bold_max(vals, higher=True, nd=2):
    """Format a column; bold the best value(s)."""
    v = pd.Series(vals, dtype=float)
    best = v.max() if higher else v.min()
    return [("\\textbf{" + f2(x, nd) + "}") if pd.notna(x) and round(x, nd) == round(best, nd) else f2(x, nd) for x in v]


def write(name, body):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


# ---------------------------------------------------------------------------------------------- data
ref = pd.read_csv(ROOT / "results/scores/reference_scores.csv").set_index(["system", "condition"])
grc = pd.read_csv(ROOT / "results/scores/group_c_scores.csv").set_index(["system", "condition"])
long = pd.read_csv(ROOT / "results/scores/reference_scores_long.csv")
inst = pd.read_csv(ROOT / "benchmark/reference_instances.csv")
labels = pd.read_csv(ROOT / "benchmark/reference_labels.csv")
agree = pd.read_csv(ROOT / "annotation/agreement/annotator_1_vs_annotator_2.csv").set_index("condition")
man = pd.read_csv(ROOT / "colab/task_inference_sortinghat/store/manifest.csv").set_index("instance_id")


# ---------------------------------------------------------------------------------------------- Table: benchmark
def source_group(s):
    # rows follow the selection strata of the Benchmark section: a target registered both ways (multitype) is in the
    # conflicting-registrations row even when it is also in a classification suite (isolet: "OpenML-CC18+OpenML-multitype")
    if "multitype" in s:
        return "OpenML, conflicting registrations"
    if s.startswith("TOC-UCO (originally"):
        return "TOC-UCO, originally ordinal"
    if s.startswith("TOC-UCO (discretised"):
        return "TOC-UCO, discretised regression"
    if "CTR23" in s or "regression" in s:
        return "OpenML-CTR23 / AMLB regression"
    return "OpenML-CC18 / AMLB classification"


inst["src"] = inst.instance_id.map(man.source).map(source_group)
order = ["OpenML-CC18 / AMLB classification", "OpenML-CTR23 / AMLB regression", "TOC-UCO, originally ordinal",
         "TOC-UCO, discretised regression", "OpenML, conflicting registrations"]
comp = inst.groupby("src").c2_reference_kind.value_counts().unstack(fill_value=0).reindex(order)
amb = inst.groupby("src").ambiguity_type.value_counts().unstack(fill_value=0).reindex(order)
rows = []
for s in order:
    c, a = comp.loc[s], amb.loc[s]
    rows.append(f"{s} & {int(c.sum())} & {int(c.get('SINGLE', 0))} & {int(c.get('SET_FAVOURED', 0))} & "
                f"{int(c.get('SET_NO_FAVOURITE', 0))} & {int(a.get('NONE', 0))} & {int(a.get('REPRESENTATIONAL', 0))} & "
                f"{int(a.get('EPISTEMIC', 0))} \\\\")
tc, ta = comp.sum(), amb.sum()
rows.append("\\midrule\nTotal & " + " & ".join(str(int(x)) for x in [tc.sum(), tc["SINGLE"], tc["SET_FAVOURED"],
            tc["SET_NO_FAVOURITE"], ta["NONE"], ta["REPRESENTATIONAL"], ta["EPISTEMIC"]]) + " \\\\")
write("benchmark_composition", "\n".join([
    "\\begin{tabular}{lrrrrrrr}", "\\toprule",
    " & & \\multicolumn{3}{c}{C2 reference} & \\multicolumn{3}{c}{Ambiguity type} \\\\",
    "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
    "Source & $n$ & single & set, favourite & set, no fav. & none & represent. & epistemic \\\\", "\\midrule",
    *rows, "\\bottomrule", "\\end{tabular}"]))

rows = []
for c in CONDS:
    lc = labels[labels.condition == c]
    a = agree.loc[c]
    rows.append(f"{c} & {lc.set_size.mean():.2f} & {(lc.set_size > 1).mean():.2f} & {a.most_likely_agreement:.2f} & "
                f"{a.most_likely_kappa:.2f} & {a.set_exact_match:.2f} & {a.set_mean_jaccard:.2f} \\\\")
write("benchmark_conditions", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    " & \\multicolumn{2}{c}{Adjudicated reference} & \\multicolumn{4}{c}{Agreement of the two annotators (before adjudication)} \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
    "Condition & mean $|S|$ & share $|S|>1$ & most likely, raw & Cohen's $\\kappa$ & set exact & set Jaccard \\\\", "\\midrule",
    *rows, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Figure: set size across conditions (RQ1)
# Alluvial plot of the 56 targets from C0 to C1 to C2 by plausible-set size (1, 2, 3 or more), ribbons coloured by the
# ambiguity type. \flownode{stage}{y lo}{y hi}{size}{n} and \flowribbon{stage}{y0 lo}{y0 hi}{y1 lo}{y1 hi}{type}, with y
# in targets (gaps included) and sizes stacked 1 (bottom), 2, 3+ (top). Drawing: figures/ambiguity_flow.tex.
SIZES, FLOW_GAP = ["3", "2", "1"], 4  # top to bottom; "3" stands for three or more tasks
TYPES = {"EPISTEMIC": "epi", "REPRESENTATIONAL": "rep", "NONE": "none"}
sz = pd.DataFrame({c: inst[f"size_{c}"].clip(upper=3).astype(int).astype(str) for c in CONDS})
sz["t"] = inst.ambiguity_type.map(TYPES)
spans = {}
for c in CONDS:
    y, spans[c] = 0, {}
    for k in reversed(SIZES):
        n = int((sz[c] == k).sum())
        spans[c][k] = (y, y + n)
        y += n + FLOW_GAP
flow = [f"\\flownode{{{s}}}{{{lo}}}{{{hi}}}{{{k}}}{{{hi - lo}}}" for s, c in enumerate(CONDS)
        for k, (lo, hi) in spans[c].items() if hi > lo]
tpos = list(TYPES.values())
for s, (a, b) in enumerate(zip(CONDS, CONDS[1:])):
    g = sz.groupby([a, b, "t"]).size()
    # fill every node from its top: outgoing ribbons by destination, incoming ribbons by source, then by type
    top_out, top_in, src = {k: spans[a][k][1] for k in SIZES}, {k: spans[b][k][1] for k in SIZES}, {}
    for (ka, kb, t), n in sorted(g.items(), key=lambda x: (SIZES.index(x[0][0]), SIZES.index(x[0][1]), tpos.index(x[0][2]))):
        src[(ka, kb, t)] = (top_out[ka] - n, top_out[ka])
        top_out[ka] -= n
    for (ka, kb, t), n in sorted(g.items(), key=lambda x: (SIZES.index(x[0][1]), SIZES.index(x[0][0]), tpos.index(x[0][2]))):
        (y0lo, y0hi), (y1lo, y1hi) = src[(ka, kb, t)], (top_in[kb] - n, top_in[kb])
        top_in[kb] -= n
        flow.append(f"\\flowribbon{{{s}}}{{{y0lo}}}{{{y0hi}}}{{{y1lo}}}{{{y1hi}}}{{{t}}}")
figmac("figFlowData", "%\n" + "%\n".join(flow) + "%\n")
for t, key in TYPES.items():
    figmac(f"figFlowN{key.capitalize()}", int((inst.ambiguity_type == t).sum()))
for c in CONDS:
    figmac(f"figFlowSet{CW[c]}", f"{int((inst[f'size_{c}'] > 1).sum())} ({(inst[f'size_{c}'] > 1).mean() * 100:.0f}\\%)")

# ---------------------------------------------------------------------------------------------- Table: task choice
GROUPS = [("Rules used by AutoML tools", ["autogluon", "sklearn", "heuristic_set", "heuristic_names"]),
          ("Feature-type inference", ["sortinghat:forced"]),
          ("Zero-shot LLMs (4-bit, open weights)", ["llm:microsoft_Phi_3_5_mini_instruct", "llm:Qwen_Qwen2_5_7B_Instruct",
                                                    "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4"]),
          ("Learned from silver labels (this work, no LLM)", ["ml:logreg", "ml:rf", "method:forced", "v2:forced"])]
allsys = [s for _, ss in GROUPS for s in ss]
cols = {}
for c in CONDS:
    cols[("g", c)] = bold_max([ref.loc[(s, c), "top1"] for s in allsys])
    cols[("c", c)] = bold_max([grc.loc[(s, c), "top1"] for s in allsys])
ci = {(s, c): ref.loc[(s, c), "top1_ci"].replace("[", "").replace("]", "").replace(" ", "") for s in allsys for c in CONDS}
lines, i = [], 0
for g, ss in GROUPS:
    lines.append(f"\\addlinespace\\multicolumn{{7}}{{l}}{{\\emph{{{g}}}}} \\\\")
    for s in ss:
        gold = " & ".join(f"{cols[('g', c)][i]} {{\\scriptsize[{ci[(s, c)]}]}}" for c in CONDS)
        gc = " & ".join(cols[("c", c)][i] for c in CONDS)
        lines.append(f"\\quad {NAMES[s]} & {gold} & {gc} \\\\")
        i += 1
write("task_choice", "\n".join([
    "\\begin{tabular}{lllllll}", "\\toprule",
    " & \\multicolumn{3}{c}{Gold, 34 single-task targets: top-1 [95\\% CI]} & \\multicolumn{3}{c}{Group C, 20 controlled: top-1} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "System & C0 & C1 & C2 & C0 & C1 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Figure: per task (complementary errors)
# Correct most likely task on the multiclass and ordinal single-task references at C0 and C2 (counts), one row per system,
# the first system at the top. figures/divide.dat; drawing: figures/ordinal_divide.tex.
s1 = long[long.single.astype(bool)]
origin = pd.read_csv(ROOT / "benchmark/annotation_key.csv").drop_duplicates("instance_id").set_index("instance_id").origin
s1 = s1.assign(ref_kind=np.where(s1.set_C2 != "ORDINAL", s1.set_C2,
                                 np.where(s1.instance_id.map(origin).str.contains("discretised"), "ORD_DISC", "ORD_ORIG")))
pt_sys = ["autogluon", "heuristic_names", "sortinghat:forced", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B",
          "llm:microsoft_phi_4", "method:forced", "v2:forced"]
n_task = s1[(s1.system == "autogluon") & (s1.condition == "C2")].ref_kind.value_counts()
PT = {"MULTICLASS": "mul", "ORD_ORIG": "orig", "ORD_DISC": "disc"}
rows = []
for i, s in enumerate(pt_sys):
    r = {"row": len(pt_sys) - 1 - i, "label": "{" + NAMES[s].replace(" ", "~") + "}", "family": family(s)}
    for task, key in PT.items():
        for c in ("C0", "C2"):
            g = s1[(s1.system == s) & (s1.condition == c) & (s1.ref_kind == task)]
            assert len(g) == n_task[task]
            r[f"{key}{c}"] = int(g.top1.sum())
    rows.append(r)
write_dat("divide.dat", pd.DataFrame(rows).sort_values("row"))  # ascending rows, so that labels follow the ticks
for task, key in PT.items():
    figmac(f"figDivideN{key.capitalize()}", int(n_task[task]))

# ---------------------------------------------------------------------------------------------- Table: paired comparisons
# won/lost, the difference in top-1 with Newcombe's 95% interval and the exact McNemar p (scripts/sensitivity_analyses.py)
paired = pd.read_csv(ROOT / "results/sensitivity/paired.csv").set_index(["system", "other", "condition", "subset"])
pairs = [("method:forced", b) for b in ["autogluon", "sortinghat:forced", "llm:Qwen_Qwen2_5_7B_Instruct",
                                        "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4"]] + \
        [("v2:forced", b) for b in ["autogluon", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4"]]
no_mfeat = [(a, b) for a in ("method:forced", "v2:forced") for b in ("llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4")]
SHORT = {"method:forced": "TIA-v1", "v2:forced": "TIA-v2"}


def paired_row(a, b, subset, label=None):
    cells = []
    for c in CONDS:
        r = paired.loc[(a, b, c, subset)]
        # the won/lost counts must match a direct count on the score file
        g = s1[s1.condition == c].pivot_table(index="instance_id", columns="system", values="top1", aggfunc="first").astype(float)
        if subset == "all":
            assert (int(((g[a] == 1) & (g[b] == 0)).sum()), int(((g[a] == 0) & (g[b] == 1)).sum())) == (r.won, r.lost)
        ps = f"{r.mcnemar_p:.3f}" if r.mcnemar_p >= 0.001 else "$<$0.001"
        diff = f"{r['diff']:+.2f} {{\\scriptsize[{r.newcombe_lo:+.2f}, {r.newcombe_hi:+.2f}]}}".replace("-", "$-$")
        cells.append(f"+{int(r.won)}/$-${int(r.lost)} ({('\\textbf{' + ps + '}') if r.mcnemar_p < 0.05 else ps}) & {diff}")
    return f"\\quad {label or NAMES[b]} & " + " & ".join(cells) + " \\\\"


n_nm = int(paired.loc[("method:forced", "llm:microsoft_phi_4", "C0", "no_mfeat")].n)
lines = ["\\multicolumn{7}{l}{\\emph{TIA-v1 against}} \\\\"]
lines += [paired_row(a, b, "all") for a, b in pairs if a == "method:forced"]
lines += ["\\addlinespace\\multicolumn{7}{l}{\\emph{TIA-v2 against}} \\\\"]
lines += [paired_row(a, b, "all") for a, b in pairs if a == "v2:forced"]
lines += [f"\\addlinespace\\multicolumn{{7}}{{l}}{{\\emph{{Without the five handwritten-digit targets ($n={n_nm}$, post hoc)}}}} \\\\"]
lines += [paired_row(a, b, "no_mfeat", f"{SHORT[a]} against {NAMES[b]}") for a, b in no_mfeat]
write("mcnemar", "\n".join([
    "\\begin{tabular}{lrlrlrl}", "\\toprule",
    " & \\multicolumn{2}{c}{C0} & \\multicolumn{2}{c}{C1} & \\multicolumn{2}{c}{C2} \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
    "Comparison & won/lost ($p$) & $\\Delta$ [95\\% CI] & won/lost ($p$) & $\\Delta$ [95\\% CI] & won/lost ($p$) & "
    "$\\Delta$ [95\\% CI] \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Table: ambiguity ranking + calibration
prob_sys = ["sortinghat:forced", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4",
            "ml:logreg", "ml:rf", "method:forced", "v2:forced"]
au = {c: bold_max([ref.loc[(s, c), "auroc_amb"] for s in prob_sys]) for c in CONDS}
br = {c: bold_max([ref.loc[(s, c), "brier"] for s in prob_sys], higher=False) for c in CONDS}
gp = {c: [f2(ref.loc[(s, c), "gap"]) for s in prob_sys] for c in CONDS}
lines = [f"{NAMES[s]} & " + " & ".join(au[c][i] for c in CONDS) + " & " + " & ".join(br[c][i] for c in CONDS) +
         " & " + " & ".join(gp[c][i] for c in CONDS) + " \\\\" for i, s in enumerate(prob_sys)]
write("ambiguity_ranking", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{AUROC $\\uparrow$} & \\multicolumn{3}{c}{Brier $\\downarrow$} & "
    "\\multicolumn{3}{c}{conf.\\ $-$ acc.} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "System & C0 & C1 & C2 & C0 & C1 & C2 & C0 & C1 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

ask_sys = ["heuristic_set", "heuristic_names", "sortinghat", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4",
           "method:conformal_a10", "v2:conformal_a05", "v2:ask10", "v2:ask20", "v2:ask30"]
lines = []
for s in ask_sys:
    cells = []
    for c in CONDS:
        r = ref.loc[(s, c)]
        risk = "--" if pd.isna(r.sel_risk) else f"{r.sel_risk:.2f}"
        cells.append(f"{r.flag_recall:.2f} & {r.flag_specificity:.2f} & {risk} ({r.sel_coverage:.2f})")
    lines.append(f"{NAMES[s]} & " + " & ".join(cells) + " \\\\")
write("ask_decisions", "\n".join([
    "\\begin{tabular}{lrrlrrlrrl}", "\\toprule",
    " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "System & rec. & spec. & risk (cov.) & rec. & spec. & risk (cov.) & rec. & spec. & risk (cov.) \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Figure: ask trade-off (RQ3)
# specificity and recall of the ask decisions of the table above, one file per condition; the ask budgets also as a line.
# Drawing: figures/ask_tradeoff.tex (the keys are its scatter classes).
TRADE = {"heuristic_set": "hset", "heuristic_names": "hnames", "sortinghat": "sh", "llm:Qwen_Qwen3_14B": "qwen",
         "llm:microsoft_phi_4": "phi", "method:conformal_a10": "vonecf", "v2:conformal_a05": "vtwocf",
         "v2:ask10": "budget", "v2:ask20": "budget", "v2:ask30": "budget"}
assert list(TRADE) == ask_sys
for c in CONDS:
    d = pd.DataFrame([{"spec": ref.loc[(s, c), "flag_specificity"], "rec": ref.loc[(s, c), "flag_recall"], "key": k}
                      for s, k in TRADE.items()])
    write_dat(f"tradeoff_{c}.dat", d)
    write_dat(f"tradeoff_budget_{c}.dat", d[d.key == "budget"])

# ---------------------------------------------------------------------------------------------- Table: set agreement with the reference
# exact match and Jaccard against the reference set under the same condition (score_reference.py), all 56 targets.
# First row: the best a system that always returns one task can reach (exact = share of single-task references,
# Jaccard = mean 1/|S|).
ex = {c: bold_max([ref.loc[(s, c), "exact_set"] for s in ask_sys]) for c in CONDS}
ja = {c: bold_max([ref.loc[(s, c), "jaccard"] for s in ask_sys]) for c in CONDS}
lines = ["\\emph{One correct task (bound)} & " + " & ".join(
    f"1.00 & {(lc.set_size == 1).mean():.2f} & {(1 / lc.set_size).mean():.2f}"
    for lc in (labels[labels.condition == c] for c in CONDS)) + " \\\\", "\\addlinespace"]
lines += [f"{NAMES[s]} & " + " & ".join(f"{ref.loc[(s, c), 'set_size']:.2f} & {ex[c][i]} & {ja[c][i]}" for c in CONDS)
          + " \\\\" for i, s in enumerate(ask_sys)]
write("set_metrics", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "System & $|S|$ & exact & Jacc. & $|S|$ & exact & Jacc. & $|S|$ & exact & Jacc. \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Table: hold-out / ablation
toc = set(i for i in s1.instance_id.unique() if str(man.at[i, "source"]).startswith("TOC-UCO"))
n_toc = s1[(s1.system == "method:forced") & (s1.condition == "C2") & s1.instance_id.isin(toc)].shape[0]
abl = ["method:forced", "method:natural_only", "heldout:no_tocuco", "heldout:no_rdatasets"]
ABL = {"method:forced": "All training data (TIA-v1)", "method:natural_only": "without Group C v2 (fixed in advance)",
       "heldout:no_tocuco": "without TOC-UCO tables and their transforms (post hoc)",
       "heldout:no_rdatasets": "without Rdatasets (no COUNT class; post hoc)",
       "heldout:no_bin": "\\quad without the bin transformations",
       "nosig:v1": "\\quad without the six source-signature features",
       "rerun:v1": "\\quad all training data (re-run of TIA-v1)"}
# robustness run (colab/task_inference_robustness): compared with the re-run of TIA-v1 in the same environment
ROBUST = [s for s in ("rerun:v1", "heldout:no_bin", "nosig:v1") if s in ref.index.get_level_values(0)]
lines = []
for s in abl + (["ROBUST_HEADER"] + ROBUST if ROBUST else []):
    if s == "ROBUST_HEADER":
        lines.append("\\addlinespace\\multicolumn{7}{l}{\\emph{Robustness run, newer library versions (post hoc)}} \\\\")
        continue
    all_ = " & ".join(f2(ref.loc[(s, c), "top1"]) for c in CONDS)
    tt = " & ".join(f2(s1[(s1.system == s) & (s1.condition == c) & s1.instance_id.isin(toc)].top1.mean()) for c in CONDS)
    lines.append(f"{ABL[s]} & {all_} & {tt} \\\\")
write("holdout", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    f" & \\multicolumn{{3}}{{c}}{{All 34 single-task}} & \\multicolumn{{3}}{{c}}{{TOC-UCO ($n={n_toc}$)}} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Training data & C0 & C1 & C2 & C0 & C1 & C2 \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Table: downstream regret (always act)
ds_sys = ["autogluon", "heuristic_names", "sortinghat:forced", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B",
          "llm:microsoft_phi_4", "method:forced", "v2:forced"]
evals = [("policy_hgb", "HGB"), ("policy_ag", "AutoGluon"), ("policy_ag_argmax", "AutoGluon, argmax")]
share, med = {}, {}
for d, _ in evals:
    L = pd.read_csv(ROOT / "results" / "downstream" / d / "policy_regret_long.csv")
    for (s, c), g in L.groupby(["system", "condition"]):
        share[(d, s, c)] = (g.regret_if_act > 0.10).mean()
        med[(d, s, c)] = g.regret_if_act.median()
n_ds = int(L[(L.system == "autogluon") & (L.condition == "C0")].shape[0])
bayes = pd.read_csv(ROOT / "results/sensitivity/ask_bayes.csv")  # the Bayes action (review_analyses.py, post hoc)


def bayes_n(system, ev, c, policy="bayes_er", k=0, measure="rel"):
    g = bayes[(bayes.system == system) & (bayes["eval"] == ev) & (bayes.condition == c) & (bayes.measure == measure)
              & (bayes.policy == policy) & (bayes.k == k)]
    assert len(g) == 1, (system, ev, c, policy, k, measure)
    return int(g.costly_left.iloc[0])


DS_ROWS = [(NAMES[s], {(d, c): int(round(share[(d, s, c)] * n_ds)) for d, _ in evals for c in CONDS}) for s in ds_sys]
for s, lab in (("method:forced", "TIA-v1, Bayes action (post hoc)"), ("v2:forced", "TIA-v2, Bayes action (post hoc)")):
    DS_ROWS.append((lab, {(d, c): bayes_n(s, d.removeprefix("policy_"), c) for d, _ in evals for c in CONDS}))
    # the always-act count of the same file must equal the always-act count of the policy files
    for d, _ in evals:
        for c in CONDS:
            assert bayes_n(s, d.removeprefix("policy_"), c, "act_conf") == int(round(share[(d, s, c)] * n_ds))
cols = {(d, c): bold_max([r[(d, c)] for _, r in DS_ROWS], higher=False, nd=0) for d, _ in evals for c in CONDS}
lines = [f"{lab} & " + " & ".join(cols[(d, c)][i] for d, _ in evals for c in CONDS) + " \\\\" for i, (lab, _) in enumerate(DS_ROWS)]
write("downstream", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{HGB} & \\multicolumn{3}{c}{AutoGluon (median)} & \\multicolumn{3}{c}{AutoGluon (argmax)} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "System & C0 & C1 & C2 & C0 & C1 & C2 & C0 & C1 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Table + figure data: ask curves
lines = []
FIG.mkdir(parents=True, exist_ok=True)
for d, lab in evals:
    cur = by_decide(pd.read_csv(ROOT / "results" / "downstream" / d / "ask_curves.csv"))
    for c in CONDS:
        g = cur[cur.condition == c].pivot_table(index="ask_rate", columns="rule", values="share_regret_gt_10pct")
        rates = g.index[:4]
        cells = " & ".join(f"{g.at[r, 'confidence'] * 56:.0f} & {g.at[r, 'expected_regret'] * 56:.0f}" for r in rates)
        lines.append(f"{lab} & {c} & {cells} \\\\")
        out = g.reset_index().rename(columns={"expected_regret": "er", "confidence": "conf"})
        out[["ask_rate", "conf", "er"]].to_csv(FIG / f"ask_{d.removeprefix('policy_')}_{c}.dat",
                                               sep=" ", index=False, float_format="%.4f")
    lines.append("\\addlinespace")
rates_txt = " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(round(r * 56))}/56}}" for r in rates)
write("ask_curves", "\n".join([
    "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
    f"Realized on & Cond. & {rates_txt} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
    " & & conf. & ER & conf. & ER & conf. & ER & conf. & ER \\\\", "\\midrule", *lines[:-1], "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Appendix: regret sensitivity (post hoc)
SENS = ROOT / "results" / "sensitivity"
thr = pd.read_csv(SENS / "regret_thresholds.csv")
rs_sys = ["autogluon", "llm:Qwen_Qwen3_14B", "llm:microsoft_phi_4", "method:forced", "v2:forced"]
EVN = {"policy_hgb": "Realized with HGB fits", "policy_ag": "Realized with AutoGluon fits (median rule)",
       "policy_ag_argmax": "Realized with AutoGluon fits (argmax rule, post hoc)"}


def thr_val(ev, s, c, tau, col):
    g = thr[(thr["eval"] == ev) & (thr.system == s) & (thr.condition == c) & np.isclose(thr.tau, tau)]
    assert len(g) == 1
    return g[col].iloc[0]


lines = []
for d, _ in evals:
    ev = d.removeprefix("policy_")
    lines.append(f"\\addlinespace\\multicolumn{{11}}{{l}}{{\\emph{{{EVN[d]}}}}} \\\\")
    for s in rs_sys:
        cells = [f"{thr_val(ev, s, c, tau, 'share_rel'):.2f}" for tau in (0.05, 0.10, 0.20, 0.50) for c in ("C0", "C2")]
        cells += [f"{thr_val(ev, s, c, 0.10, 'share_norm'):.2f}" for c in ("C0", "C2")]
        lines.append(f"\\quad {NAMES[s]} & " + " & ".join(cells) + " \\\\")
    assert abs(thr_val(ev, "autogluon", "C0", 0.10, "share_rel") - share[(d, "autogluon", "C0")]) < 1e-9
write("regret_thresholds", "\n".join([
    "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{8}{c}{Relative regret above} & \\multicolumn{2}{c}{Normalised} \\\\",
    "\\cmidrule(lr){2-9}",
    " & \\multicolumn{2}{c}{5\\%} & \\multicolumn{2}{c}{10\\%} & \\multicolumn{2}{c}{20\\%} & "
    "\\multicolumn{2}{c}{50\\%} & \\multicolumn{2}{c}{above 10\\%} \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
    "System & C0 & C2 & C0 & C2 & C0 & C2 & C0 & C2 & C0 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Figure: costly share by threshold (post hoc)
# Share of the 35 multi-formulation targets whose always-act regret exceeds tau, as a step function of tau (2% to 100%):
# one row per jump, for a const plot. Must agree with regret_thresholds.csv at its thresholds. Drawing: figures/thresholds.tex.
THR_SYS = {"autogluon": "ag", "llm:Qwen_Qwen3_14B": "qwen", "llm:microsoft_phi_4": "phi", "method:forced": "vone",
           "v2:forced": "vtwo"}
TAU_LO, TAU_HI = 0.02, 1.0
for d in ("policy_hgb", "policy_ag"):
    ev = d.removeprefix("policy_")
    L = pd.read_csv(ROOT / "results" / "downstream" / d / "policy_regret_long.csv")
    for c in ("C0", "C2"):
        reg = {k: L[(L.system == s) & (L.condition == c)].regret_if_act.to_numpy() for s, k in THR_SYS.items()}
        jumps = np.concatenate(list(reg.values()))
        taus = np.unique(np.r_[TAU_LO, TAU_HI, jumps[(jumps > TAU_LO) & (jumps < TAU_HI)]])
        out = pd.DataFrame({"tau": taus, **{k: [(r > t).mean() for t in taus] for k, r in reg.items()}})
        for s, k in THR_SYS.items():
            for tau in (0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
                assert len(reg[k]) == n_ds and abs((reg[k] > tau).mean() - thr_val(ev, s, c, tau, "share_rel")) < 1e-9
        write_dat(f"thr_{ev}_{c}.dat", out)

ask_all = pd.read_csv(SENS / "ask_robust.csv")
ask = by_decide(ask_all)
MEAS = [("rel", 0.05, "relative, above 5\\%", "expected_regret"),
        ("rel", 0.10, "relative, above 10\\% (primary)", "expected_regret"),
        ("rel_clip", 0.10, "\\quad negative regrets set to 0, ER from clipped estimates", "expected_regret_clip"),
        ("rel", 0.20, "relative, above 20\\%", "expected_regret"), ("rel", 0.50, "relative, above 50\\%", "expected_regret"),
        ("norm", 0.10, "normalised, above 10\\%", "expected_regret"),
        ("norm", 0.10, "\\quad ER from normalised estimates", "expected_regret_norm")]
K_ASK = 6


def bayes_cells(ev, measure, c, system="v2:forced"):
    """none: act on a_B and never ask; conf.: a_B and ask the least confident; ER: a_B and ask by its expected regret."""
    none, conf, er = (bayes_n(system, ev, c, p, k, measure) for p, k in (("bayes_er", 0), ("bayes_conf", K_ASK), ("bayes_er", K_ASK)))
    return [str(none), str(conf), f"\\textbf{{{er}}}" if er < conf else str(er)]


# B2 (post hoc): ER from the AutoGluon-like proxy, only where the proxy's ordinal decision rule matches the evaluation
PROXY_OF = {"ag": "proxy_median", "ag_argmax": "proxy_argmax"}
lines = []
for d, _ in evals:
    ev = d.removeprefix("policy_")
    lines.append(f"\\addlinespace\\multicolumn{{10}}{{l}}{{\\emph{{{EVN[d]}}}}} \\\\")
    rows = list(MEAS)
    if ev in PROXY_OF and ask.rule.str.endswith(PROXY_OF[ev]).any():
        rows.insert(2, ("rel", 0.10, "\\quad ER from the AutoGluon-like proxy", f"expected_regret_{PROXY_OF[ev]}"))
        rows.append(("norm", 0.10, "\\quad ER from normalised proxy estimates", f"expected_regret_norm_{PROXY_OF[ev]}"))
    for m, tau, lab, er_rule in rows:
        cells = []
        for c in CONDS:
            g = ask[(ask["eval"] == ev) & (ask.measure == m) & np.isclose(ask.tau, tau) & (ask.condition == c)]
            none = int(g[(g.k == 0) & (g.rule == "confidence")].costly_left.iloc[0])
            conf = int(g[(g.k == K_ASK) & (g.rule == "confidence")].costly_left.iloc[0])
            er = int(g[(g.k == K_ASK) & (g.rule == er_rule)].costly_left.iloc[0])
            cells += [str(none), str(conf), f"\\textbf{{{er}}}" if er < conf else str(er)]
        lines.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    for meas, lab in (("rel", "\\quad Bayes action $a_B$, relative above 10\\% (post hoc)"),
                      ("norm", "\\quad Bayes action $a_B$, normalised above 10\\% (post hoc)")):
        lines.append(f"{lab} & " + " & ".join(x for c in CONDS for x in bayes_cells(ev, meas, c)) + " \\\\")
    if ev != "ag_argmax":  # the primary cell must match the ask-curve table (share of 56 targets)
        cur = by_decide(pd.read_csv(ROOT / "results" / "downstream" / d / "ask_curves.csv"))
        for c in CONDS:
            r = cur[(cur.condition == c) & (cur.rule == "expected_regret") & np.isclose(cur.ask_rate, K_ASK / 56)]
            g = ask[(ask["eval"] == ev) & (ask.measure == "rel") & np.isclose(ask.tau, 0.10) & (ask.condition == c)
                    & (ask.k == K_ASK) & (ask.rule == "expected_regret")]
            assert abs(r.share_regret_gt_10pct.iloc[0] * 56 - g.costly_left.iloc[0]) < 1e-6
write("ask_robust", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Costly choice & none & conf. & ER & none & conf. & ER & none & conf. & ER \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# figure data (post hoc): the ER rule with the AutoGluon-like median proxy, at the budgets of the ask curves that
# ask_robust.csv also has (no 22). The confidence and primary ER counts there must equal the ask curves.
cur = by_decide(pd.read_csv(ROOT / "results" / "downstream" / "policy_ag" / "ask_curves.csv"))
for c in CONDS:
    g = ask[(ask["eval"] == "ag") & (ask.measure == "rel") & np.isclose(ask.tau, 0.10) & (ask.condition == c)]
    ks = [k for k in sorted(g.k.unique()) if np.isclose(cur.ask_rate * 56, k).any()]
    for k in ks:
        for rule in ("confidence", "expected_regret"):
            r = cur[(cur.condition == c) & (cur.rule == rule) & np.isclose(cur.ask_rate * 56, k)]
            assert abs(r.share_regret_gt_10pct.iloc[0] * 56 - g[(g.k == k) & (g.rule == rule)].costly_left.iloc[0]) < 1e-6
    p = g[g.rule == "expected_regret_proxy_median"].set_index("k").costly_left
    write_dat(f"ask_ag_proxy_{c}.dat", pd.DataFrame({"ask_rate": [k / 56 for k in ks], "proxy": [p[k] / 56 for k in ks]}))


# ---------------------------------------------------------------------------------------------- Table: ask rules by cost measure (main text)
# TIA-v2 with six questions (fourth reading): costly targets left (above 10%) when never asking, after asking the least
# confident, and after asking by ER. Each row is realized under one measure; the clipped and normalised ones are post hoc.
def ask_cells(a, ev, m, rule, c, k=K_ASK, tau=0.10):
    g = a[(a["eval"] == ev) & (a.measure == m) & np.isclose(a.tau, tau) & (a.condition == c)]
    v = [g[(g.k == kk) & (g.rule == r)].costly_left for kk, r in ((0, "confidence"), (k, "confidence"), (k, rule))]
    assert all(len(x) == 1 for x in v), (ev, m, rule, c)
    none, conf, er = (int(x.iloc[0]) for x in v)
    return [str(none), str(conf), f"\\textbf{{{er}}}" if er < conf else str(er)]


unc = pd.read_csv(SENS / "ask_uncertainty.csv")


def int_str(x):
    return f"{int(round(x)):+d}".replace("-", "$-$")


def unc_row(ev, c, stat="d6", system="v2:forced", unit="target"):
    r = unc[(unc.system == system) & (unc["eval"] == ev) & (unc.condition == c) & (unc.stat == stat) & (unc.unit == unit)]
    assert len(r) == 1, (ev, c, stat, system, unit)
    return r.iloc[0]


def unc_cell(ev, c, stat="d6", system="v2:forced", unit="target"):
    r = unc_row(ev, c, stat, system, unit)
    return f"\\multicolumn{{3}}{{c}}{{{int_str(r.point)} [{int_str(r.ci_lo)}, {int_str(r.ci_hi)}]}}"


MEAS_MAIN = [("rel", "expected_regret", "Relative regret (fixed in advance)"),
             ("rel_clip", "expected_regret_clip", "Relative, negative regrets set to 0"),
             ("norm", "expected_regret", "Normalised, ER from relative estimates"),
             ("norm", "expected_regret_norm", "Normalised, ER from normalised estimates")]
lines = []
for ev in ("hgb", "ag"):
    lines.append(f"\\addlinespace\\multicolumn{{10}}{{l}}{{\\emph{{{EVN['policy_' + ev]}}}}} \\\\")
    for m, rule, name in MEAS_MAIN:
        lines.append(f"\\quad {name} & " + " & ".join(x for c in CONDS for x in ask_cells(ask, ev, m, rule, c)) + " \\\\")
    lines.append("\\quad Bayes action $a_B$ (relative) & " + " & ".join(x for c in CONDS for x in bayes_cells(ev, "rel", c)) + " \\\\")
    lines.append("\\quad Bayes action $a_B$ (normalised) & " + " & ".join(x for c in CONDS for x in bayes_cells(ev, "norm", c)) + " \\\\")
    lines.append("\\quad conf.\\ $-$ ER, relative [95\\% CI] & " + " & ".join(unc_cell(ev, c) for c in CONDS) + " \\\\")
write("ask_measures", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Cost measure & none & conf. & ER & none & conf. & ER & none & conf. & ER \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Appendix: decision design (nested refit, post hoc)
# The same counts with the ER rule deciding from the original fits of folds 0-2 (trained on four folds, including the
# evaluation folds 3-4) and from fits trained on folds 0-2 only (colab/task_inference_nested).
DESIGNS = [("shared", "four folds, including 3--4"), ("nested", "folds 0--2 only")]
if "decide" in ask_all.columns and set(ask_all.decide) >= {d for d, _ in DESIGNS}:
    lines = []
    for d, _ in evals:
        ev = d.removeprefix("policy_")
        lines.append(f"\\addlinespace\\multicolumn{{10}}{{l}}{{\\emph{{{EVN[d]}}}}} \\\\")
        for m, rule, name in (MEAS_MAIN[0], MEAS_MAIN[3]):
            for dz, lab in DESIGNS:
                cells = [x for c in CONDS for x in ask_cells(ask_all[ask_all.decide == dz], ev, m, rule, c)]
                lines.append(f"\\quad {name.split(' (')[0].split(',')[0]}, fits on {lab} & " + " & ".join(cells) + " \\\\")
    write("ask_design", "\n".join([
        "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
        " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Costly choice, decision fits & none & conf. & ER & none & conf. & ER & none & conf. & ER \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}"]))
else:
    write("ask_design", "\\begin{tabular}{l}\\toprule\n\\todo{nested decision fits pending (colab/task\\_inference\\_nested)}"
                        " \\\\\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------------------------------------- Appendix: Equation 3 with LLM probabilities (post hoc)
# expected costly targets (relative regret > 10%) left after six questions; ties in confidence broken at random (exact
# expectation, sensitivity_analyses.py). TIA-v2 must reproduce the ask_robust table, which uses a fixed order.
plug = by_decide(pd.read_csv(SENS / "ask_plugin.csv"))
plug = plug[(plug.measure == "rel") & np.isclose(plug.tau, 0.10)]
PLUG = ["v2:forced", "llm:microsoft_phi_4", "llm:Qwen_Qwen3_14B"]


def f_cnt(x):
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


lines = []
for d, _ in evals:
    ev = d.removeprefix("policy_")
    lines.append(f"\\addlinespace\\multicolumn{{10}}{{l}}{{\\emph{{{EVN[d]}}}}} \\\\")
    for s in PLUG:
        cells = []
        for c in CONDS:
            g = plug[(plug["eval"] == ev) & (plug.system == s) & (plug.condition == c)]
            none = g[(g.k == 0) & (g.rule == "confidence")].costly_left.iloc[0]
            conf = g[(g.k == K_ASK) & (g.rule == "confidence")].costly_left.iloc[0]
            er = g[(g.k == K_ASK) & (g.rule == "expected_regret")].costly_left.iloc[0]
            if s == "v2:forced":
                a = ask[(ask["eval"] == ev) & (ask.measure == "rel") & np.isclose(ask.tau, 0.10) & (ask.condition == c)]
                assert (none, conf, er) == tuple(float(a[(a.k == k) & (a.rule == r)].costly_left.iloc[0]) for k, r in
                                                 ((0, "confidence"), (K_ASK, "confidence"), (K_ASK, "expected_regret")))
            cells += [f_cnt(none), f_cnt(conf), f"\\textbf{{{f_cnt(er)}}}" if er < conf - 1e-9 else f_cnt(er)]
        lines.append(f"\\quad {NAMES[s]} & " + " & ".join(cells) + " \\\\")
write("ask_plugin", "\n".join([
    "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
    " & \\multicolumn{3}{c}{C0} & \\multicolumn{3}{c}{C1} & \\multicolumn{3}{c}{C2} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
    "Probabilities from & none & conf. & ER & none & conf. & ER & none & conf. & ER \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# ---------------------------------------------------------------------------------------------- Appendix: COUNT on derived labels (Colab robustness run)
pc_path = SENS / "robustness_oof_per_class.csv"
if pc_path.exists():
    pc = pd.read_csv(pc_path)
    pc = pc[pc.part == "natural"]
    lines = []
    for t in ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]:
        g0 = pc[(pc.task == t) & (pc.system == "rerun:v1") & (pc.condition == "C0")].iloc[0]
        cells = []
        for s in ("rerun:v1", "rerun:v2"):
            for c in CONDS:
                r = pc[(pc.task == t) & (pc.system == s) & (pc.condition == c)].iloc[0]
                cells.append(f"{r.recall:.2f} & {f2(r.precision)}")
        lines.append(f"{t.split('_')[0].capitalize()} & {int(g0.n_label)} ({int(g0.n_families)}) & " + " & ".join(cells) + " \\\\")
    body = ["\\begin{tabular}{lrrrrrrrrrrrr}", "\\toprule",
            " & & \\multicolumn{6}{c}{TIA-v1} & \\multicolumn{6}{c}{TIA-v2} \\\\",
            "\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}",
            " & & \\multicolumn{2}{c}{C0} & \\multicolumn{2}{c}{C1} & \\multicolumn{2}{c}{C2} & \\multicolumn{2}{c}{C0} & "
            "\\multicolumn{2}{c}{C1} & \\multicolumn{2}{c}{C2} \\\\",
            "Derived label & $n$ (families) & rec. & prec. & rec. & prec. & rec. & prec. & rec. & prec. & rec. & prec. & rec. & prec. \\\\",
            "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]
    body[0] = "\\begin{tabular}{l" + "r" * 13 + "}"
    write("count_oof", "\n".join(body))
else:
    write("count_oof", "\\begin{tabular}{l}\\toprule\n\\todo{robustness run pending (colab/task\\_inference\\_robustness)} \\\\\n"
                       "\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------------------------------------- Figure 1: worked example
# (a) Reference sets of two targets under each condition; figures/example.tex quotes them, so they are checked here.
# (b) TIA-v2 under C0 with six questions: the car grade is confident and costly (the ER rule asks, the confidence rule
# does not), the abalone rings are unconfident and cheap (the reverse). ER is computed as in scripts/policy_regret.py:
# HGB folds 0-2 decide, folds 3-4 give the realized regret.
ii = inst.set_index("instance_id")
EX_SETS = {"I018": ("REPRESENTATIONAL", "MULTICLASS|ORDINAL", "MULTICLASS|ORDINAL", "MULTICLASS"),
           "I004": ("EPISTEMIC", "CONTINUOUS_REGRESSION|MULTICLASS|ORDINAL", "CONTINUOUS_REGRESSION|MULTICLASS|ORDINAL",
                    "CONTINUOUS_REGRESSION|ORDINAL")}
for iid, v in EX_SETS.items():
    assert tuple(ii.loc[iid, ["ambiguity_type"] + [f"set_{c}" for c in CONDS]]) == v, f"update figures/example.tex ({iid})"

EX_COND, EX_K = "C0", 6
Ld, Ls = pr.decision_tables(DECIDE), pr.eval_tables()
Rd = {i: pr.regret_matrix(L) for i, L in Ld.items()}
Rs = {i: pr.regret_matrix(L) for i, L in Ls.items()}
v2p = pd.read_csv(ROOT / "results/systems/tia_v2/predictions.csv")
v2p = v2p[(v2p.system == "v2:forced") & v2p.instance_id.str.startswith("I") & (v2p.condition == EX_COND)]
recs = []
for r in v2p.itertuples():
    p = np.array([getattr(r, f"p_{t}") for t in pr.TASKS])
    a = pr.TASKS[int(p.argmax())]
    fitted = r.instance_id in Rd and r.instance_id in Rs
    er = float(sum(p[j] * (Rd[r.instance_id].at[a, t] if Rd[r.instance_id][t].notna().any() else 0.0)
                   for j, t in enumerate(pr.TASKS))) if fitted else 0.0
    act = pr.intended_regret(Rs[r.instance_id], ii.at[r.instance_id, "set_C2"].split("|"), a) if fitted else 0.0
    recs.append({"iid": r.instance_id, "p": p, "a": a, "conf": p.max(), "er": er, "act": act})
d = pd.DataFrame(recs)
conf_order, er_order = d.conf.argsort().to_numpy(), (-d.er).argsort().to_numpy()  # the orders of policy_regret.py
cur = by_decide(pd.read_csv(ROOT / "results" / "downstream" / "policy_hgb" / "ask_curves.csv"))
for rule, order in (("confidence", conf_order), ("expected_regret", er_order)):
    asked = np.zeros(len(d), bool)
    asked[order[:EX_K]] = True
    r = cur[(cur.condition == EX_COND) & (cur.rule == rule) & np.isclose(cur.ask_rate * len(d), EX_K)]
    assert abs(r.share_regret_gt_10pct.iloc[0] - (np.where(asked, 0.0, d.act.fillna(0.0)) > 0.10).mean()) < 1e-9
d["conf_rank"], d["er_rank"] = np.argsort(conf_order) + 1, np.argsort(er_order) + 1  # 1 = least confident, highest ER
d = d.set_index("iid")


def nth(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


TN = {"BINARY": "binary", "MULTICLASS": "multiclass", "ORDINAL": "ordinal", "COUNT_REGRESSION": "count",
      "CONTINUOUS_REGRESSION": "continuous"}
EX = {"Car": ("I049", "COUNT_REGRESSION", True), "Aba": ("I014", "CONTINUOUS_REGRESSION", False)}  # ER rule asks?
for key, (iid, a_exp, er_asks) in EX.items():
    r = d.loc[iid]
    assert r.a == a_exp and (r.er_rank <= EX_K, r.conf_rank <= EX_K) == (er_asks, not er_asks)
    bars = []
    for j in np.argsort(-r.p):
        t = pr.TASKS[j]
        if r.p[j] >= 0.005:  # the caption says tasks below 0.005 are not drawn
            # three decimals, so that ER can be recomputed from the panel (review of 2026-11-10)
            rh = "" if t == r.a else (f"{Rd[iid].at[r.a, t]:+.3f}" if Rd[iid][t].notna().any() else "\\text{n/a}")
            bars.append(f"\\exbar{{{len(bars)}}}{{{TN[t]}}}{{{r.p[j]:.3f}}}{{{rh}}}")
    figmac(f"figEx{key}Bars", "".join(bars))
    figmac(f"figEx{key}Chosen", TN[r.a])
    figmac(f"figEx{key}Conf", f"{r.conf:.3f}")
    figmac(f"figEx{key}ER", f"{r.er:.3f}")
    figmac(f"figEx{key}ERrank", nth(int(r.er_rank)))
    figmac(f"figEx{key}Confrank", nth(int(r.conf_rank)))
    figmac(f"figEx{key}Realized", f"{r.act:.2f}" if abs(r.act) < 1 else f"{r.act:.1f}")
figmac("figExN", len(d))
figmac("figExK", EX_K)
figmac("figExKpct", f"{EX_K / len(d) * 100:.0f}")
# what acting costs on folds 3-4: car, error rate of the count model against the intended (multiclass, ordinal) models;
# abalone rings, MAE of the continuous model against the count model
car = Ls["I049"]
intended = [t for t in ii.at["I049", "set_C2"].split("|") if t in car.index]
figmac("figExCarErrChosen", f"{car.at['COUNT_REGRESSION', 'exact_acc'] * 100:.1f}")
figmac("figExCarErrIntended", "--".join(f"{x * 100:.1f}" for x in sorted({car.at[t, 'exact_acc'] for t in intended})))
aba = Ls["I014"]
figmac("figExAbaMaeChosen", f"{aba.at['CONTINUOUS_REGRESSION', 'mae']:.2f}")
figmac("figExAbaMaeAlt", f"{aba.at['COUNT_REGRESSION', 'mae']:.2f}")


# ---------------------------------------------------------------------------------------------- Frontiers review (post hoc): new tables
# Every number below comes from results/sensitivity/ (scripts/review_analyses.py and sensitivity_analyses.py).
SYS_LAB = {"v2:forced": "TIA-v2", "method:forced": "TIA-v1"}
EV_LAB = {"hgb": "HGB", "ag": "AutoGluon (median)", "ag_argmax": "AutoGluon (argmax)"}


def diff_str(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "$-$")


# Table S: uncertainty of confidence minus ER
lines = []
for system in ("v2:forced", "method:forced"):
    for ev in ("hgb", "ag"):
        lines.append(f"\\addlinespace\\multicolumn{{7}}{{l}}{{\\emph{{{SYS_LAB[system]}, {EV_LAB[ev]}}}}} \\\\")
        for c in CONDS:
            for unit in ("target", "family"):
                cells = [f"{int_str(unc_row(ev, c, s, system, unit).point)} [{int_str(unc_row(ev, c, s, system, unit).ci_lo)}, "
                         f"{int_str(unc_row(ev, c, s, system, unit).ci_hi)}]" for s in ("d6", "d11", "d17")]
                r = unc_row(ev, c, "auc", system, unit)
                cells.append(f"{diff_str(r.point)} [{diff_str(r.ci_lo)}, {diff_str(r.ci_hi)}]")
                r = unc_row(ev, c, "norm6", system, unit)
                cells.append(f"{diff_str(r.point, 3)} [{diff_str(r.ci_lo, 3)}, {diff_str(r.ci_hi, 3)}]")
                lines.append(f"\\quad {c} & {unit} & " + " & ".join(cells) + " \\\\")
write("ask_uncertainty", "\n".join([
    "\\begin{tabular}{lllllll}", "\\toprule",
    " & Resampling unit & 6 questions & 11 questions & 17 questions & mean over $k=0$--17 & mean norm.\\ regret, 6 questions \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# Table S: TIA-v1 next to TIA-v2 for both ask rules at 6, 11 and 17 questions
lines = []
for system in ("method:forced", "v2:forced"):
    for ev in ("hgb", "ag"):
        cells = []
        for c in CONDS:
            cells.append(str(bayes_n(system, ev, c, "act_conf", 0)))
            for k in (6, 11, 17):
                cells += [str(bayes_n(system, ev, c, "act_conf", k)), str(bayes_n(system, ev, c, "act_er", k))]
        lines.append(f"{SYS_LAB[system]}, {EV_LAB[ev]} & " + " & ".join(cells) + " \\\\")
write("ask_v1_v2", "\n".join([
    "\\begin{tabular}{l" + "r" * 21 + "}", "\\toprule",
    " & \\multicolumn{7}{c}{C0} & \\multicolumn{7}{c}{C1} & \\multicolumn{7}{c}{C2} \\\\",
    "\\cmidrule(lr){2-8}\\cmidrule(lr){9-15}\\cmidrule(lr){16-22}",
    " & none & \\multicolumn{2}{c}{6} & \\multicolumn{2}{c}{11} & \\multicolumn{2}{c}{17} & none & \\multicolumn{2}{c}{6} & "
    "\\multicolumn{2}{c}{11} & \\multicolumn{2}{c}{17} & none & \\multicolumn{2}{c}{6} & \\multicolumn{2}{c}{11} & \\multicolumn{2}{c}{17} \\\\",
    " & & conf. & ER & conf. & ER & conf. & ER & & conf. & ER & conf. & ER & conf. & ER & & conf. & ER & conf. & ER & conf. & ER \\\\",
    "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Table S: which targets drive each count (TIA-v2)
per = pd.read_csv(SENS / "ask_per_target.csv")
per = per[per["eval"].isin(["hgb", "ag"])]
TN2 = {"BINARY": "\\B", "MULTICLASS": "\\M", "ORDINAL": "\\Ord", "COUNT_REGRESSION": "\\Cnt", "CONTINUOUS_REGRESSION": "\\Cont"}
lines = []
for ev, gv in per.groupby("eval", sort=False):
    lines.append(f"\\addlinespace\\multicolumn{{9}}{{l}}{{\\emph{{{EV_LAB[ev]}}}}} \\\\")
    for r in gv.itertuples():
        tar = str(r.target).replace("_", "\\_")
        dat = str(r.dataset).replace("_", "\\_")
        rg = f"{r.regret_rel:.2f}" if r.regret_rel < 10 else f"{r.regret_rel:.1f}"
        lines.append(f"{r.condition} & {dat} & {TN2[r.chosen]} & {'|'.join(TN2[t] for t in r.set_C2.split('|'))} & {r.conf:.2f} & "
                     f"{diff_str(r.er)} & {rg} & {r.conf_rank}{'*' if r.asked_conf_k6 else ''} & {r.er_rank}{'*' if r.asked_er_k6 else ''} \\\\")
write("ask_per_target", "\n".join([
    "\\begin{tabular}{llllrrrrr}", "\\toprule",
    "Cond. & Target & Acts on & Human set & conf. & ER & regret & conf.\\ rank & ER rank \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# Table S: fold rotation
rot = pd.read_csv(SENS / "ask_rotation.csv", dtype={"decision_folds": str, "eval_folds": str})
rot_ask = rot[rot.k == 6]
lines = []
for system in ("v2:forced", "method:forced"):
    for c in CONDS:
        g = rot_ask[(rot_ask.system == system) & (rot_ask.condition == c)]
        lines.append(f"{SYS_LAB[system]} & {c} & {g.conf.mean():.1f} ({g.conf.min()}--{g.conf.max()}) & "
                     f"{g.er.mean():.1f} ({g.er.min()}--{g.er.max()}) & {int((g.er < g.conf).sum())} & "
                     f"{int((g.er == g.conf).sum())} & {int((g.er > g.conf).sum())} \\\\")
write("ask_rotation", "\n".join([
    "\\begin{tabular}{llrrrrr}", "\\toprule",
    "System & Cond. & conf.\\ mean (range) & ER mean (range) & ER fewer & tie & ER more \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# Table S: new fold seeds (post hoc; declared in docs/findings_log.md F37, results F38). The nested gradient-boosting evaluation
# repeated on fold seeds 1-5 by scripts/seed_replication.py; seed 0 is the assignment of the main text.
seeds = pd.read_csv(SENS / "ask_seeds_curves.csv")
seeds = seeds[(seeds.measure == "rel") & seeds.policy.isin(["act_conf", "act_er"])]
seeds = seeds.pivot_table(index=["system", "condition", "seed", "k"], columns="policy", values="costly_left").reset_index()
SEED_IDS = sorted(seeds.seed.unique())
assert SEED_IDS == [0, 1, 2, 3, 4, 5], SEED_IDS
seeds_summ = pd.read_csv(SENS / "ask_seeds_summary.csv").set_index(["system", "condition", "measure", "summary"])


def seed_values(system, c, budget):
    """Costly choices left (conf., ER) per seed after `budget` questions, or their mean over k = 0..17 ("auc")."""
    g = seeds[(seeds.system == system) & (seeds.condition == c)]
    if budget == "auc":
        return g[g.k <= 17].groupby("seed")[["act_conf", "act_er"]].mean()
    return g[g.k == budget].set_index("seed")[["act_conf", "act_er"]]


def signed(x, nd):
    return f"{0:.{nd}f}" if abs(x) < 1e-12 else diff_str(x, nd)


lines = []
for system in ("v2:forced", "method:forced"):
    lines.append(f"\\addlinespace\\multicolumn{{13}}{{l}}{{\\emph{{{SYS_LAB[system]}}}}} \\\\")
    for c in CONDS:
        for budget, lab in ((6, "6"), (11, "11"), (17, "17"), ("auc", "mean 0--17")):
            g = seed_values(system, c, budget)
            d = g.act_conf - g.act_er  # positive: the ER rule leaves fewer costly choices
            new = d[d.index != 0]
            key = "mean_k0_17" if budget == "auc" else f"k={budget}"
            assert np.isclose(new.mean(), seeds_summ.loc[(system, c, "rel", key), "mean_diff"]), (system, c, key)
            nd = 2 if budget == "auc" else 0
            cells = [f"{r.act_conf:.{nd}f}/{r.act_er:.{nd}f}" for r in g.itertuples()]
            lines.append(f"{c if budget == 6 else ''} & {lab} & " + " & ".join(cells)
                         + f" & {signed(new.mean(), 2 if budget == 'auc' else 1)} & {signed(new.min(), nd)} to {signed(new.max(), nd)}"
                         + f" & {int((new > 0).sum())} & {int((new == 0).sum())} & {int((new < 0).sum())} \\\\")
write("ask_seeds", "\n".join([
    "\\begin{tabular}{llrrrrrrrrrrr}", "\\toprule",
    " & & \\multicolumn{6}{c}{Fold seed (conf./ER)} & \\multicolumn{5}{c}{Seeds 1--5} \\\\",
    "\\cmidrule(lr){3-8}\\cmidrule(lr){9-13}",
    "Cond. & Questions & 0 & 1 & 2 & 3 & 4 & 5 & mean diff. & range & ER fewer & tie & ER more \\\\", "\\midrule", *lines,
    "\\bottomrule", "\\end{tabular}"]))

# Tables S: intervals for the probability metrics, and the paired Brier differences
im = pd.read_csv(SENS / "interval_metrics.csv").set_index(["system", "condition", "metric"])
lines = []
IV_SYS = [(s, ("auroc", "brier")) for s in prob_sys] + [(s, ("exact_set", "jaccard")) for s in ask_sys]
for metric, title in (("auroc", "AUROC for ambiguity ($n=56$)"), ("brier", "Brier score ($n=34$)"), ("exact_set", "Exact set match ($n=56$)"),
                      ("jaccard", "Jaccard ($n=56$)")):
    lines.append(f"\\addlinespace\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\")
    for s, ms in IV_SYS:
        if metric in ms:
            lines.append(f"\\quad {NAMES[s]} & " + " & ".join(
                f"{im.loc[(s, c, metric), 'point']:.2f} [{im.loc[(s, c, metric), 'ci_lo']:.2f}, {im.loc[(s, c, metric), 'ci_hi']:.2f}]"
                for c in CONDS) + " \\\\")
write("interval_metrics", "\n".join([
    "\\begin{tabular}{llll}", "\\toprule", "System & C0 & C1 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

bp = pd.read_csv(SENS / "brier_paired.csv").set_index(["other", "condition"])
lines = [f"{NAMES[o]} & " + " & ".join(
    f"{diff_str(bp.loc[(o, c), 'diff_v2_minus_other'])} [{diff_str(bp.loc[(o, c), 'ci_lo'])}, {diff_str(bp.loc[(o, c), 'ci_hi'])}]"
    for c in CONDS) + " \\\\" for o in ("llm:microsoft_Phi_3_5_mini_instruct", "llm:Qwen_Qwen2_5_7B_Instruct", "llm:Qwen_Qwen3_14B",
                                           "llm:microsoft_phi_4", "ml:rf", "ml:logreg")]
write("brier_paired", "\n".join([
    "\\begin{tabular}{llll}", "\\toprule", "TIA-v2 minus & C0 & C1 & C2 \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Table S: imperfect users and favourite weights (six questions, TIA-v2)
ny = pd.read_csv(SENS / "noisy_user.csv")
fw = pd.read_csv(SENS / "favourite_weights.csv")
USERS = [("wrong_p0.0", "Oracle (as in the main text)"), ("wrong_p0.1", "Wrong answer with $p=0.1$"), ("wrong_p0.2", "Wrong answer with $p=0.2$"),
         ("wrong_p0.3", "Wrong answer with $p=0.3$"), ("favourite", "Answers with the named favourite")]
lines = []
for ev in ("hgb", "ag"):
    lines.append(f"\\addlinespace\\multicolumn{{7}}{{l}}{{\\emph{{{EV_LAB[ev]}}}}} \\\\")
    for u, lab in USERS:
        cells = []
        for c in CONDS:
            for rule in ("confidence", "expected_regret"):
                v = ny[(ny["eval"] == ev) & (ny.condition == c) & (ny.rule == rule) & (ny.user == u)].costly_left.iloc[0]
                cells.append(f"{v:.1f}")
        lines.append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    cells = []
    for c in CONDS:
        for rule in ("confidence", "expected_regret"):
            v = fw[(fw["eval"] == ev) & (fw.condition == c) & (fw.policy == rule) & (fw.weights == "favourite_2_3")].costly.iloc[0]
            cells.append(f"{v:.1f}")
    lines.append("\\quad Oracle, favourite weighted 2/3 & " + " & ".join(cells) + " \\\\")
write("ask_users", "\n".join([
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    " & \\multicolumn{2}{c}{C0} & \\multicolumn{2}{c}{C1} & \\multicolumn{2}{c}{C2} \\\\",
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
    "Simulated user & conf. & ER & conf. & ER & conf. & ER \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"]))

# Table S: formulations per target
fm = pd.read_csv(SENS / "formulations.csv")
FT = {"": "one (the task itself)", "MULTICLASS|ORDINAL|COUNT_REGRESSION|CONTINUOUS_REGRESSION": "\\M, \\Ord, \\Cnt, \\Cont"}
lines = [f"{r.instance_id} & {str(r.dataset).replace('_', chr(92) + '_')} & {str(r.target).replace('_', chr(92) + '_')} & "
         f"{FT['' if pd.isna(r.formulations) else r.formulations]} \\\\" for r in fm.itertuples()]
write("formulations", "\n".join([
    "\\begin{tabular}{llll}", "\\toprule", "ID & Dataset & Target & Formulations fitted \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}"]))

# Table (main text): every robustness check in one table
paired_all = pd.read_csv(SENS / "paired.csv").set_index(["system", "other", "condition", "subset"])
grid = pd.read_csv(SENS / "ask_grid.csv")
times = pd.read_csv(SENS / "decision_fit_time.csv")
oof = pd.read_csv(SENS / "robustness_oof_per_class.csv")


def pr_counts(ev, c, d="nested"):
    g = ask_all[(ask_all.decide == d) & (ask_all["eval"] == ev) & (ask_all.measure == "rel") & np.isclose(ask_all.tau, 0.10)
                & (ask_all.condition == c) & (ask_all.k == K_ASK)]
    return (int(g[g.rule == "confidence"].costly_left.iloc[0]), int(g[g.rule == "expected_regret"].costly_left.iloc[0]))


p04 = paired_all.loc[("method:forced", "llm:microsoft_phi_4", "C0", "all")]
p04n = paired_all.loc[("method:forced", "llm:microsoft_phi_4", "C0", "no_mfeat")]
hold = {c: (s1[(s1.system == "method:forced") & (s1.condition == c) & s1.instance_id.isin(toc)].top1.mean(),
            s1[(s1.system == "heldout:no_tocuco") & (s1.condition == c) & s1.instance_id.isin(toc)].top1.mean()) for c in CONDS}
same_v1 = {}
for c in CONDS:
    a = s1[(s1.system == "rerun:v1") & (s1.condition == c)].set_index("instance_id").most_likely_task
    b = s1[(s1.system == "nosig:v1") & (s1.condition == c)].set_index("instance_id").most_likely_task
    same_v1[c] = int((a.reindex(b.index) != b).sum())
disc = s1[s1.instance_id.map(origin).str.contains("discretised", na=False)]
binned = {c: int(disc[(disc.system == "heldout:no_bin") & (disc.condition == c)].top1.sum()) for c in CONDS}
oof_cnt = oof[(oof.part == "natural") & (oof.task == "COUNT_REGRESSION") & (oof.system == "rerun:v1") & (oof.condition == "C2")].iloc[0]
never_worse = {}
for ev in ("hgb", "ag"):
    g = grid[(grid["eval"] == ev) & (grid.decide == "nested")].pivot_table(index=["condition", "tau"], columns="rule", values="costly_left").reset_index()
    never_worse[ev] = int((g.expected_regret > g.confidence).sum())
rot6 = rot_ask[rot_ask.system == "v2:forced"]
clip_same = True
for ev in ("hgb", "ag"):
    for c in CONDS:
        for k in (0, K_ASK):
            for r in ("confidence", "expected_regret"):
                a = ask_all[(ask_all.decide == "nested") & (ask_all["eval"] == ev) & (ask_all.condition == c) & (ask_all.k == k)
                            & (ask_all.rule == r) & np.isclose(ask_all.tau, 0.10)]
                clip_same = clip_same and int(a[a.measure == "rel"].costly_left.iloc[0]) == int(a[a.measure == "rel_clip"].costly_left.iloc[0])
fw_same = all(
    int(fw[(fw["eval"] == ev) & (fw.condition == c) & (fw.policy == p) & (fw.weights == "uniform")].costly.iloc[0])
    == int(fw[(fw["eval"] == ev) & (fw.condition == c) & (fw.policy == p) & (fw.weights == "favourite_2_3")].costly.iloc[0])
    for ev in ("hgb", "ag") for c in CONDS for p in ("never ask", "confidence", "expected_regret"))
ny_add = max(float(ny[(ny["eval"] == ev) & (ny.condition == c) & (ny.rule == r) & (ny.user == "wrong_p0.3")].costly_left.iloc[0]
                   - ny[(ny["eval"] == ev) & (ny.condition == c) & (ny.rule == r) & (ny.user == "wrong_p0.0")].costly_left.iloc[0])
             for ev in ("hgb", "ag") for c in CONDS for r in ("confidence", "expected_regret"))
sh = {c: pr_counts("hgb", c, "shared") for c in CONDS}
ne = {c: pr_counts("hgb", c, "nested") for c in CONDS}
assert clip_same, "the clipped regret was expected to change no count of the ask table"
assert never_worse["hgb"] == 0 and never_worse["ag"] == 0, "the 'no more costly choices' statement needs revising"
assert fw_same, "favourite weights were expected to change no count of TIA-v2"
seed6 = {c: seed_values("v2:forced", c, K_ASK) for c in CONDS}
for c in CONDS:  # seed 0 of the seed scorer is the reported ask table (nested fits, HGB evaluation)
    assert tuple(int(v) for v in seed6[c].loc[0]) == pr_counts("hgb", c), c
seed6_new = {c: (g.act_conf - g.act_er)[g.index != 0] for c, g in seed6.items()}
assert all((d >= 0).all() for d in seed6_new.values()), "the text says the ER rule never leaves more on the new fold seeds"
ROB = [
    ("Related targets (five digit tables removed)",
     f"TIA-v1 against Phi-4 under C0: +{int(p04.won)}/$-${int(p04.lost)} becomes +{int(p04n.won)}/$-${int(p04n.lost)}"),
    ("TOC-UCO tables removed from training",
     f"top-1 on the 9 TOC-UCO targets: {hold['C0'][0]:.2f} to {hold['C0'][1]:.2f} (C0), {hold['C2'][0]:.2f} to {hold['C2'][1]:.2f} (C2)"),
    ("Source-signature features removed", f"TIA-v1 answers changed: {same_v1['C0']}, {same_v1['C1']}, {same_v1['C2']} of 34 (C0, C1, C2)"),
    ("Bin transformations removed", f"discretised targets recognised: {binned['C0']}, {binned['C1']}, {binned['C2']} of 6"),
    ("Count on derived labels (out of fold)", f"TIA-v1 recall {int(round(oof_cnt.recall * oof_cnt.n_label))} of {int(oof_cnt.n_label)}"),
    ("Regret threshold 5--50\\%, six questions", "ER leaves no more costly choices than confidence at any threshold (HGB, AutoGluon median)"),
    ("Negative regrets set to zero", "no count of costly choices changes"),
    ("Favourite weighted 2/3", "no count of TIA-v2 changes"),
    ("Users who err with $p\\le0.3$", f"adds at most {ny_add:.1f} costly choices at six questions"),
    ("Decision fits, shared against nested", f"ER at six questions, HGB: {sh['C0'][1]}, {sh['C1'][1]}, {sh['C2'][1]} against {ne['C0'][1]}, {ne['C1'][1]}, {ne['C2'][1]} left"),
    ("Ten fold assignments (HGB, shared fits)", f"ER fewer in {int((rot6.er < rot6.conf).sum())}, tie in {int((rot6.er == rot6.conf).sum())}, more in {int((rot6.er > rot6.conf).sum())} of {len(rot6)} cells"),
    ("Five new fold seeds (HGB, nested fits)",
     f"at six questions, ER fewer in {', '.join(str(int((seed6_new[c] > 0).sum())) for c in CONDS)} of {len(seed6_new['C0'])} seeds"
     f" (C0, C1, C2), more in none"),
    ("Decision-fit time per target",f"median {times.decision_fit_seconds.median():.0f}~s, largest {times.decision_fit_seconds.max():.0f}~s"),
]
write("robustness", "\n".join([
    "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{6.4cm}>{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
    "Check & Result \\\\", "\\midrule", *[f"{a} & {b} \\\\" for a, b in ROB], "\\bottomrule", "\\end{tabularx}"]))

(FIG / "figure_data.tex").write_text(
    "% Generated by manuscript/make_tables.py: the numbers that the TikZ figures print. Do not edit by hand.\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in FIGMAC.items()) + "\n", encoding="utf-8")
print("wrote figure_data.tex")

# a few numbers used in the text, checked against the tables
print("downstream targets:", n_ds, "| TOC-UCO single-task:", n_toc, "| per-task n:", n_task.to_dict())
