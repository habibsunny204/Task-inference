# Ambiguity-aware predictive task inference for tabular AutoML with cost-aware clarification

Code and benchmark for the paper *Ambiguity-aware predictive task inference for tabular AutoML with cost-aware clarification*
(under review). Given a tabular target column, TIA infers its predictive task (binary, multiclass, ordinal, count or
continuous regression), returns the set of defensible tasks, and decides whether to act or to ask the user from the
expected downstream regret.

## Requirements

Python 3.12 or later.

```
pip install -r requirements.txt
python scripts/fetch_tocuco.py                # downloads the 20 TOC-UCO tables of the benchmark and checks them
python scripts/build_annotation_packets.py    # rebuilds the packets the annotators and LLMs saw (annotation/packets/)
```

## Benchmark

56 real targets, each with a set of plausible tasks under three information conditions: C0 (values), C1 (plus column
names) and C2 (plus documentation).

| file | content |
|---|---|
| `benchmark/reference_labels.csv` | adjudicated plausible set and most likely task per target and condition (hash in `reference_labels.sha256`, checked by every script) |
| `benchmark/reference_instances.csv` | one row per target: source dataset, target column, C2 set |
| `annotation/packets/` | what the annotators and the LLM baselines saw for each target and condition, rebuilt from the source tables by `scripts/build_annotation_packets.py` (identical to the packets used) |
| `annotation/annotator_1/`, `annotation/annotator_2/` | the two annotators' independent sheets |
| `annotation/adjudication/` | every disagreement with its adjudicated decision and reason |
| `benchmark/group_c/` | 20 transformed targets whose task is known by construction |
| `data/raw/` | the 36 OpenML tables (licences in `data/LICENSES.md`); TOC-UCO is fetched by `scripts/fetch_tocuco.py` |

## Training

TIA is trained without any benchmark label and without an LLM, on labels derived from source metadata and on
controlled transformations.

| step | code | output |
|---|---|---|
| derived training labels | `scripts/build_silver_pool.py` | `benchmark/silver/pool_silver.csv` |
| instance store (Google Colab) | `colab/task_inference_data/` | |
| TIA-v1 and TIA-v2 (Google Colab) | `colab/task_inference_method/` | `results/systems/tia_v1/`, `results/systems/tia_v2/` |

## Baselines and downstream fits

| what | code | output |
|---|---|---|
| AutoML rules and heuristic sets | `scripts/run_rules.py` | `results/systems/rules/` (frozen, with hash) |
| zero-shot LLMs (prompt: `prompt_system.md`) | `colab/task_inference_llm_baseline/`, `colab/task_inference_llm_strong/` | `results/systems/llm/`, `results/systems/llm_strong/` (raw answers) |
| SortingHat | `colab/task_inference_sortinghat/` | `results/systems/sortinghat/` |
| gradient-boosting fit of every task formulation | `scripts/run_downstream_ambiguity.py` | `results/downstream/hgb_fits/` |
| AutoGluon forced into each task | `colab/task_inference_e2e/` | `results/downstream/autogluon_fits/` |
| nested decision fits, proxy refit, robustness of TIA | `colab/task_inference_nested/`, `colab/task_inference_proxy/`, `colab/task_inference_robustness/` | `results/downstream/`, `results/systems/tia_robustness/` |
| nested evaluation on five new fold seeds (Colab or Kaggle) | `colab/task_inference_seeds/` | `results/downstream/hgb_seed_fits/` |

## Evaluation

Every number in the paper's tables and figures is computed from the files above:

```
python scripts/compile_annotations.py annotation/annotator_1
python scripts/compile_annotations.py annotation/annotator_2
python scripts/annotation_agreement.py annotation/annotator_1 annotation/annotator_2
python scripts/build_reference.py                    # rebuilds the reference from the adjudication
python scripts/score_reference.py                    # task choice, sets, calibration
python scripts/score_group_c.py                      # Group C
python scripts/policy_regret.py --eval hgb           # downstream regret and asking, gradient-boosting fits
python scripts/policy_regret.py --eval ag            # the same, AutoGluon fits
python scripts/policy_regret.py --eval ag_argmax
python scripts/sensitivity_analyses.py
python scripts/review_analyses.py
python scripts/seed_replication.py
python scripts/score_mltaskbench.py                  # external check on MLTaskBench (predictions from the Kaggle notebook)
python manuscript/make_tables.py                     # every table and figure data file of the paper
```

The outputs equal the files in `results/`; a few probabilities can differ in the last digit (about 1e-16), because the
order of a sum varies between runs. On Windows consoles, set `PYTHONIOENCODING=utf-8` first.
TIA-v2, the nested decision fits, the proxy refit, the robustness run, the new fold seeds, the MLTaskBench check,
`sensitivity_analyses.py`, `review_analyses.py`, `seed_replication.py` and `score_mltaskbench.py` are post hoc, as labelled in
the paper.

## Results

Top-1 accuracy of the most likely task on the 34 targets with a single reference task:

| system | C0: values | C1: + names | C2: + documentation |
|---|---|---|---|
| AutoGluon rule | 0.74 | 0.74 | 0.74 |
| SortingHat | 0.56 | 0.71 | 0.71 |
| Qwen3-14B (zero-shot) | 0.82 | 0.82 | 0.97 |
| Phi-4 (zero-shot) | 0.79 | 0.82 | 1.00 |
| TIA-v1 | 0.94 | 0.94 | 0.91 |
| TIA-v2 (post hoc) | 0.91 | 0.94 | 0.91 |

With 34 targets the confidence intervals are wide; the differences between TIA and the 14B LLMs are not significant.

Targets left with a costly choice (relative regret above 10%) among the 56 when TIA-v2 asks six questions (11%):

| regret realized with | questions | C0 | C1 | C2 |
|---|---|---|---|---|
| gradient boosting | none | 2 | 3 | 3 |
| gradient boosting | 6, least confident | 1 | 2 | 3 |
| gradient boosting | 6, highest expected regret | 1 | 1 | 2 |
| AutoGluon | none | 4 | 5 | 4 |
| AutoGluon | 6, least confident | 3 | 4 | 4 |
| AutoGluon | 6, highest expected regret | 3 | 3 | 3 |

With larger budgets, asking the least confident targets does better.

External check (post hoc): top-1 accuracy on the 210 MLTaskBench targets whose tables share no records with a training or
benchmark table (labels binary, multiclass or regression; an ordinal answer counts for multiclass or regression). The
notebook `colab/task_inference_mltaskbench/mltaskbench_kaggle.ipynb` reads the tables from Zenodo; the target list and the
overlap decisions are in `data/external/mltaskbench/`.

| system | C0: values | C1: + names | C2: + documentation |
|---|---|---|---|
| AutoGluon rule | 0.95 | 0.95 | 0.95 |
| TIA-v1 | 0.91 | 0.94 | 0.91 |
| TIA-v2 (post hoc) | 0.94 | 0.94 | 0.94 |

Every paired difference between TIA-v2 and the AutoGluon rule has a 95% interval that includes zero.

## License

MIT (`LICENSE`) for the code, annotations, labels and results. The source datasets keep their original licences;
the MLTaskBench metadata in `data/external/mltaskbench/` are CC BY 4.0 (Tsialis et al., doi:10.5281/zenodo.21649607).

## Citation

To be added on publication.
