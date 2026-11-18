You are an expert data scientist. You will see information about one target column of one
tabular dataset. Decide which predictive task(s) a competent data scientist could defend as the
intended task for this target, given ONLY the information shown.

Task labels (use exactly these spellings):

- BINARY: two outcome categories.
- MULTICLASS: more than two unordered categories.
- ORDINAL: more than two ordered categories where distances between levels are not meaningful
  (ratings, grades, severity levels, scales, discretised bins).
- COUNT_REGRESSION: non-negative integers counting events or items.
- CONTINUOUS_REGRESSION: a real-valued quantity, or an integer-rounded measurement treated as
  continuous.

A target can plausibly belong to several labels. Integer scores such as 3-9 can be ORDINAL,
MULTICLASS or CONTINUOUS_REGRESSION; small counts can be COUNT_REGRESSION or ORDINAL. Include
every label you could defend to a careful colleague, and none you could not. When the evidence is
thin, a wide set with low confidence is a correct answer.

Judge only from what is shown. Do not rely on remembered knowledge of a specific named dataset
beyond what the text states.

Answer with a single JSON object and nothing else:

{
  "plausible_task_set": ["<label>", ...],
  "most_likely_task": "<label or NONE>",
  "probabilities": {"BINARY": p, "MULTICLASS": p, "ORDINAL": p, "COUNT_REGRESSION": p, "CONTINUOUS_REGRESSION": p},
  "confidence": <integer 1-5>,
  "evidence": "<one sentence>"__EVIDENCE_LEVEL_FIELD__
}

Rules for the fields:
- most_likely_task is one label from plausible_task_set, or NONE if no label is clearly more likely.
- probabilities: your probability that each label is the intended task; they sum to 1.
- confidence: 1 = guess, 2 = weak lean, 3 = moderate, 4 = strong, 5 = certain.
__EVIDENCE_LEVEL_RULE__