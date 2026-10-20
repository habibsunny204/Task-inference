"""A-priori keyword lexicons for the semantic features (no pretrained language model; fixed 2026-10-05).

Written from the task definitions (PROTOCOL.md §2) and general data-science vocabulary, before any model was trained.
They are never edited after looking at gold results. Words that name a specific gold or silver target
(e.g. dataset-specific column names or class labels) were deliberately left out, to keep the lexicons general vocabulary. Every lexicon is matched on lower-cased word tokens; multi-word
entries are matched as phrases.
"""

# words in a target or feature name
NAME = {
    "count": ["count", "counts", "cnt", "num", "number", "n", "nb", "freq", "frequency", "visits", "visit", "calls",
              "claims", "accidents", "events", "occurrences", "cases", "deaths", "births", "arrests", "trips",
              "admissions", "days", "nights", "articles", "papers", "publications", "children", "kids", "doctors",
              "goals", "runs", "wins", "errors", "defects", "faults", "failures", "sales",
              "patents", "abundance", "eggs", "seeds", "offspring", "attempts", "hits"],
    "ordinal": ["grade", "grades", "level", "levels", "rating", "ratings", "rank", "ranking", "score", "scores",
                "severity", "stage", "stages", "quality", "satisfaction", "likert", "agreement", "priority", "tier",
                "degree", "intensity", "condition", "evaluation", "assessment", "opinion", "attitude", "preference",
                "risk", "happiness", "health", "pain", "importance"],
    "nominal": ["type", "types", "species", "category", "categories", "kind", "label", "id", "code", "letter",
                "digit", "author", "job", "occupation", "location", "region", "country", "state", "city", "brand",
                "color", "colour", "genre", "site", "group", "family", "name", "class",
                "cluster", "segment", "sector", "religion", "race", "ethnicity", "party", "mode", "make", "model"],
    "continuous": ["price", "cost", "amount", "value", "rate", "time", "duration", "temp", "temperature", "income",
                   "wage", "salary", "weight", "height", "length", "width", "area", "volume", "density", "pressure",
                   "speed", "velocity", "distance", "concentration", "pct", "percent", "percentage", "ratio", "log",
                   "ln", "strength", "energy", "power", "load", "yield", "revenue", "profit", "mass", "size", "age",
                   "latitude", "longitude", "voltage", "current", "noise", "toxicity", "runtime"],
    "binary": ["is", "has", "flag", "default", "churn", "defect", "defects", "survived", "survival", "fraud",
               "spam", "click", "outcome", "diagnosis", "disease", "approved", "accepted", "success",
               "failure", "buy", "purchase", "signal", "malignant", "positive", "presence", "alive", "dead", "yes"],
    "generic": ["y", "target", "class", "label", "outcome", "response", "result", "output", "goal", "dependent"],
}

# words inside the target values themselves (string targets): visible at C0
VALUE = {
    "ordinal": ["none", "low", "medium", "mid", "middle", "high", "very", "mild", "moderate", "severe", "poor",
                "fair", "good", "excellent", "average", "best", "worst", "bad", "never", "rarely", "sometimes",
                "often", "always", "small", "large", "least", "most", "minor", "major", 
                "agree", "disagree", "neutral", "strongly", "first", "second", "third", "lower", "upper"],
    "binary": ["yes", "no", "true", "false", "pos", "neg", "positive", "negative", "present", "absent",
               "good", "bad", "lost", "win", "malignant", "benign"],
}

# phrases in the documentation and title: visible at C2
DOC = {
    "count": ["number of", "count of", "counts of", "how many", "count data", "poisson", "negative binomial",
              "times per", "per week", "per year", "per month", "per day", "frequency of"],
    "ordinal": ["ordinal", "ordered", "order", "scale", "graded", "grade", "rating", "rated", "likert", "severity",
                "levels", "level of", "ranked", "rank", "from 1 to", "1 to 5", "1 to 10", "0 to 10", "1-5", "1-10",
                "intervals", "discretised", "discretized", "binned", "interval index", "ordinal regression"],
    "nominal": ["category", "categories", "categorical", "nominal", "unordered", "type of", "types of", "species",
                "identity", "identify", "letter", "letters", "digit", "digits", "classes", "multiclass", "multi-class"],
    "continuous": ["continuous", "measured", "measurement", "real-valued", "real valued", "in dollars", "price",
                   "percentage", "percent", "kg", "cm", "mm", "mpa", "temperature", "concentration", "rate", "ratio",
                   "log", "regression"],
    "binary": ["binary", "whether", "yes/no", "yes or no", "presence or absence", "two classes", "two-class",
               "positive or negative", "true or false", "0/1", "1/0", "predict whether", "or not"],
    "classification": ["classification", "classify", "classifier", "class label", "class attribute"],
    "regression": ["regression", "predict the value", "estimate the", "predicting the"],
    "encoded": ["integer-encoded", "encoded", "coded as", "code", "codes"],
}
