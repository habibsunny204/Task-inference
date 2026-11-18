"""Zero-shot open-model LLM baseline for task inference.

For every (instance, condition) packet, the model gets prompt_system.md plus the packet and must
return JSON: plausible set, most likely task, per-task probabilities, confidence, evidence (and
evidence level for C2). Greedy decoding. One JSON per item under results/<model_slug>/<condition>/,
written as soon as the item finishes, so an interrupted Colab session resumes.

Used by llm_baseline.ipynb. Locally, `python llm_lib.py selftest` checks prompts and parsing
without a model, and `python llm_lib.py compile <results_dir>` rebuilds predictions.csv.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LABELS = ["BINARY", "MULTICLASS", "ORDINAL", "COUNT_REGRESSION", "CONTINUOUS_REGRESSION"]
LEVELS = ["E1", "E2", "E3", "E5"]
EVIDENCE_FIELD = ',\n  "evidence_level": "<E1|E2|E3|E5>"'
EVIDENCE_RULE = ("- evidence_level (the strongest evidence for the intended task in the documentation): "
                 "E1 = the documentation states the task explicitly; E2 = it describes the target precisely "
                 "enough to fix the task without naming it; E3 = only contextual cues such as names; "
                 "E5 = evidence insufficient.")


def load_bundle(base):
    base = Path(base)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    items = pd.read_csv(base / "items.csv")
    return cfg, items


def system_prompt(base, condition):
    t = (Path(base) / "prompt_system.md").read_text(encoding="utf-8")
    c2 = condition == "C2"
    return (t.replace("__EVIDENCE_LEVEL_FIELD__", EVIDENCE_FIELD if c2 else "")
             .replace("__EVIDENCE_LEVEL_RULE__", EVIDENCE_RULE if c2 else "").strip())


def user_prompt(base, packet_rel):
    packet = (Path(base) / packet_rel).read_text(encoding="utf-8")
    return packet + "\n\nReturn the JSON object now."


def model_slug(model_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def extract_json(text):
    """First balanced {...} block in the text, parsed; tolerates code fences and trailing prose."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\" and not esc)
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))  # trailing commas
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None


def normalise(obj, condition):
    """Validate a parsed answer; returns (clean_dict, list_of_problems)."""
    probs = []
    if not isinstance(obj, dict):
        return None, ["no JSON object"]
    s = obj.get("plausible_task_set") or []
    if isinstance(s, str):
        s = [x.strip() for x in re.split(r"[|,]", s) if x.strip()]
    s = [str(x).strip().upper() for x in s]
    bad = [x for x in s if x not in LABELS]
    s = [x for x in s if x in LABELS]
    if bad:
        probs.append(f"unknown labels dropped: {bad}")
    if not s:
        probs.append("empty plausible set")
    ml = str(obj.get("most_likely_task", "")).strip().upper()
    if ml != "NONE" and ml not in s:
        probs.append(f"most_likely {ml!r} not in set")
    p = obj.get("probabilities") or {}
    pv = {lab: float(p.get(lab, 0) or 0) for lab in LABELS} if isinstance(p, dict) else {}
    tot = sum(pv.values())
    if tot > 0:
        pv = {k: v / tot for k, v in pv.items()}
    else:
        probs.append("no usable probabilities")
    try:
        conf = int(obj.get("confidence"))
    except (TypeError, ValueError):
        conf = None
        probs.append("confidence not an integer")
    clean = {"plausible_task_set": sorted(s), "most_likely_task": ml,
             **{f"p_{k}": v for k, v in pv.items()}, "probability_sum_raw": tot,
             "confidence": conf, "evidence": str(obj.get("evidence", ""))}
    if condition == "C2":
        lvl = str(obj.get("evidence_level", "")).strip().upper()
        clean["evidence_level"] = lvl if lvl in LEVELS else None
        if lvl not in LEVELS:
            probs.append(f"evidence_level {lvl!r} invalid")
    return clean, probs


# --------------------------------------------------------------------------------- model

class HFModel:
    def __init__(self, model_id, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        qcfg = None
        if cfg.get("load_in_4bit", True):
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                          quantization_config=qcfg, torch_dtype=torch.float16)
        self.model.eval()
        self.max_new_tokens = cfg.get("max_new_tokens", 400)
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def chat(self, system, user):
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # return_dict=True: transformers v5 returns a BatchEncoding here anyway; ask for it explicitly
        # so the same code works on v4 and v5
        try:
            enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True)
        except Exception:  # templates without a system role
            enc = self.tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}],
                                               add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True)
        ids = enc["input_ids"].to(self.model.device)
        mask = enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=mask, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text, int(ids.shape[1]), int(out.shape[1] - ids.shape[1])


def run_model(base, results, model, only_groups=None, limit=None, log=print):
    cfg, items = load_bundle(base)
    if only_groups:
        items = items[items.group.isin(only_groups)]
    if limit:
        items = items.head(limit)
    slug = model_slug(model.model_id)
    out_dir = Path(results) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps({
        "model_id": model.model_id, "revision": getattr(model, "revision", None),
        "config": cfg, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1),
        encoding="utf-8")
    for it in items.itertuples():
        p = out_dir / it.condition / f"{it.instance_id}.json"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        sysp, usrp = system_prompt(base, it.condition), user_prompt(base, it.packet)
        rec = {"instance_id": it.instance_id, "group": it.group, "condition": it.condition,
               "model_id": model.model_id,
               "prompt_sha256": hashlib.sha256((sysp + "\n---\n" + usrp).encode()).hexdigest()}
        t0 = time.time()
        attempts = []
        for attempt in range(1 + cfg.get("retries", 1)):
            u = usrp if attempt == 0 else usrp + "\n\nYour previous answer was not valid JSON. Return only the JSON object."
            text, n_in, n_out = model.chat(sysp, u)
            parsed = extract_json(text)
            clean, problems = normalise(parsed, it.condition)
            attempts.append({"raw": text, "tokens_in": n_in, "tokens_out": n_out, "problems": problems})
            if clean is not None and clean["plausible_task_set"]:
                break
        rec.update(answer=clean, problems=problems, attempts=attempts,
                   parse_ok=clean is not None and bool(clean["plausible_task_set"]),
                   seconds=round(time.time() - t0, 2))
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        log(f"{slug} {it.condition} {it.instance_id}: {'ok' if rec['parse_ok'] else 'PARSE FAIL'} "
            f"({rec['seconds']}s)")


def compile_results(results):
    rows = []
    for mdir in sorted(p for p in Path(results).iterdir() if p.is_dir()):
        for f in sorted(mdir.glob("C*/*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            a = r.get("answer") or {}
            rows.append({"model": mdir.name, "instance_id": r["instance_id"], "group": r["group"],
                         "condition": r["condition"], "parse_ok": r["parse_ok"],
                         "plausible_task_set": "|".join(a.get("plausible_task_set", [])),
                         "most_likely_task": a.get("most_likely_task"), "confidence": a.get("confidence"),
                         **{f"p_{k}": a.get(f"p_{k}") for k in LABELS},
                         "evidence_level": a.get("evidence_level"), "evidence": a.get("evidence"),
                         "problems": "; ".join(r.get("problems", [])), "seconds": r.get("seconds"),
                         "tokens_in": sum(x["tokens_in"] for x in r["attempts"]),
                         "tokens_out": sum(x["tokens_out"] for x in r["attempts"])})
    df = pd.DataFrame(rows)
    df.to_csv(Path(results) / "predictions.csv", index=False)
    return df


# --------------------------------------------------------------------------------- local self-test

class FakeModel:
    model_id = "fake/echo"
    revision = None

    def chat(self, system, user):
        c2 = "evidence_level" in system
        body = {"plausible_task_set": ["ORDINAL", "MULTICLASS"], "most_likely_task": "ORDINAL",
                "probabilities": {"ORDINAL": 0.6, "MULTICLASS": 0.3, "CONTINUOUS_REGRESSION": 0.1},
                "confidence": 3, "evidence": "test"}
        if c2:
            body["evidence_level"] = "E3"
        return "Sure! ```json\n" + json.dumps(body) + ",\n```", len(system + user) // 4, 60


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    if sys.argv[1:2] == ["selftest"]:
        import tempfile
        tmp = tempfile.mkdtemp()
        run_model(here, tmp, FakeModel(), limit=6)
        print(compile_results(tmp)[["instance_id", "condition", "parse_ok", "plausible_task_set", "p_ORDINAL"]])
        print(system_prompt(here, "C2")[-600:])
    elif sys.argv[1:2] == ["compile"]:
        print(compile_results(sys.argv[2]).groupby(["model", "condition"]).parse_ok.mean())
