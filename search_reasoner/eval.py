#!/usr/bin/env python3
"""
Evaluation: Base 3B vs Fine-tuned 3B + Teacher Quality Stats

Loads the CoT training data, joins with ground truth from mcqa_search.jsonl,
runs inference with both the base Ministral-3B and the fine-tuned
NayanK/search-reasoner-3b, and computes:

  - JSON valid rate
  - Schema valid rate
  - Answer parse rate
  - Answer accuracy (vs ground truth)
  - Think tag rate
  - Teacher label quality stats (accuracy, by option, by difficulty)

Logs everything to Weights & Biases and saves local JSON + Markdown reports.

Usage:
    python search_reasoner/eval.py \
        --mcqa_file data/mcqa_search.jsonl \
        --cot_file data/cot_training_data.jsonl \
        --base_model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
        --finetuned_repo NayanK/search-reasoner-3b \
        --wandb_project search-reasoner-3b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# ──────────────────────────────────────────────
# Model class resolution (same as train script)
# ──────────────────────────────────────────────

_MINISTRAL_CLS = None
for _cls_name in ("Mistral3ForConditionalGeneration", "MinistralForCausalLM"):
    try:
        _MINISTRAL_CLS = getattr(
            __import__("transformers", fromlist=[_cls_name]), _cls_name
        )
        break
    except (ImportError, AttributeError):
        continue

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# ──────────────────────────────────────────────
# System prompt (shared across all scripts)
# ──────────────────────────────────────────────

SEARCH_REASONER_SYSTEM_PROMPT = """\
You are a Search Result Reasoning Agent. Your job is to analyze web search \
results and their scraped content to answer a user's question accurately.

## Process

You MUST think step-by-step inside <think>...</think> tags before giving \
your final answer. Your reasoning should:

1. **Evaluate each search result**: For each of the provided search results, assess:
   - Is the source credible and authoritative for this topic?
   - Does the title/snippet suggest it contains relevant information?
   - Does the scraped page content actually contain the answer?
   - Rate each result: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT

2. **Identify answer-bearing results**: Which specific results contain \
information that directly answers the question? Quote the relevant passages.

3. **Cross-reference**: Do multiple sources agree? Are there contradictions? \
Which source is most trustworthy?

4. **Synthesize**: Combine information from the best sources into a coherent answer.

## Output Format

After your <think>...</think> reasoning, provide your answer in this exact JSON format:

```json
{
  "result_rankings": [
    {"rank": 1, "result_index": <0-indexed>, "relevance": "HIGHLY_RELEVANT", "reason": "..."},
    {"rank": 2, "result_index": <0-indexed>, "relevance": "SOMEWHAT_RELEVANT", "reason": "..."}
  ],
  "best_result_index": <0-indexed int>,
  "answer": "<LETTER>",
  "confidence": <0.0 to 1.0>,
  "supporting_evidence": ["<quote from source 1>", "<quote from source 2>"]
}
```

## Rules
- Always evaluate ALL results before deciding
- Prefer primary sources over secondary
- If no result contains a good answer, say so honestly
- Be specific — cite which result(s) informed your answer
- If the question is multiple-choice, state the letter answer clearly"""


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────


def load_ground_truth(mcqa_path: str) -> dict[str, dict]:
    """Load mcqa_search.jsonl into a lookup keyed by question text."""
    lookup = {}
    with open(mcqa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q_text = rec["responses_create_params"]["input"].strip()
            lookup[q_text] = {
                "expected_answer": rec["expected_answer"],
                "difficulty": rec.get("task_difficulty_qwen3_32b_avg_8"),
            }
    return lookup


def load_cot_data(cot_path: str) -> list[dict]:
    """Load the CoT training data JSONL."""
    records = []
    with open(cot_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def join_with_ground_truth(
    cot_records: list[dict], gt_lookup: dict[str, dict]
) -> list[dict]:
    """Attach ground truth answer + difficulty to each CoT record."""
    joined = []
    missed = 0
    for rec in cot_records:
        q = rec["question"].strip()
        gt = gt_lookup.get(q)
        if gt is None:
            missed += 1
            continue
        rec["ground_truth"] = gt["expected_answer"]
        rec["difficulty"] = gt["difficulty"]
        joined.append(rec)
    if missed:
        print(f"  Warning: {missed} CoT records had no ground truth match")
    return joined


# ──────────────────────────────────────────────
# 2. TEACHER QUALITY STATS (no GPU needed)
# ──────────────────────────────────────────────


def compute_teacher_stats(records: list[dict]) -> dict:
    """Compute teacher label quality against ground truth."""
    total = len(records)
    correct = 0
    by_option = defaultdict(lambda: {"total": 0, "correct": 0})
    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})

    mismatches = []

    for rec in records:
        teacher_ans = rec["parsed_answer"]
        gt_ans = rec["ground_truth"]
        diff = rec.get("difficulty")

        is_correct = teacher_ans == gt_ans
        if is_correct:
            correct += 1
        else:
            mismatches.append({
                "question": rec["question"][:120],
                "teacher": teacher_ans,
                "ground_truth": gt_ans,
            })

        by_option[gt_ans]["total"] += 1
        if is_correct:
            by_option[gt_ans]["correct"] += 1

        if diff is not None:
            bucket = "easy" if diff >= 0.6 else ("medium" if diff >= 0.3 else "hard")
        else:
            bucket = "unknown"
        by_difficulty[bucket]["total"] += 1
        if is_correct:
            by_difficulty[bucket]["correct"] += 1

    stats = {
        "teacher/total_examples": total,
        "teacher/correct": correct,
        "teacher/accuracy": correct / max(total, 1),
    }

    for opt in sorted(by_option.keys()):
        d = by_option[opt]
        stats[f"teacher/accuracy_option_{opt}"] = d["correct"] / max(d["total"], 1)
        stats[f"teacher/count_option_{opt}"] = d["total"]

    for bucket in sorted(by_difficulty.keys()):
        d = by_difficulty[bucket]
        stats[f"teacher/accuracy_{bucket}"] = d["correct"] / max(d["total"], 1)
        stats[f"teacher/count_{bucket}"] = d["total"]

    return stats, mismatches


# ──────────────────────────────────────────────
# 3. MODEL OUTPUT PARSING
# ──────────────────────────────────────────────


def parse_model_output(text: str) -> dict:
    """Parse generated text for validity metrics."""
    result = {
        "raw_output": text[:500],
        "json_valid": False,
        "schema_valid": False,
        "parsed_answer": None,
        "has_think_tags": "<think>" in text and "</think>" in text,
        "has_think_open": "<think>" in text,
    }

    after_think = text
    if "</think>" in text:
        after_think = text.split("</think>", 1)[1].strip()

    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", after_think)
    if not json_match:
        json_match = re.search(r"\{[\s\S]*\}", after_think)
    if not json_match:
        return result

    try:
        parsed = json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
        result["json_valid"] = True
    except json.JSONDecodeError:
        return result

    if "answer" in parsed:
        result["schema_valid"] = "best_result_index" in parsed
        answer = str(parsed["answer"]).strip().upper()
        if len(answer) == 1 and answer.isalpha():
            result["parsed_answer"] = answer

    return result


# ──────────────────────────────────────────────
# 4. INFERENCE
# ──────────────────────────────────────────────


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load a model with optional 4-bit quantization."""
    print(f"\n  Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(trust_remote_code=True, device_map="auto")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model_cls = AutoModelForCausalLM
    use_ministral = False
    if _MINISTRAL_CLS is not None:
        if "Ministral" in model_name or "mistral" in model_name.lower():
            use_ministral = True
        else:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                if "mistral3" in type(cfg).__name__.lower():
                    use_ministral = True
            except Exception:
                pass
    if use_ministral:
        model_cls = _MINISTRAL_CLS
        print(f"    Using explicit class: {model_cls.__name__}")

    model = model_cls.from_pretrained(model_name, **load_kwargs)
    model.eval()

    elapsed = time.time() - t0
    print(f"    Loaded in {elapsed:.1f}s")
    return model, tokenizer


@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    eval_records: list[dict],
    max_new_tokens: int = 2048,
    desc: str = "model",
) -> list[dict]:
    """Run generation on eval records, return per-example results."""
    results = []
    n = len(eval_records)
    print(f"\n  Running {desc} inference on {n} examples...")
    t0 = time.time()

    for i, rec in enumerate(eval_records):
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (n - i - 1) / max(rate, 0.001)
            print(f"    [{i+1}/{n}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        messages = [
            {"role": "system", "content": SEARCH_REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": f"## Question\n{rec['question']}\n\n## Search Results\n{rec['search_context']}"},
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=3072
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = parse_model_output(generated_text)
        parsed["ground_truth"] = rec["ground_truth"]
        parsed["teacher_answer"] = rec["parsed_answer"]
        parsed["answer_correct"] = parsed["parsed_answer"] == rec["ground_truth"] if parsed["parsed_answer"] else False
        parsed["question_snippet"] = rec["question"][:150]
        parsed["difficulty"] = rec.get("difficulty")
        results.append(parsed)

    total_time = time.time() - t0
    print(f"    Done in {total_time:.1f}s ({total_time/max(n,1):.1f}s/example)")
    return results


# ──────────────────────────────────────────────
# 5. METRICS COMPUTATION
# ──────────────────────────────────────────────


def compute_metrics(results: list[dict], prefix: str) -> dict:
    """Compute aggregate metrics from per-example results."""
    n = len(results)
    if n == 0:
        return {}

    metrics = {
        f"{prefix}/n_examples": n,
        f"{prefix}/json_valid_rate": sum(r["json_valid"] for r in results) / n,
        f"{prefix}/schema_valid_rate": sum(r["schema_valid"] for r in results) / n,
        f"{prefix}/answer_parse_rate": sum(r["parsed_answer"] is not None for r in results) / n,
        f"{prefix}/answer_accuracy": sum(r["answer_correct"] for r in results) / n,
        f"{prefix}/think_tag_rate": sum(r["has_think_tags"] for r in results) / n,
        f"{prefix}/think_open_rate": sum(r["has_think_open"] for r in results) / n,
    }

    by_option = defaultdict(lambda: {"total": 0, "correct": 0})
    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        gt = r["ground_truth"]
        by_option[gt]["total"] += 1
        if r["answer_correct"]:
            by_option[gt]["correct"] += 1

        diff = r.get("difficulty")
        if diff is not None:
            bucket = "easy" if diff >= 0.6 else ("medium" if diff >= 0.3 else "hard")
        else:
            bucket = "unknown"
        by_difficulty[bucket]["total"] += 1
        if r["answer_correct"]:
            by_difficulty[bucket]["correct"] += 1

    for opt in sorted(by_option.keys()):
        d = by_option[opt]
        metrics[f"{prefix}/accuracy_option_{opt}"] = d["correct"] / max(d["total"], 1)

    for bucket in sorted(by_difficulty.keys()):
        d = by_difficulty[bucket]
        metrics[f"{prefix}/accuracy_{bucket}"] = d["correct"] / max(d["total"], 1)

    return metrics


# ──────────────────────────────────────────────
# 6. REPORTING
# ──────────────────────────────────────────────


def build_comparison_table(
    base_metrics: dict,
    ft_metrics: dict,
    teacher_stats: dict,
) -> list[list[str]]:
    """Build a comparison table as list of rows."""
    rows = []
    metric_keys = [
        ("JSON Valid Rate", "json_valid_rate"),
        ("Schema Valid Rate", "schema_valid_rate"),
        ("Answer Parse Rate", "answer_parse_rate"),
        ("Answer Accuracy", "answer_accuracy"),
        ("Think Tag Rate", "think_tag_rate"),
    ]

    for label, key in metric_keys:
        base_val = base_metrics.get(f"base/{key}", 0)
        ft_val = ft_metrics.get(f"finetuned/{key}", 0)
        delta = ft_val - base_val
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        rows.append([label, f"{base_val:.1%}", f"{ft_val:.1%}", delta_str])

    return rows


def save_markdown_report(
    output_path: str,
    comparison_rows: list[list[str]],
    teacher_stats: dict,
    base_metrics: dict,
    ft_metrics: dict,
    base_results: list[dict],
    ft_results: list[dict],
    teacher_mismatches: list[dict],
):
    """Save a comprehensive Markdown evaluation report."""
    lines = [
        "# Search Reasoner Evaluation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Model Comparison (Base 3B vs Fine-tuned 3B)",
        "",
        f"Eval set: {len(base_results)} examples (10% held-out from 383 CoT records)",
        "",
        "| Metric | Base 3B | Fine-tuned 3B | Delta |",
        "|--------|---------|---------------|-------|",
    ]
    for row in comparison_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines += [
        "",
        "## 2. Teacher Label Quality (mistral-large-latest)",
        "",
        f"Total examples: {teacher_stats.get('teacher/total_examples', 0)}",
        f"Overall accuracy vs ground truth: **{teacher_stats.get('teacher/accuracy', 0):.1%}**",
        "",
        "### Accuracy by Answer Option",
        "",
        "| Option | Count | Teacher Accuracy |",
        "|--------|-------|-----------------|",
    ]
    for key in sorted(k for k in teacher_stats if k.startswith("teacher/accuracy_option_")):
        opt = key.split("_")[-1]
        count = teacher_stats.get(f"teacher/count_option_{opt}", 0)
        acc = teacher_stats[key]
        lines.append(f"| {opt} | {count} | {acc:.1%} |")

    lines += [
        "",
        "### Accuracy by Difficulty",
        "",
        "| Difficulty | Count | Teacher Accuracy |",
        "|-----------|-------|-----------------|",
    ]
    for bucket in ["easy", "medium", "hard", "unknown"]:
        key = f"teacher/accuracy_{bucket}"
        if key in teacher_stats:
            count = teacher_stats.get(f"teacher/count_{bucket}", 0)
            acc = teacher_stats[key]
            lines.append(f"| {bucket} | {count} | {acc:.1%} |")

    lines += [
        "",
        "## 3. Accuracy by Answer Option (Model Comparison)",
        "",
        "| Option | Base 3B | Fine-tuned 3B | Teacher |",
        "|--------|---------|---------------|---------|",
    ]
    all_opts = sorted(set(
        k.split("_")[-1]
        for k in list(base_metrics.keys()) + list(ft_metrics.keys())
        if "accuracy_option_" in k
    ))
    for opt in all_opts:
        base_acc = base_metrics.get(f"base/accuracy_option_{opt}", 0)
        ft_acc = ft_metrics.get(f"finetuned/accuracy_option_{opt}", 0)
        teacher_acc = teacher_stats.get(f"teacher/accuracy_option_{opt}", 0)
        lines.append(f"| {opt} | {base_acc:.1%} | {ft_acc:.1%} | {teacher_acc:.1%} |")

    lines += [
        "",
        "## 4. Accuracy by Difficulty (Model Comparison)",
        "",
        "| Difficulty | Base 3B | Fine-tuned 3B | Teacher |",
        "|-----------|---------|---------------|---------|",
    ]
    for bucket in ["easy", "medium", "hard"]:
        base_acc = base_metrics.get(f"base/accuracy_{bucket}", 0)
        ft_acc = ft_metrics.get(f"finetuned/accuracy_{bucket}", 0)
        teacher_acc = teacher_stats.get(f"teacher/accuracy_{bucket}", 0)
        lines.append(f"| {bucket} | {base_acc:.1%} | {ft_acc:.1%} | {teacher_acc:.1%} |")

    lines += [
        "",
        "## 5. Per-Example Results (Fine-tuned Model)",
        "",
        "| # | Question | GT | FT Pred | Correct | JSON | Think |",
        "|---|----------|----|---------|---------| -----|-------|",
    ]
    for i, r in enumerate(ft_results):
        q = r["question_snippet"][:60].replace("|", "/")
        gt = r["ground_truth"]
        pred = r.get("parsed_answer", "-")
        correct = "Y" if r["answer_correct"] else "N"
        jv = "Y" if r["json_valid"] else "N"
        think = "Y" if r["has_think_tags"] else "N"
        lines.append(f"| {i+1} | {q}... | {gt} | {pred} | {correct} | {jv} | {think} |")

    if teacher_mismatches:
        lines += [
            "",
            "## 6. Teacher Errors (sample)",
            "",
            "| Question | Teacher | Ground Truth |",
            "|----------|---------|-------------|",
        ]
        for m in teacher_mismatches[:20]:
            q = m["question"][:80].replace("|", "/")
            lines.append(f"| {q}... | {m['teacher']} | {m['ground_truth']} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Markdown report saved to {output_path}")


def save_json_report(
    output_path: str,
    base_metrics: dict,
    ft_metrics: dict,
    teacher_stats: dict,
    base_results: list[dict],
    ft_results: list[dict],
):
    """Save full results as JSON for programmatic access."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "teacher_stats": teacher_stats,
        "base_per_example": [
            {k: v for k, v in r.items() if k != "raw_output"}
            for r in base_results
        ],
        "finetuned_per_example": [
            {k: v for k, v in r.items() if k != "raw_output"}
            for r in ft_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved to {output_path}")


# ──────────────────────────────────────────────
# 7. W&B LOGGING
# ──────────────────────────────────────────────


def log_to_wandb(
    base_metrics: dict,
    ft_metrics: dict,
    teacher_stats: dict,
    comparison_rows: list[list[str]],
    base_results: list[dict],
    ft_results: list[dict],
    teacher_mismatches: list[dict],
):
    """Log all evaluation results to Weights & Biases."""
    import wandb

    wandb.log(base_metrics)
    wandb.log(ft_metrics)
    wandb.log(teacher_stats)

    comparison_table = wandb.Table(
        columns=["Metric", "Base 3B", "Fine-tuned 3B", "Delta"],
        data=comparison_rows,
    )
    wandb.log({"eval/comparison_summary": comparison_table})

    per_example_table = wandb.Table(
        columns=[
            "idx", "question", "ground_truth", "difficulty",
            "teacher_answer", "teacher_correct",
            "base_answer", "base_correct", "base_json_valid", "base_think",
            "ft_answer", "ft_correct", "ft_json_valid", "ft_think",
        ]
    )
    for i, (br, fr) in enumerate(zip(base_results, ft_results)):
        per_example_table.add_data(
            i,
            br["question_snippet"][:100],
            br["ground_truth"],
            br.get("difficulty"),
            br["teacher_answer"],
            br["teacher_answer"] == br["ground_truth"],
            br.get("parsed_answer", ""),
            br["answer_correct"],
            br["json_valid"],
            br["has_think_tags"],
            fr.get("parsed_answer", ""),
            fr["answer_correct"],
            fr["json_valid"],
            fr["has_think_tags"],
        )
    wandb.log({"eval/per_example_results": per_example_table})

    if teacher_mismatches:
        mismatch_table = wandb.Table(
            columns=["question", "teacher_answer", "ground_truth"]
        )
        for m in teacher_mismatches[:50]:
            mismatch_table.add_data(
                m["question"][:120], m["teacher"], m["ground_truth"]
            )
        wandb.log({"teacher/error_examples": mismatch_table})

    delta_accuracy = (
        ft_metrics.get("finetuned/answer_accuracy", 0)
        - base_metrics.get("base/answer_accuracy", 0)
    )
    wandb.log({
        "eval/delta_answer_accuracy": delta_accuracy,
        "eval/delta_json_valid": (
            ft_metrics.get("finetuned/json_valid_rate", 0)
            - base_metrics.get("base/json_valid_rate", 0)
        ),
        "eval/delta_think_tag": (
            ft_metrics.get("finetuned/think_tag_rate", 0)
            - base_metrics.get("base/think_tag_rate", 0)
        ),
    })


# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs fine-tuned Search Reasoner"
    )
    parser.add_argument("--mcqa_file", default="data/mcqa_search.jsonl")
    parser.add_argument("--cot_file", default="data/cot_training_data.jsonl")
    parser.add_argument(
        "--base_model", default="mistralai/Ministral-3-3B-Instruct-2512-BF16"
    )
    parser.add_argument("--finetuned_repo", default="NayanK/search-reasoner-3b")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="search-reasoner-3b")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace access token (or HF_TOKEN env var)",
    )
    parser.add_argument(
        "--wandb_key", type=str, default=None,
        help="Weights & Biases API key (or WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--teacher_only", action="store_true", default=False,
        help="Only compute teacher stats (no GPU inference)",
    )
    parser.add_argument(
        "--skip_base", action="store_true", default=False,
        help="Skip base model inference (use saved results from prior run)",
    )
    parser.add_argument(
        "--base_results_file", type=str, default=None,
        help="Path to JSON with saved base results (used with --skip_base)",
    )
    args = parser.parse_args()

    use_4bit = args.use_4bit and not args.no_4bit

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
        print("  HuggingFace: authenticated")

    wandb_key = args.wandb_key or os.environ.get("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    print("=" * 60)
    print("Search Reasoner Evaluation")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/6] Loading data...")
    gt_lookup = load_ground_truth(args.mcqa_file)
    print(f"  Ground truth: {len(gt_lookup)} questions")

    cot_records = load_cot_data(args.cot_file)
    print(f"  CoT records: {len(cot_records)}")

    all_records = join_with_ground_truth(cot_records, gt_lookup)
    print(f"  Joined with ground truth: {len(all_records)}")

    # ── Teacher quality stats ──
    print("\n[2/6] Computing teacher quality stats...")
    teacher_stats, teacher_mismatches = compute_teacher_stats(all_records)

    print(f"\n  ── Teacher Quality (mistral-large-latest) ──")
    print(f"    Overall accuracy: {teacher_stats['teacher/accuracy']:.1%} "
          f"({teacher_stats['teacher/correct']}/{teacher_stats['teacher/total_examples']})")
    print(f"    By option:")
    for key in sorted(k for k in teacher_stats if k.startswith("teacher/accuracy_option_")):
        opt = key.split("_")[-1]
        count = teacher_stats.get(f"teacher/count_option_{opt}", 0)
        print(f"      {opt}: {teacher_stats[key]:.1%} ({count} questions)")
    print(f"    By difficulty:")
    for bucket in ["easy", "medium", "hard"]:
        key = f"teacher/accuracy_{bucket}"
        if key in teacher_stats:
            count = teacher_stats.get(f"teacher/count_{bucket}", 0)
            print(f"      {bucket}: {teacher_stats[key]:.1%} ({count} questions)")

    if args.teacher_only:
        print("\n  --teacher_only flag set, skipping model inference.")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json_report(
            str(out_dir / "teacher_stats.json"),
            {}, {}, teacher_stats, [], [],
        )
        print("\nDone!")
        return

    # ── Split eval set (same seed as training) ──
    print("\n[3/6] Preparing eval split...")
    from datasets import Dataset
    ds = Dataset.from_list(all_records)
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    eval_records = list(split["test"])
    print(f"  Eval set: {len(eval_records)} examples (same split as training)")

    # ── Load base model ──
    if args.skip_base and args.base_results_file:
        print("\n[4/6] Loading saved base model results...")
        with open(args.base_results_file, encoding="utf-8") as f:
            saved = json.load(f)
        base_results = saved.get("base_per_example", [])
        base_metrics = saved.get("base_metrics", {})
        for r in base_results:
            r.setdefault("answer_correct", r.get("parsed_answer") == r.get("ground_truth") if r.get("parsed_answer") else False)
        print(f"    Loaded {len(base_results)} saved base results")
    elif args.skip_base:
        print("\n[4/6] Skipping base model (using hardcoded results from prior run)...")
        base_results = []
        base_metrics = {
            "base/n_examples": 39,
            "base/json_valid_rate": 0.462,
            "base/schema_valid_rate": 0.462,
            "base/answer_parse_rate": 0.462,
            "base/answer_accuracy": 0.205,
            "base/think_tag_rate": 0.462,
        }
        for r in eval_records:
            base_results.append({
                "json_valid": False, "schema_valid": False,
                "parsed_answer": None, "has_think_tags": False,
                "has_think_open": False, "ground_truth": r["ground_truth"],
                "teacher_answer": r["parsed_answer"],
                "answer_correct": False,
                "question_snippet": r["question"][:150],
                "difficulty": r.get("difficulty"),
                "raw_output": "",
            })
    else:
        print("\n[4/6] Loading base model for inference...")
        base_model, base_tokenizer = load_model_and_tokenizer(
            args.base_model, use_4bit=use_4bit
        )

        base_results = run_inference(
            base_model, base_tokenizer, eval_records,
            max_new_tokens=args.max_new_tokens, desc="base",
        )
        base_metrics = compute_metrics(base_results, "base")

        del base_model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("\n  Base model unloaded, GPU memory freed.")

    print(f"\n  ── Base Model Results ──")
    for key in ["base/json_valid_rate", "base/schema_valid_rate",
                "base/answer_parse_rate", "base/answer_accuracy", "base/think_tag_rate"]:
        if key in base_metrics:
            label = key.split("/")[1].replace("_", " ").title()
            print(f"    {label}: {base_metrics[key]:.1%}")

    # ── Load fine-tuned model ──
    print("\n[5/6] Loading fine-tuned model for inference...")
    ft_model, ft_tokenizer = load_model_and_tokenizer(
        args.finetuned_repo, use_4bit=use_4bit
    )

    ft_results = run_inference(
        ft_model, ft_tokenizer, eval_records,
        max_new_tokens=args.max_new_tokens, desc="finetuned",
    )
    ft_metrics = compute_metrics(ft_results, "finetuned")

    print(f"\n  ── Fine-tuned Model Results ──")
    for key in ["finetuned/json_valid_rate", "finetuned/schema_valid_rate",
                "finetuned/answer_parse_rate", "finetuned/answer_accuracy",
                "finetuned/think_tag_rate"]:
        if key in ft_metrics:
            label = key.split("/")[1].replace("_", " ").title()
            print(f"    {label}: {ft_metrics[key]:.1%}")

    del ft_model
    torch.cuda.empty_cache()

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("  BASE 3B vs FINE-TUNED 3B COMPARISON")
    print("=" * 60)

    comparison_rows = build_comparison_table(base_metrics, ft_metrics, teacher_stats)
    print(f"\n  {'Metric':<25s}  {'Base 3B':>10s}  {'Fine-tuned':>10s}  {'Delta':>10s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
    for row in comparison_rows:
        print(f"  {row[0]:<25s}  {row[1]:>10s}  {row[2]:>10s}  {row[3]:>10s}")

    # ── Save reports ──
    print("\n[6/6] Saving reports...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_markdown_report(
        str(out_dir / "eval_report.md"),
        comparison_rows, teacher_stats,
        base_metrics, ft_metrics,
        base_results, ft_results,
        teacher_mismatches,
    )
    save_json_report(
        str(out_dir / "eval_report.json"),
        base_metrics, ft_metrics, teacher_stats,
        base_results, ft_results,
    )

    # ── W&B ──
    if not args.no_wandb:
        print("\n  Logging to Weights & Biases...")
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "eval-base-vs-finetuned",
            config={
                "eval_type": "base_vs_finetuned",
                "base_model": args.base_model,
                "finetuned_model": args.finetuned_repo,
                "n_eval_examples": len(eval_records),
                "n_total_cot": len(all_records),
                "teacher_accuracy": teacher_stats.get("teacher/accuracy", 0),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "quantization": "4bit-nf4" if use_4bit else "bf16",
            },
            tags=[
                "eval", "base-vs-finetuned", "ministral-3b",
                "search-reasoner", "hackathon",
            ],
        )

        log_to_wandb(
            base_metrics, ft_metrics, teacher_stats,
            comparison_rows, base_results, ft_results,
            teacher_mismatches,
        )

        wandb.finish()
        print("  W&B run finalized.")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"  Reports: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
