#!/usr/bin/env python3
"""
Baseline Evaluation — 100 Queries via Mistral API (mistral-small-latest)

Chain:
  data/mcqa_search.jsonl          → original question + options + ground truth (expected_answer)
  data/search_query_data.jsonl    → maps question → search query
  data/search_dataset.jsonl       → search query → 10 Brave results + scraped content

Flow:
  1. Join all three datasets
  2. Send original MCQ question + 10 search results to model
  3. Model reasons (<think>) and picks answer letter (A/B/C/D/E...)
  4. Compare model's letter vs ground truth
  5. Report accuracy

Outputs:
  - outputs/baseline_100_results.json
  - outputs/baseline_100_results.csv
  - outputs/baseline_100_accuracy.md
"""

import csv
import json
import os
import re
import time

from mistralai import Mistral

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MODEL = "mistral-small-latest"
NUM_QUERIES = 100
MAX_CHARS_PER_PAGE = 2500
DELAY_BETWEEN = 2
MAX_RETRIES = 5

SYSTEM_PROMPT = """\
You are a Search Result Reasoning Agent. Your job is to analyze web search results and their scraped content to answer a user's multiple-choice question accurately.

## Process

You MUST think step-by-step inside <think>...</think> tags before giving your final answer. Your reasoning should:

1. **Evaluate each search result**: For each of the provided search results, assess:
   - Is the source credible and authoritative for this topic?
   - Does the title/snippet suggest it contains relevant information?
   - Does the scraped page content actually contain the answer?
   - Rate each result: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT

2. **Identify answer-bearing results**: Which specific results contain information that directly answers the question? Quote the relevant passages.

3. **Cross-reference**: Do multiple sources agree? Are there contradictions? Which source is most trustworthy?

4. **Synthesize**: Combine information from the best sources to determine the correct answer option.

## Output Format

After your <think>...</think> reasoning, provide your answer in this exact JSON format:

{
  "result_rankings": [
    {"rank": 1, "result_index": <0-indexed>, "relevance": "HIGHLY_RELEVANT", "reason": "..."},
    {"rank": 2, "result_index": <0-indexed>, "relevance": "SOMEWHAT_RELEVANT", "reason": "..."}
  ],
  "best_result_index": <0-indexed int>,
  "answer_letter": "<A, B, C, D, E, F, G, H, I, or J>",
  "answer_text": "<the full text of the chosen option>",
  "confidence": <0.0 to 1.0>,
  "supporting_evidence": ["<quote from source 1>", "<quote from source 2>"]
}

## CRITICAL RULES
- You MUST pick exactly one answer letter from the given options
- The "answer_letter" field MUST be a single capital letter (A, B, C, D, E, F, G, H, I, or J)
- Always evaluate ALL search results before deciding
- If no result is helpful, use your best judgment based on snippets
- Be specific — cite which result(s) informed your answer"""


# ──────────────────────────────────────────────
# Data loading & joining
# ──────────────────────────────────────────────


def load_jsonl(path, max_n=None):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_n and len(records) >= max_n:
                    break
    return records


def build_paired_dataset(max_pairs=NUM_QUERIES):
    """
    Join the 3 datasets:
      data/mcqa_search.jsonl      → question text + GT answer letter
      data/search_query_data.jsonl → question text → search query
      data/search_dataset.jsonl   → search query → 10 results + scraped pages

    Returns list of dicts with all the info we need.
    """
    print("  Loading data/mcqa_search.jsonl...")
    mcqa = load_jsonl("data/mcqa_search.jsonl")
    print(f"    {len(mcqa)} records")

    print("  Loading data/search_query_data.jsonl...")
    sqd = load_jsonl("data/search_query_data.jsonl")
    print(f"    {len(sqd)} records")

    print("  Loading data/search_dataset.jsonl...")
    search = load_jsonl("data/search_dataset.jsonl")
    print(f"    {len(search)} records")

    # Build mcqa index: first 80 chars of input → mcqa record
    mcqa_index = {}
    for m in mcqa:
        inp = m["responses_create_params"]["input"]
        key = inp[:80].lower().strip()
        mcqa_index[key] = m

    # Build search index: query → search record
    search_index = {}
    for s in search:
        key = s["query"].lower().strip()
        search_index[key] = s

    # Join: sqd → mcqa (via question text) + search (via query)
    paired = []
    no_mcqa = 0
    no_search = 0

    for rec in sqd:
        question = rec["user_prompt"]
        search_query = json.loads(rec["output"])["query"]

        # Find mcqa match
        key = question[:80].lower().strip()
        m = mcqa_index.get(key)
        if not m:
            no_mcqa += 1
            continue

        # Find search match
        s = search_index.get(search_query.lower().strip())
        if not s:
            no_search += 1
            continue

        gt = m["expected_answer"].strip().upper()
        full_question = m["responses_create_params"]["input"]
        difficulty = m.get("task_difficulty_qwen3_32b_avg_8")

        paired.append({
            "question": full_question,
            "search_query": search_query,
            "ground_truth": gt,
            "difficulty": difficulty,
            "search_record": s,
        })

        if len(paired) >= max_pairs:
            break

    print(f"\n  Paired: {len(paired)} queries")
    print(f"  Skipped (no mcqa match): {no_mcqa}")
    print(f"  Skipped (no search data): {no_search}")

    return paired


# ──────────────────────────────────────────────
# Message building
# ──────────────────────────────────────────────


def build_user_message(question, search_record):
    scraped_map = {}
    for page in search_record.get("scraped_pages", []):
        if page.get("success") and page.get("text"):
            scraped_map[page["url"]] = page["text"]

    parts = []
    parts.append("## Original Question\n")
    parts.append(question)
    parts.append(f"\n## Search Results for: \"{search_record['query']}\"\n")

    for i, r in enumerate(search_record.get("search_results", [])):
        parts.append(f"### Result {i}")
        parts.append(f"**Title**: {r.get('title', 'N/A')}")
        parts.append(f"**URL**: {r.get('url', 'N/A')}")
        parts.append(f"**Snippet**: {r.get('description', 'N/A')}")
        url = r.get("url", "")
        if url in scraped_map:
            text = scraped_map[url][:MAX_CHARS_PER_PAGE]
            if len(scraped_map[url]) > MAX_CHARS_PER_PAGE:
                text += "\n[...truncated...]"
            parts.append(f"**Scraped Content**:\n{text}")
        else:
            parts.append("**Scraped Content**: [unavailable]")
        parts.append("")

    return "\n".join(parts)


# ──────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────


def extract_letter(text):
    """Extract answer letter from model response. Multiple strategies."""
    if not text:
        return None

    # Strategy 1: Parse JSON answer_letter
    json_str = None
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()

    if json_str:
        try:
            parsed = json.loads(json_str)
            letter = parsed.get("answer_letter", "")
            if letter and len(str(letter)) == 1 and str(letter).upper() in "ABCDEFGHIJ":
                return str(letter).upper()
        except json.JSONDecodeError:
            pass

        # Regex fallback on malformed JSON
        match = re.search(r'"answer_letter"\s*:\s*"([A-Ja-j])"', json_str)
        if match:
            return match.group(1).upper()

    # Strategy 2: Regex on full text for answer_letter field
    match = re.search(r'"answer_letter"\s*:\s*"([A-Ja-j])"', text)
    if match:
        return match.group(1).upper()

    # Strategy 3: Common answer patterns
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]?\s*\(?([A-Ja-j])\)?",
        r"(?:I\s+(?:would\s+)?(?:choose|select|pick))\s+(?:option\s+)?\(?([A-Ja-j])\)?",
        r"Option\s+\(?([A-Ja-j])\)?\s+is\s+(?:the\s+)?(?:correct|best)",
        r"\*\*([A-Ja-j])\*\*",
        r"\\boxed\{([A-Ja-j])\}",
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def parse_response(text):
    result = {
        "has_think": "<think>" in text and "</think>" in text,
        "has_json": False,
        "answer_letter": None,
        "confidence": None,
        "best_result_index": None,
        "num_rankings": 0,
    }

    json_str = None
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()

    if json_str:
        try:
            parsed = json.loads(json_str)
            result["has_json"] = True
            result["confidence"] = parsed.get("confidence")
            result["best_result_index"] = parsed.get("best_result_index")
            result["num_rankings"] = len(parsed.get("result_rankings", []))
        except json.JSONDecodeError:
            # Still try to extract confidence from malformed JSON
            match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str or "")
            if match:
                result["confidence"] = float(match.group(1))

    result["answer_letter"] = extract_letter(text)
    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    if not API_KEY:
        print("ERROR: MISTRAL_API_KEY env var required.")
        return
    print("=" * 70)
    print(f"  Baseline Evaluation — {NUM_QUERIES} Queries")
    print(f"  Model: {MODEL}")
    print(f"  Ground truth: mcqa_search.jsonl")
    print("=" * 70)

    # ── Build dataset ──
    print("\n[1/3] Building paired dataset...")
    paired = build_paired_dataset(NUM_QUERIES)

    if not paired:
        print("ERROR: No paired data. Check your files.")
        return

    actual_n = len(paired)
    print(f"\n  Will evaluate {actual_n} queries")
    print(f"  Sample GTs: {[p['ground_truth'] for p in paired[:10]]}")

    # ── Run inference ──
    print(f"\n[2/3] Running inference...")
    print(f"  Estimated time: ~{actual_n * 15 / 60:.0f} minutes\n")

    client = Mistral(api_key=API_KEY)
    all_results = []
    correct = 0
    incorrect = 0
    no_answer = 0
    json_ok = 0
    think_ok = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_total = time.time()

    for idx, pair in enumerate(paired):
        gt = pair["ground_truth"]
        query = pair["search_query"]

        print(f"  [{idx + 1}/{actual_n}] {query[:60]}...", end="")

        user_msg = build_user_message(pair["question"], pair["search_record"])

        # API call with retry
        text = ""
        usage = None
        error = None
        start = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.complete(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                )
                text = response.choices[0].message.content
                usage = response.usage
                error = None
                break
            except Exception as e:
                error = str(e)
                if "429" in error or "rate" in error.lower():
                    wait = 2 ** attempt * 5
                    print(f"\n    ⏳ Rate limited, waiting {wait}s...", end="")
                    time.sleep(wait)
                else:
                    print(f"\n    ✗ ERROR: {error}")
                    break

        elapsed = time.time() - start
        prompt_tok = usage.prompt_tokens if usage else 0
        comp_tok = usage.completion_tokens if usage else 0
        total_prompt_tokens += prompt_tok
        total_completion_tokens += comp_tok

        parsed = parse_response(text)
        model_letter = parsed["answer_letter"]

        if parsed["has_think"]:
            think_ok += 1
        if parsed["has_json"]:
            json_ok += 1

        # Compare
        if model_letter:
            if model_letter == gt:
                correct += 1
                status = "✓"
            else:
                incorrect += 1
                status = "✗"
            print(f" GT={gt} Model={model_letter} {status} ({elapsed:.1f}s)")
        else:
            no_answer += 1
            status = "?"
            print(f" GT={gt} Model=— ? ({elapsed:.1f}s)")

        all_results.append({
            "index": idx,
            "query": query,
            "question_preview": pair["question"][:200],
            "ground_truth": gt,
            "model_letter": model_letter,
            "is_correct": model_letter == gt if model_letter else None,
            "status": status,
            "has_think": parsed["has_think"],
            "has_json": parsed["has_json"],
            "confidence": parsed["confidence"],
            "best_result_index": parsed["best_result_index"],
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "difficulty": pair.get("difficulty"),
            "error": error,
            "raw_response": text,
        })

        # Running accuracy every 10
        evaluated = correct + incorrect
        if evaluated > 0 and (idx + 1) % 10 == 0:
            acc = correct / evaluated
            print(f"\n  ── After {idx + 1}: {correct}/{evaluated} = {acc:.1%} "
                  f"(think={think_ok}/{idx+1} json={json_ok}/{idx+1}) ──\n")

        if idx < actual_n - 1:
            time.sleep(DELAY_BETWEEN)

    total_elapsed = time.time() - start_total
    evaluated = correct + incorrect
    accuracy = correct / evaluated if evaluated > 0 else 0

    # ── Report ──
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Model:                {MODEL}")
    print(f"  Total queries:        {actual_n}")
    print(f"  ─────────────────────────────────────")
    print(f"  ✓ Correct:            {correct}")
    print(f"  ✗ Incorrect:          {incorrect}")
    print(f"  ? No answer:          {no_answer}")
    print(f"  ─────────────────────────────────────")
    print(f"  📊 ACCURACY:          {correct}/{evaluated} = {accuracy:.1%}")
    print(f"  ─────────────────────────────────────")
    print(f"  Think rate:           {think_ok}/{actual_n} = {think_ok/actual_n:.0%}")
    print(f"  JSON rate:            {json_ok}/{actual_n} = {json_ok/actual_n:.0%}")
    print(f"  Letter extract rate:  {evaluated}/{actual_n} = {evaluated/actual_n:.0%}")
    confs = [r["confidence"] for r in all_results if r["confidence"] is not None]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"  Avg confidence:       {avg_conf:.2f}")
    print(f"  Total time:           {total_elapsed / 60:.1f} min")
    print(f"  Total tokens:         {total_prompt_tokens + total_completion_tokens:,}")

    # Confidence breakdown
    correct_confs = [r["confidence"] for r in all_results if r["is_correct"] and r["confidence"]]
    wrong_confs = [r["confidence"] for r in all_results if r["is_correct"] is False and r["confidence"]]
    if correct_confs:
        print(f"  Conf when CORRECT:    {sum(correct_confs)/len(correct_confs):.2f} (n={len(correct_confs)})")
    if wrong_confs:
        print(f"  Conf when WRONG:      {sum(wrong_confs)/len(wrong_confs):.2f} (n={len(wrong_confs)})")

    # Difficulty breakdown
    easy = [r for r in all_results if r["difficulty"] is not None and r["difficulty"] < 0.4]
    medium = [r for r in all_results if r["difficulty"] is not None and 0.4 <= r["difficulty"] <= 0.7]
    hard = [r for r in all_results if r["difficulty"] is not None and r["difficulty"] > 0.7]
    for label, group in [("Easy (<0.4)", easy), ("Medium (0.4-0.7)", medium), ("Hard (>0.7)", hard)]:
        if group:
            g_correct = sum(1 for r in group if r["is_correct"])
            g_eval = sum(1 for r in group if r["is_correct"] is not None)
            if g_eval:
                print(f"  {label:20s}: {g_correct}/{g_eval} = {g_correct/g_eval:.1%}")

    print("=" * 70)

    # ── Save JSON ──
    output = {
        "meta": {
            "model": MODEL,
            "num_queries": actual_n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": round(total_elapsed, 2),
        },
        "accuracy": {
            "correct": correct,
            "incorrect": incorrect,
            "no_answer": no_answer,
            "evaluated": evaluated,
            "accuracy": round(accuracy, 4),
        },
        "format": {
            "think_rate": round(think_ok / actual_n, 3),
            "json_rate": round(json_ok / actual_n, 3),
            "letter_rate": round(evaluated / actual_n, 3),
            "avg_confidence": round(avg_conf, 3),
        },
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens,
        },
        "results": all_results,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_100_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved → outputs/baseline_100_results.json")

    # ── Save CSV ──
    with open("outputs/baseline_100_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "query", "ground_truth", "model_answer",
            "is_correct", "has_think", "has_json", "confidence",
            "difficulty", "time_s", "tokens",
        ])
        for r in all_results:
            writer.writerow([
                r["index"], r["query"][:80], r["ground_truth"],
                r["model_letter"] or "", r["is_correct"],
                r["has_think"], r["has_json"], r["confidence"],
                r["difficulty"], r["elapsed_s"],
                r["prompt_tokens"] + r["completion_tokens"],
            ])
    print(f"✓ Saved → outputs/baseline_100_results.csv")

    # ── Save Markdown ──
    md = []
    md.append("# Baseline Accuracy Report — mistral-small-latest\n")
    md.append(f"**Model**: `{MODEL}` (no fine-tuning)")
    md.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Queries evaluated**: {actual_n}\n")

    md.append("## Accuracy\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| ✓ Correct | {correct} |")
    md.append(f"| ✗ Incorrect | {incorrect} |")
    md.append(f"| ? No answer | {no_answer} |")
    md.append(f"| **Accuracy** | **{accuracy:.1%}** ({correct}/{evaluated}) |")
    md.append(f"| Think rate | {think_ok}/{actual_n} ({think_ok/actual_n:.0%}) |")
    md.append(f"| JSON rate | {json_ok}/{actual_n} ({json_ok/actual_n:.0%}) |")
    md.append(f"| Avg confidence | {avg_conf:.2f} |")
    md.append(f"| Total time | {total_elapsed/60:.1f} min |")
    md.append(f"| Total tokens | {total_prompt_tokens + total_completion_tokens:,} |\n")

    md.append("## Per-Query Results\n")
    md.append("| # | Status | GT | Model | Conf | Diff | Query |")
    md.append("|---|--------|----|----- -|------|------|-------|")
    for r in all_results:
        q = r["query"][:45] + ("..." if len(r["query"]) > 45 else "")
        d = f"{r['difficulty']:.1f}" if r["difficulty"] is not None else "—"
        md.append(
            f"| {r['index']} | {r['status']} | {r['ground_truth']} | "
            f"{r['model_letter'] or '—'} | {r['confidence'] or '—'} | {d} | {q} |"
        )

    # Wrong answers
    wrong = [r for r in all_results if r["is_correct"] is False]
    if wrong:
        md.append(f"\n## Wrong Answers ({len(wrong)})\n")
        for r in wrong:
            md.append(f"- **[{r['index']}]** GT={r['ground_truth']} Model={r['model_letter']} "
                      f"conf={r['confidence']} | {r['query'][:70]}")

    with open("outputs/baseline_100_accuracy.md", "w") as f:
        f.write("\n".join(md))
    print(f"✓ Saved → outputs/baseline_100_accuracy.md")


if __name__ == "__main__":
    main()
