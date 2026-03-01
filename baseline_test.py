#!/usr/bin/env python3
"""
Baseline Test — Run N queries through Mistral API (no fine-tuning).

Outputs:
  - outputs/baseline_results.json   (full structured results)
  - outputs/baseline_results.csv    (quick comparison table)
  - outputs/baseline_summary.md     (human-readable report)
"""

import csv
import json
import time

from mistralai import Mistral

API_KEY = "x7VUmwCrgXfOubAu6CLodGFMUqj7cioi"
MODEL = "mistral-small-latest"
NUM_QUERIES = 10

SYSTEM_PROMPT = """\
You are a Search Result Reasoning Agent. Your job is to analyze web search results and their scraped content to answer a user's question accurately.

## Process

You MUST think step-by-step inside <think>...</think> tags before giving your final answer. Your reasoning should:

1. **Evaluate each search result**: For each of the provided search results, assess:
   - Is the source credible and authoritative for this topic?
   - Does the title/snippet suggest it contains relevant information?
   - Does the scraped page content actually contain the answer?
   - Rate each result: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT

2. **Identify answer-bearing results**: Which specific results contain information that directly answers the question? Quote the relevant passages.

3. **Cross-reference**: Do multiple sources agree? Are there contradictions? Which source is most trustworthy?

4. **Synthesize**: Combine information from the best sources into a coherent answer.

## Output Format

After your <think>...</think> reasoning, provide your answer in this exact JSON format:

{
  "result_rankings": [
    {"rank": 1, "result_index": <0-indexed>, "relevance": "HIGHLY_RELEVANT", "reason": "..."},
    {"rank": 2, "result_index": <0-indexed>, "relevance": "SOMEWHAT_RELEVANT", "reason": "..."},
    ...
  ],
  "best_result_index": <0-indexed int>,
  "answer": "<your synthesized answer>",
  "confidence": <0.0 to 1.0>,
  "supporting_evidence": ["<quote from source 1>", "<quote from source 2>"]
}"""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def load_records(path, n):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if len(records) >= n:
                    break
    return records


def build_user_message(record, max_chars=3000):
    scraped_map = {}
    for page in record.get("scraped_pages", []):
        if page.get("success") and page.get("text"):
            scraped_map[page["url"]] = page["text"]

    parts = [f"## Search Query: {record['query']}\n"]
    for i, r in enumerate(record.get("search_results", [])):
        parts.append(f"### Result {i}")
        parts.append(f"**Title**: {r.get('title', 'N/A')}")
        parts.append(f"**URL**: {r.get('url', 'N/A')}")
        parts.append(f"**Snippet**: {r.get('description', 'N/A')}")
        url = r.get("url", "")
        if url in scraped_map:
            text = scraped_map[url][:max_chars]
            if len(scraped_map[url]) > max_chars:
                text += "\n[...truncated...]"
            parts.append(f"**Scraped Content**:\n{text}")
        else:
            parts.append("**Scraped Content**: [unavailable]")
        parts.append("")
    return "\n".join(parts)


def parse_response(text):
    result = {
        "has_think": "<think>" in text and "</think>" in text,
        "has_json": False,
        "parsed_json": None,
        "answer": None,
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
            result["parsed_json"] = parsed
            result["answer"] = parsed.get("answer")
            result["confidence"] = parsed.get("confidence")
            result["best_result_index"] = parsed.get("best_result_index")
            result["num_rankings"] = len(parsed.get("result_rankings", []))
        except json.JSONDecodeError:
            pass
    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    print("=" * 60)
    print(f"  Baseline Evaluation — {NUM_QUERIES} Queries")
    print(f"  Model: {MODEL}")
    print("=" * 60)

    records = load_records("outputs/search_dataset.jsonl", NUM_QUERIES)
    print(f"\nLoaded {len(records)} records\n")

    client = Mistral(api_key=API_KEY)
    all_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0

    for idx, record in enumerate(records):
        query = record["query"]
        print(f"[{idx + 1}/{NUM_QUERIES}] {query[:70]}...")

        user_msg = build_user_message(record)

        # Retry with exponential backoff for rate limits
        text = ""
        usage = None
        error = None
        max_retries = 5
        start = time.time()

        for attempt in range(max_retries):
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
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                    print(f"  ⏳ Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"  ✗ ERROR: {error}")
                    break

        elapsed = time.time() - start

        parsed = parse_response(text)

        prompt_tok = usage.prompt_tokens if usage else 0
        comp_tok = usage.completion_tokens if usage else 0
        total_prompt_tokens += prompt_tok
        total_completion_tokens += comp_tok
        total_time += elapsed

        answer_preview = (parsed["answer"] or "N/A")[:80]
        print(f"  ✓ {elapsed:.1f}s | {prompt_tok}+{comp_tok} tok | "
              f"think={'✓' if parsed['has_think'] else '✗'} "
              f"json={'✓' if parsed['has_json'] else '✗'} | "
              f"conf={parsed['confidence']} | ans: {answer_preview}")

        all_results.append({
            "index": idx,
            "id": record.get("id"),
            "query": query,
            "model": MODEL,
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "has_think": parsed["has_think"],
            "has_json": parsed["has_json"],
            "answer": parsed["answer"],
            "confidence": parsed["confidence"],
            "best_result_index": parsed["best_result_index"],
            "num_rankings": parsed["num_rankings"],
            "raw_response": text,
            "structured_output": parsed["parsed_json"],
            "error": error,
        })

        # Delay between queries to avoid rate limits
        if idx < len(records) - 1:
            time.sleep(3)

    # ── Aggregate stats ──
    n = len(all_results)
    think_rate = sum(1 for r in all_results if r["has_think"]) / n
    json_rate = sum(1 for r in all_results if r["has_json"]) / n
    confidences = [r["confidence"] for r in all_results if r["confidence"] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    avg_time = total_time / n
    error_count = sum(1 for r in all_results if r["error"])

    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Queries:             {n}")
    print(f"  Think block rate:    {think_rate:.0%} ({sum(1 for r in all_results if r['has_think'])}/{n})")
    print(f"  JSON parse rate:     {json_rate:.0%} ({sum(1 for r in all_results if r['has_json'])}/{n})")
    print(f"  Avg confidence:      {avg_confidence:.2f}")
    print(f"  Avg response time:   {avg_time:.1f}s")
    print(f"  Total tokens:        {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"  Errors:              {error_count}")
    print("=" * 60)

    # ── Save JSON ──
    output = {
        "meta": {
            "model": MODEL,
            "num_queries": n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": SYSTEM_PROMPT,
        },
        "aggregate": {
            "think_rate": round(think_rate, 3),
            "json_parse_rate": round(json_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "avg_response_time_s": round(avg_time, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "errors": error_count,
        },
        "results": all_results,
    }
    with open("outputs/baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\n✓ Saved → outputs/baseline_results.json")

    # ── Save CSV ──
    with open("outputs/baseline_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "query", "has_think", "has_json",
            "confidence", "best_result", "answer", "time_s",
            "prompt_tok", "completion_tok",
        ])
        for r in all_results:
            writer.writerow([
                r["index"],
                r["query"][:80],
                r["has_think"],
                r["has_json"],
                r["confidence"],
                r["best_result_index"],
                (r["answer"] or "")[:120],
                r["elapsed_s"],
                r["prompt_tokens"],
                r["completion_tokens"],
            ])
    print("✓ Saved → outputs/baseline_results.csv")

    # ── Save Markdown report ──
    md = []
    md.append("# Baseline Evaluation Report")
    md.append(f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Model**: `{MODEL}` (no fine-tuning)")
    md.append(f"**Queries**: {n}")
    md.append(f"**System prompt**: Search Result Reasoning Agent with `<think>` CoT + JSON output\n")

    md.append("## Aggregate Metrics\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Think block rate | {think_rate:.0%} |")
    md.append(f"| JSON parse rate | {json_rate:.0%} |")
    md.append(f"| Avg confidence | {avg_confidence:.2f} |")
    md.append(f"| Avg response time | {avg_time:.1f}s |")
    md.append(f"| Total prompt tokens | {total_prompt_tokens:,} |")
    md.append(f"| Total completion tokens | {total_completion_tokens:,} |")
    md.append(f"| Total tokens | {total_prompt_tokens + total_completion_tokens:,} |")
    md.append(f"| Errors | {error_count} |\n")

    md.append("## Per-Query Results\n")
    md.append("| # | Query | Think | JSON | Conf | Best | Time | Answer |")
    md.append("|---|-------|-------|------|------|------|------|--------|")
    for r in all_results:
        q = r["query"][:50] + ("..." if len(r["query"]) > 50 else "")
        ans = (r["answer"] or "N/A")[:60] + ("..." if r["answer"] and len(r["answer"]) > 60 else "")
        md.append(
            f"| {r['index']} | {q} | "
            f"{'✓' if r['has_think'] else '✗'} | "
            f"{'✓' if r['has_json'] else '✗'} | "
            f"{r['confidence'] or 'N/A'} | "
            f"#{r['best_result_index'] if r['best_result_index'] is not None else '?'} | "
            f"{r['elapsed_s']}s | {ans} |"
        )

    md.append("\n## Detailed Results\n")
    for r in all_results:
        md.append(f"### Query {r['index']}: {r['query']}")
        md.append(f"- **Answer**: {r['answer'] or 'N/A'}")
        md.append(f"- **Confidence**: {r['confidence'] or 'N/A'}")
        md.append(f"- **Best result**: #{r['best_result_index'] if r['best_result_index'] is not None else '?'}")
        md.append(f"- **Time**: {r['elapsed_s']}s | Tokens: {r['prompt_tokens']}+{r['completion_tokens']}")
        if r["structured_output"] and r["structured_output"].get("supporting_evidence"):
            md.append(f"- **Evidence**:")
            for ev in r["structured_output"]["supporting_evidence"]:
                md.append(f"  - {str(ev)[:200]}")
        md.append("")

    with open("outputs/baseline_summary.md", "w") as f:
        f.write("\n".join(md))
    print("✓ Saved → outputs/baseline_summary.md")


if __name__ == "__main__":
    main()
