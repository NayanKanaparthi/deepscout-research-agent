#!/usr/bin/env python3
"""
CoT Training Data Generator — Teacher Distillation Pipeline

Joins search_query_data.jsonl(500) with outputs/search_dataset.jsonl,
calls mistral-large-latest to generate chain-of-thought reasoning traces,
and saves the results as JSONL for SFT training.

The teacher model receives each MCQA question alongside truncated search
results and scraped page content, then produces <think>...</think>
reasoning followed by a structured JSON answer.

Usage:
    export MISTRAL_API_KEY=...

    # Verify join + see a formatted prompt (no API calls)
    python generate_cot_dataset.py --dry-run

    # Test with 3 examples
    python generate_cot_dataset.py --limit 3

    # Full run (~502 examples, ~30-60 min)
    python generate_cot_dataset.py
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# System prompt — shared with train_search_reasoner.py
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Lockfile — prevents concurrent runs on the same output
# ---------------------------------------------------------------------------

_lock_fd = None


def acquire_lock(output_path: Path):
    global _lock_fd
    lock_path = output_path.parent / f".{output_path.stem}.lock"
    _lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Error: Another instance is already writing to {output_path}")
        print(f"  Lock file: {lock_path}")
        print("  Kill the other process first, or remove the lock file if stale.")
        sys.exit(1)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
    atexit.register(release_lock)


def release_lock():
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None


def _signal_handler(signum, frame):
    release_lock()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Data join
# ---------------------------------------------------------------------------


def load_query_data(path: str) -> list[dict]:
    """Load search_query_data.jsonl and extract query + full question."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                output = json.loads(rec["output"])
                records.append({
                    "query": output["query"].strip(),
                    "question": rec["user_prompt"],
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: skipping line {line_num} in query file: {e}")
    return records


def load_search_data(path: str) -> dict[str, dict]:
    """Load search_dataset.jsonl into a dict keyed by query string."""
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                lookup[rec["query"].strip()] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return lookup


def join_datasets(query_records: list[dict], search_lookup: dict[str, dict]) -> list[dict]:
    """Join query data with search data on the query string."""
    joined = []
    missed = 0
    for qrec in query_records:
        srec = search_lookup.get(qrec["query"])
        if srec is None:
            missed += 1
            continue
        joined.append({
            "question": qrec["question"],
            "search_query": qrec["query"],
            "search_results": srec["search_results"],
            "scraped_pages": srec["scraped_pages"],
        })
    print(f"  Join: {len(joined)} matched, {missed} unmatched")
    return joined


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def format_search_context(
    search_results: list[dict],
    scraped_pages: list[dict],
    max_results: int = 3,
    max_chars: int = 2000,
) -> str:
    """Build the search context string shown to the teacher/student model."""
    page_lookup = {}
    for page in scraped_pages:
        page_lookup[page["url"]] = page

    parts = []
    shown = 0
    for sr in search_results:
        if shown >= max_results:
            break

        url = sr.get("url", "")
        page = page_lookup.get(url, {})
        page_text = page.get("text", "")
        page_success = page.get("success", False)

        if not page_success or len(page_text) < 50:
            continue

        truncated = page_text[:max_chars]
        if len(page_text) > max_chars:
            truncated += "\n[truncated]"

        part = (
            f"Result {shown + 1}: {sr.get('title', 'Untitled')}\n"
            f"URL: {url}\n"
            f"Snippet: {sr.get('description', '')}\n"
            f"Page content:\n{truncated}"
        )
        parts.append(part)
        shown += 1

    if not parts:
        return "(No usable search results with scraped content)"

    return "\n\n---\n\n".join(parts)


def format_user_message(question: str, search_context: str) -> str:
    return f"## Question\n{question}\n\n## Search Results\n{search_context}"


# ---------------------------------------------------------------------------
# JSONL helpers (append-only)
# ---------------------------------------------------------------------------


def load_existing_queries(path: Path) -> set[str]:
    """Stream output file and return set of queries already processed."""
    queries = set()
    if not path.exists():
        return queries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                queries.add(rec["search_query"])
            except (json.JSONDecodeError, KeyError):
                pass
    return queries


def append_record(path: Path, record: dict, existing_queries: set[str]) -> bool:
    """Append a single record with duplicate guard."""
    q = record["search_query"]
    if q in existing_queries:
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    existing_queries.add(q)
    return True


# ---------------------------------------------------------------------------
# Teacher API call
# ---------------------------------------------------------------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


def call_teacher(
    user_message: str,
    api_key: str,
    model: str = "mistral-large-latest",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Optional[str]:
    """Call the Mistral teacher model. Returns the assistant content or None on failure."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SEARCH_REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"    HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"    Request error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _extract_json_answer(text: str) -> Optional[dict]:
    """Try to extract a valid answer JSON from text, checking fences first."""
    # Markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            answer = str(parsed.get("answer", "")).strip().upper()
            if answer in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"):
                return parsed
        except json.JSONDecodeError:
            pass

    # Raw JSON object
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    answer = str(parsed.get("answer", "")).strip().upper()
    if answer in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"):
        return parsed
    return None


def validate_teacher_response(response: str) -> Optional[dict]:
    """
    Validate the teacher response has <think>...</think> + valid JSON.
    Falls back to salvaging truncated responses that have <think> but
    got cut off before </think> (common with long reasoning).
    Returns the parsed JSON dict or None if invalid.
    """
    has_think_open = "<think>" in response
    has_think_close = "</think>" in response

    # Best case: proper <think>...</think> + JSON after
    if has_think_open and has_think_close:
        after_think = response.split("</think>", 1)[1].strip()
        result = _extract_json_answer(after_think)
        if result:
            return result

    # Fallback: <think> present but truncated before </think>
    # Try to find JSON anywhere in the response
    if has_think_open:
        result = _extract_json_answer(response)
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args):
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: MISTRAL_API_KEY required. Set env var or use --api-key.")
        sys.exit(1)

    print("=" * 60)
    print("CoT Training Data Generator")
    print("=" * 60)

    # Load and join
    print("\n[1/3] Loading datasets...")
    query_records = load_query_data(args.query_file)
    print(f"  Query file: {len(query_records)} records")

    search_lookup = load_search_data(args.search_file)
    print(f"  Search file: {len(search_lookup)} unique queries")

    joined = join_datasets(query_records, search_lookup)
    if not joined:
        print("Error: No records matched. Check your input files.")
        sys.exit(1)

    if args.limit:
        joined = joined[: args.limit]
        print(f"  Limited to first {args.limit} examples")

    # Dry run: show stats + one formatted prompt
    if args.dry_run:
        print("\n[DRY RUN] Join stats + sample prompt")
        print(f"  Total joinable: {len(joined)}")
        sample = joined[0]
        context = format_search_context(
            sample["search_results"],
            sample["scraped_pages"],
            max_results=args.max_results,
            max_chars=args.max_chars,
        )
        user_msg = format_user_message(sample["question"], context)
        print(f"\n{'─' * 60}")
        print("SYSTEM PROMPT:")
        print(SEARCH_REASONER_SYSTEM_PROMPT[:300] + "...")
        print(f"\n{'─' * 60}")
        print("USER MESSAGE:")
        print(user_msg[:2000])
        if len(user_msg) > 2000:
            print(f"... ({len(user_msg)} chars total, ~{len(user_msg) // 4} tokens)")
        print(f"\n{'─' * 60}")
        print(f"Total examples to process: {len(joined)}")
        print(f"Estimated cost: ~${len(joined) * 0.016:.2f} (rough)")
        return

    # Setup output
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acquire_lock(out_path)

    existing = load_existing_queries(out_path)
    if existing:
        print(f"\n  Resuming: {len(existing)} records already in {out_path}")

    # Process
    print(f"\n[2/3] Generating CoT for {len(joined)} examples...")
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, rec in enumerate(joined):
        query = rec["search_query"]

        if query in existing:
            skipped += 1
            continue

        context = format_search_context(
            rec["search_results"],
            rec["scraped_pages"],
            max_results=args.max_results,
            max_chars=args.max_chars,
        )
        user_msg = format_user_message(rec["question"], context)

        elapsed = time.time() - start_time
        rate = processed / max(elapsed, 1)
        remaining = len(joined) - i
        eta = remaining / max(rate, 0.01)
        print(
            f"\n[{i + 1}/{len(joined)}] query: \"{query[:60]}...\" "
            f"(done={processed}, skip={skipped}, fail={failed}, "
            f"ETA={eta / 60:.0f}m)"
        )

        response = call_teacher(
            user_msg,
            api_key=api_key,
            model=args.teacher_model,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )

        if response is None:
            print("    FAILED: no response from teacher")
            failed += 1
            time.sleep(1)
            continue

        parsed_json = validate_teacher_response(response)
        if parsed_json is None:
            print(f"    FAILED: invalid response format (first 200 chars: {response[:200]})")
            failed += 1
            time.sleep(1)
            continue

        answer = str(parsed_json["answer"]).strip().upper()

        # Ensure well-formed <think>...</think> for training data
        clean_response = response
        if "<think>" in response and "</think>" not in response:
            json_str = json.dumps(parsed_json, indent=2)
            think_content = response.split("<think>", 1)[1]
            clean_response = f"<think>\n{think_content.strip()}\n</think>\n\n```json\n{json_str}\n```"

        output_record = {
            "question": rec["question"],
            "search_query": query,
            "search_context": context,
            "teacher_response": clean_response,
            "parsed_answer": answer,
            "teacher_model": args.teacher_model,
            "num_results_shown": args.max_results,
            "max_chars_per_page": args.max_chars,
        }

        written = append_record(out_path, output_record, existing)
        if written:
            processed += 1
            print(f"    OK: answer={answer}, confidence={parsed_json.get('confidence', '?')}")
        else:
            skipped += 1

        time.sleep(1)

    # Summary
    total_time = time.time() - start_time
    total_in_file = len(existing)
    print(f"\n{'=' * 60}")
    print("[3/3] Done!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed (bad response): {failed}")
    print(f"  Total records in file: {total_in_file}")
    print(f"  Time: {total_time / 60:.1f} minutes")
    print(f"  Output: {out_path}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT training data via teacher distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query-file",
        default="data/search_query_data.jsonl",
        help="Path to search_query_data JSONL file",
    )
    parser.add_argument(
        "--search-file",
        default="data/search_dataset.jsonl",
        help="Path to search_dataset JSONL file",
    )
    parser.add_argument(
        "--output-file",
        default="data/cot_training_data.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--teacher-model",
        default="mistral-large-latest",
        help="Mistral model to use as teacher",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Max search results to include in context",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max chars per scraped page",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max API call retries per example",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Teacher model temperature",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N examples (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show join stats + sample prompt, no API calls",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mistral API key (or MISTRAL_API_KEY env var)",
    )
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
