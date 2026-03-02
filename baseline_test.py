#!/usr/bin/env python3
"""
Baseline Test — Single query inference via Mistral API (no fine-tuning).
"""

import json
import time

from mistralai import Mistral

API_KEY = "x7VUmwCrgXfOubAu6CLodGFMUqj7cioi"
MODEL = "mistral-small-latest"

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


def main():
    # ── Load ONE record ──
    with open("outputs/search_dataset.jsonl") as f:
        record = json.loads(f.readline())

    query = record["query"]
    search_results = record["search_results"]
    scraped_pages = record["scraped_pages"]

    print("=" * 60)
    print(f"  Query: {query}")
    print(f"  Results: {len(search_results)} | Scraped: {len(scraped_pages)}")
    print("=" * 60)

    # ── Build user message ──
    scraped_map = {}
    for page in scraped_pages:
        if page.get("success") and page.get("text"):
            scraped_map[page["url"]] = page["text"]

    parts = [f"## Search Query: {query}\n"]
    for i, r in enumerate(search_results):
        parts.append(f"### Result {i}")
        parts.append(f"**Title**: {r.get('title', 'N/A')}")
        parts.append(f"**URL**: {r.get('url', 'N/A')}")
        parts.append(f"**Snippet**: {r.get('description', 'N/A')}")
        url = r.get("url", "")
        if url in scraped_map:
            text = scraped_map[url][:3000]
            if len(scraped_map[url]) > 3000:
                text += "\n[...truncated...]"
            parts.append(f"**Scraped Content**:\n{text}")
        else:
            parts.append("**Scraped Content**: [unavailable]")
        parts.append("")

    user_message = "\n".join(parts)

    # ── Call Mistral API ──
    print("\nCalling Mistral API...")
    client = Mistral(api_key=API_KEY)

    start = time.time()
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    elapsed = time.time() - start

    text = response.choices[0].message.content
    usage = response.usage

    print(f"✓ Done in {elapsed:.1f}s | Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out\n")
    print("=" * 60)
    print("FULL RESPONSE:")
    print("=" * 60)
    print(text)
    print("=" * 60)

    # ── Save ──
    out = {
        "query": query,
        "model": MODEL,
        "elapsed_s": round(elapsed, 2),
        "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
        "response": text,
    }
    with open("outputs/baseline_single.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved → outputs/baseline_single.json")


if __name__ == "__main__":
    main()
