#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Real Trajectory Generator

  Uses REAL tool results instead of simulated ones:
    - search() → Brave Search API (real Google-quality results)
    - browse() → requests + BeautifulSoup (real page content)

  Mistral Large acts as the agent, making decisions based on
  actual web content. The resulting trajectories are grounded
  in real data — no hallucinated URLs or fake page content.

  Flow per sample:
    1. Load sample (system_prompt, user_prompt, tools)
    2. Send to Mistral Large with tool definitions
    3. Mistral calls search("query") → we hit Brave API → return real results
    4. Mistral calls browse("url") → we fetch + parse real page → return content
    5. Repeat until Mistral gives final answer
    6. Score each step with judge model
    7. Export: SFT + PRM training data

  Usage:
    export MISTRAL_API_KEY="..."
    export BRAVE_API_KEY="..."
    export open_ai_api_key="sk-..."    # for PRM scoring only

    # One sample test
    python generate_real_trajectories.py --num 1

    # Full run
    python generate_real_trajectories.py --num 2000 --workers 5
═══════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Optional

import requests
from logger import get_logger
from openai_api import OpenAIAPI
from prompts import SCORING_SYSTEM
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# REAL TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════


class BraveSearch:
    """Real search using Brave Search API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503]
                )
            ),
        )
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            }
        )

    def search(self, query: str, count: int = 10) -> str:
        """
        Search Brave and return formatted results.

        Returns text in the same format the Nemotron dataset expects:
        numbered results with titles, URLs, and snippets.
        """
        try:
            resp = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Brave search error: {e}")
            return f"Search error: {e}. Try a different query."

        results = data.get("web", {}).get("results", [])
        if not results:
            return "No search results found. Try a different query."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("description", "No description available.")
            formatted.append(f"{i}. {title}\n   {url}\n   {snippet}")

        return "\n\n".join(formatted)

    def close(self):
        self.session.close()


class WebBrowser:
    """Real page fetching with content extraction."""

    def __init__(self):
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503]
                )
            ),
        )
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def browse(self, url: str, max_words: int = 10000) -> str:
        """
        Fetch a URL and return cleaned text content.
        Truncates to max_words (matching Nemotron's browse tool spec).
        """
        try:
            resp = self.session.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            return f"Error: Page at {url} timed out after 15 seconds."
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to {url}."
        except requests.exceptions.HTTPError:
            return f"Error: {url} returned HTTP {resp.status_code}."
        except Exception as e:
            return f"Error fetching {url}: {str(e)[:200]}"

        # Extract text content
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"Error: {url} returned non-HTML content ({content_type})."

        text = self._extract_text(resp.text, url)

        # Truncate to max_words
        words = text.split()
        if len(words) > max_words:
            text = (
                " ".join(words[:max_words]) + "\n\n[Content truncated at 10,000 words]"
            )

        return text

    def _extract_text(self, html: str, url: str) -> str:
        """Extract clean text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            # Fallback: regex-based extraction
            return self._extract_text_regex(html, url)

        # Remove script, style, nav, footer, header
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "noscript"]
        ):
            tag.decompose()

        # Get title
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Try to find main content area
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.find(id="main-content")
            or soup.body
        )

        if main is None:
            main = soup

        # Extract text with some structure preserved
        lines = []
        if title:
            lines.append(f"Title: {title}")
            lines.append(f"URL: {url}")
            lines.append("")

        for element in main.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote"]
        ):
            text = element.get_text(separator=" ", strip=True)
            if not text or len(text) < 5:
                continue

            tag = element.name
            if tag in ("h1",):
                lines.append(f"\n# {text}")
            elif tag in ("h2",):
                lines.append(f"\n## {text}")
            elif tag in ("h3", "h4"):
                lines.append(f"\n### {text}")
            elif tag == "li":
                lines.append(f"  - {text}")
            else:
                lines.append(text)

        result = "\n".join(lines)

        # Clean up excessive whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)

        if len(result.strip()) < 100:
            return f"Title: {title}\nURL: {url}\n\n[Page content could not be extracted — may require JavaScript]"

        return result.strip()

    def _extract_text_regex(self, html: str, url: str) -> str:
        """Fallback text extraction using regex (no BS4)."""
        # Remove scripts and styles
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return f"URL: {url}\n\n{text}"

    def close(self):
        self.session.close()


def execute_tool(
    tool_name: str, arguments: dict, brave: BraveSearch, browser: WebBrowser
) -> str:
    """Execute a real tool call and return the result."""
    if tool_name == "search":
        query = arguments.get("query", "")
        log.info(f'    🔍 Brave search: "{query}"')
        return brave.search(query)

    elif tool_name == "browse":
        url = arguments.get("url", "")
        log.info(f"    🌐 Fetching: {url[:80]}")
        return browser.browse(url)

    else:
        return f"Error: Unknown tool '{tool_name}'"


# ═══════════════════════════════════════════════════════════════════
# MISTRAL LARGE CLIENT
# ═══════════════════════════════════════════════════════════════════


class MistralClient:
    """Thin wrapper for Mistral's OpenAI-compatible chat/completions with tools."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=2,
                    status_forcelist=[429, 500, 502, 503],
                    respect_retry_after_header=True,
                )
            ),
        )
        self.session.headers.update(
            {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        )

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Call Mistral's chat/completions with tool definitions.
        Returns the raw response dict.
        """
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self.session.post(
            "https://api.mistral.ai/v1/chat/completions", json=payload, timeout=60
        )

        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.text, "code": resp.status_code}

    def close(self):
        self.session.close()


# ═══════════════════════════════════════════════════════════════════
# AGENTIC LOOP (REAL TOOLS)
# ═══════════════════════════════════════════════════════════════════


def run_real_agentic_loop(
    mistral: MistralClient,
    brave: BraveSearch,
    browser: WebBrowser,
    sample: dict,
    max_turns: int = 8,
) -> dict:
    """
    Run Mistral Large in a tool-calling loop with REAL tool results.

    Each tool call hits real APIs:
      search() → Brave Search API
      browse() → actual HTTP fetch + HTML parsing
    """

    # Build initial messages
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_prompt"]},
    ]

    # Convert tools to OpenAI function format
    tools_for_api = []
    for tool in sample["tools"]:
        tools_for_api.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )

    trajectory = {
        "system_prompt": sample["system_prompt"],
        "user_prompt": sample["user_prompt"],
        "tools": sample["tools"],
        "turns": [],
    }

    log.info(f"  🔄 Starting agentic loop (max {max_turns} turns)")

    for turn_num in range(max_turns):
        log.info(f"  ── Turn {turn_num + 1} ──")

        # ── Call Mistral Large ──
        response = mistral.chat_with_tools(
            messages=messages, tools=tools_for_api, temperature=0.7
        )

        if "error" in response:
            log.error(f"  Mistral error: {response['error'][:200]}")
            break

        choice = response["choices"][0]
        assistant_msg = choice["message"]
        tool_calls = assistant_msg.get("tool_calls")

        if tool_calls:
            # ── Mistral wants to call a tool ──
            tc = tool_calls[0]
            func = tc.get("function", tc)
            tool_name = func.get("name", "")
            tool_args_raw = func.get("arguments", "{}")
            tool_call_id = tc.get("id", f"call_{turn_num:04d}")

            if isinstance(tool_args_raw, str):
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError:
                    tool_args = (
                        {"query": tool_args_raw}
                        if tool_name == "search"
                        else {"url": tool_args_raw}
                    )
            else:
                tool_args = tool_args_raw

            reasoning = assistant_msg.get("content", "") or ""

            log.info(
                f"  🤖 Agent: {reasoning[:100]}"
                if reasoning
                else "  🤖 Agent: (no reasoning)"
            )
            log.info(f"  🔧 Tool: {tool_name}({json.dumps(tool_args)})")

            # Record assistant turn
            trajectory["turns"].append(
                {
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [
                        {"id": tool_call_id, "name": tool_name, "arguments": tool_args}
                    ],
                }
            )

            # Add to messages for next turn
            messages.append(
                {
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                }
            )

            # ── Execute REAL tool ──
            result = execute_tool(tool_name, tool_args, brave, browser)
            log.info(f"  📄 Result: {result[:120]}...")

            # Record tool result
            trajectory["turns"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            )

            messages.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            )

        else:
            # ── Final answer (no tool call) ──
            final_content = assistant_msg.get("content", "")
            log.info(f"  ✅ Final: {final_content[:150]}...")

            trajectory["turns"].append(
                {"role": "assistant", "content": final_content, "tool_calls": None}
            )
            break

    else:
        log.warning("  ⚠️ Hit max turns without final answer")
        # Force a final answer note
        trajectory["turns"].append(
            {
                "role": "assistant",
                "content": "[Agent did not produce a final answer within max turns]",
                "tool_calls": None,
            }
        )

    # ── Metadata ──
    num_tool_calls = sum(
        1
        for t in trajectory["turns"]
        if t["role"] == "assistant" and t.get("tool_calls")
    )
    tools_used = list(
        set(
            tc["name"]
            for t in trajectory["turns"]
            if t["role"] == "assistant" and t.get("tool_calls")
            for tc in t["tool_calls"]
        )
    )

    trajectory["metadata"] = {
        "num_turns": len(trajectory["turns"]),
        "num_tool_calls": num_tool_calls,
        "tools_used": tools_used,
        "expected_answer": sample.get("expected_answer"),
    }

    log.info(f"  📊 Done: {num_tool_calls} tool calls, tools: {tools_used}")
    return trajectory


# ═══════════════════════════════════════════════════════════════════
# SCORING (PRM LABELS)
# ═══════════════════════════════════════════════════════════════════


def score_trajectory(api: OpenAIAPI, trajectory: dict) -> list[dict]:
    """Score each assistant step using OpenAI judge."""
    conversation_text = f"USER QUERY: {trajectory['user_prompt'][:500]}\n\n"
    assistant_indices = []

    for i, turn in enumerate(trajectory["turns"]):
        if turn["role"] == "assistant":
            assistant_indices.append(i)
            if turn.get("tool_calls"):
                tc = turn["tool_calls"][0]
                conversation_text += f"[Step {i}] ASSISTANT (tool call):\n"
                if turn.get("content"):
                    conversation_text += f"  Reasoning: {turn['content'][:200]}\n"
                conversation_text += (
                    f"  Tool: {tc['name']}({json.dumps(tc['arguments'])})\n\n"
                )
            else:
                conversation_text += f"[Step {i}] ASSISTANT (final answer):\n"
                conversation_text += f"  {turn.get('content', '')[:500]}\n\n"
        elif turn["role"] == "tool":
            content = turn.get("content", "")[:300]
            conversation_text += f"[Step {i}] TOOL RESULT ({turn.get('name', '?')}):\n"
            conversation_text += f"  {content}...\n\n"

    prompt = f"""Score each ASSISTANT step in this trajectory.

{conversation_text}

Score assistant steps at indices: {assistant_indices}
Return JSON: {{"scores": [{{"step_index": N, "score": 0.X, "reasoning": "..."}}]}}"""

    response = api.chat_completions(
        messages=[
            {"role": "system", "content": SCORING_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model="gpt-5-mini",
        temperature=0.2,
        max_completion_tokens=2048,
    )

    if "error" in response:
        return []

    try:
        content = response["choices"][0]["message"]["content"]
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text).get("scores", [])
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# EXPORT FORMATS
# ═══════════════════════════════════════════════════════════════════


def to_sft_messages(trajectory: dict) -> dict:
    """Convert trajectory to SFT chat format."""
    messages = [
        {"role": "system", "content": trajectory["system_prompt"]},
        {"role": "user", "content": trajectory["user_prompt"]},
    ]

    for turn in trajectory["turns"]:
        if turn["role"] == "assistant":
            msg = {"role": "assistant", "content": turn.get("content") or ""}
            if turn.get("tool_calls"):
                msg["tool_calls"] = []
                for tc in turn["tool_calls"]:
                    args = tc["arguments"]
                    msg["tool_calls"].append(
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(args)
                                if isinstance(args, dict)
                                else args,
                            },
                        }
                    )
            messages.append(msg)
        elif turn["role"] == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn["tool_call_id"],
                    "content": turn["content"],
                }
            )

    return {"messages": messages, "tools": trajectory["tools"]}


def to_prm_examples(trajectory: dict, scores: list[dict]) -> list[dict]:
    """Convert trajectory + scores to PRM step-level examples."""
    score_map = {s["step_index"]: s for s in scores}
    examples = []

    prefix = [
        {"role": "system", "content": trajectory["system_prompt"]},
        {"role": "user", "content": trajectory["user_prompt"]},
    ]

    for i, turn in enumerate(trajectory["turns"]):
        if turn["role"] == "assistant" and i in score_map:
            step_msg = {"role": "assistant", "content": turn.get("content", "")}
            if turn.get("tool_calls"):
                step_msg["tool_calls"] = turn["tool_calls"]

            examples.append(
                {
                    "prefix": list(prefix),
                    "step": step_msg,
                    "score": score_map[i]["score"],
                    "reasoning": score_map[i].get("reasoning", ""),
                    "is_final_answer": turn.get("tool_calls") is None,
                }
            )

        # Grow prefix
        if turn["role"] == "assistant":
            msg = {"role": "assistant", "content": turn.get("content", "")}
            if turn.get("tool_calls"):
                msg["tool_calls"] = turn["tool_calls"]
            prefix.append(msg)
        elif turn["role"] == "tool":
            prefix.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.get("tool_call_id", ""),
                    "content": turn["content"],
                }
            )

    return examples


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE: ONE SAMPLE
# ═══════════════════════════════════════════════════════════════════


def process_sample(
    sample: dict,
    sample_id: str,
    mistral: MistralClient,
    brave: BraveSearch,
    browser: WebBrowser,
    scorer: OpenAIAPI,
) -> Optional[dict]:
    """Process one sample end-to-end: generate → score → format."""

    log.info(f"\n{'=' * 60}")
    log.info(f"  Processing {sample_id}")
    log.info(f"  Query: {sample['user_prompt'][:80]}...")
    log.info(f"{'=' * 60}")

    # Generate trajectory with real tools
    trajectory = run_real_agentic_loop(
        mistral=mistral, brave=brave, browser=browser, sample=sample, max_turns=8
    )

    # Validate
    has_tool = any(
        t["role"] == "assistant" and t.get("tool_calls") for t in trajectory["turns"]
    )
    has_final = any(
        t["role"] == "assistant" and not t.get("tool_calls")
        for t in trajectory["turns"]
    )
    if not has_tool or not has_final:
        log.warning(f"  ✗ Invalid trajectory for {sample_id}")
        return None

    # Score
    log.info("  ⭐ Scoring steps...")
    scores = score_trajectory(scorer, trajectory)
    for s in scores:
        log.info(
            f"    Step {s['step_index']}: {s['score']:.2f} — {s.get('reasoning', '')[:60]}"
        )

    # Format
    sft = to_sft_messages(trajectory)
    prm = to_prm_examples(trajectory, scores)

    score_values = [s["score"] for s in scores]
    overall = sum(score_values) / len(score_values) if score_values else 0.5

    return {
        "id": sample_id,
        "trajectory": trajectory,
        "sft": sft,
        "prm_examples": prm,
        "scores": scores,
        "overall_score": round(overall, 3),
        "metadata": trajectory["metadata"],
    }


# ═══════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════


def load_dataset_samples(num: int, seed: int = 42) -> list[dict]:
    """Load samples from Nemotron dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset("nvidia/Nemotron-RL-knowledge-web_search-mcqa", split="train")
        log.info(f"  Loaded {len(ds)} rows from HuggingFace")
    except Exception as e:
        log.error(f"  Failed to load from HF: {e}")
        log.info("  Trying local fallback...")
        local = Path("./nemotron_samples.jsonl")
        if local.exists():
            rows = [json.loads(l) for l in open(local)]
            return rows[:num]
        raise

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    samples = []
    for idx in indices[:num]:
        row = ds[idx]
        params = row["responses_create_params"]
        samples.append(
            {
                "id": hashlib.md5(f"{idx}".encode()).hexdigest()[:12],
                "system_prompt": params["instructions"],
                "user_prompt": params["input"],
                "tools": params["tools"],
                "expected_answer": row["expected_answer"],
                "difficulty": row.get("task_difficulty_qwen3_32b_avg_8", 0.5),
            }
        )

    return samples


# ═══════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════


def run_batch(
    samples: list[dict],
    output_dir: str = "./synthetic_data",
    max_workers: int = 5,  # lower than simulated — we're hitting real APIs
    checkpoint_every: int = 10,
):
    """Process all samples with checkpointing."""
    os.makedirs(output_dir, exist_ok=True)

    # Resume
    checkpoint = Path(output_dir) / "checkpoint.jsonl"
    completed = set()
    results = []
    if checkpoint.exists():
        for line in open(checkpoint):
            r = json.loads(line)
            completed.add(r["id"])
            results.append(r)
        log.info(f"  Resumed: {len(results)} completed")

    remaining = [s for s in samples if s["id"] not in completed]
    if not remaining:
        log.info("  All done!")
        return results

    log.info(f"  Processing {len(remaining)} samples ({max_workers} workers)")

    # Rate limiting: Brave free tier = 1 req/sec, paid = 15/sec
    # We process sequentially with small delays to be safe
    # For parallel: each worker gets its own clients

    def worker(sample):
        mistral = MistralClient(os.environ["MISTRAL_API_KEY"])
        brave = BraveSearch(os.environ["BRAVE_API_KEY"])
        browser = WebBrowser()
        scorer = OpenAIAPI(os.environ["open_ai_api_key"])

        try:
            return process_sample(
                sample=sample,
                sample_id=sample["id"],
                mistral=mistral,
                brave=brave,
                browser=browser,
                scorer=scorer,
            )
        except Exception as e:
            log.error(f"  Exception for {sample['id']}: {e}")
            return None
        finally:
            mistral.close()
            brave.close()
            browser.close()
            scorer.close()

    new_results = []
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, s): s for s in remaining}

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    new_results.append(result)
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                log.error(f"  Worker exception: {e}")
                failed += 1

            done = i + 1
            if done % 5 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                log.info(
                    f"  {done}/{len(remaining)} "
                    f"({len(new_results)} ok, {failed} failed) "
                    f"[{rate:.2f}/s]"
                )

            if done % checkpoint_every == 0:
                with open(checkpoint, "w") as f:
                    for r in results:
                        f.write(json.dumps(r, default=str) + "\n")

    # Final save
    with open(checkpoint, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    return results


def export(results: list[dict], output_dir: str):
    """Export to SFT + PRM formats."""
    os.makedirs(output_dir, exist_ok=True)

    # SFT
    sft_path = Path(output_dir) / "sft_multiturn_real.jsonl"
    with open(sft_path, "w") as f:
        for r in results:
            f.write(json.dumps(r["sft"]) + "\n")
    log.info(f"  SFT: {sft_path} ({len(results)} examples)")

    # SFT high-quality
    hq = [r for r in results if r["overall_score"] >= 0.8]
    hq_path = Path(output_dir) / "sft_multiturn_real_hq.jsonl"
    with open(hq_path, "w") as f:
        for r in hq:
            f.write(json.dumps(r["sft"]) + "\n")
    log.info(f"  SFT HQ: {hq_path} ({len(hq)} examples)")

    # PRM
    prm_path = Path(output_dir) / "prm_step_scores_real.jsonl"
    prm_count = 0
    with open(prm_path, "w") as f:
        for r in results:
            for ex in r["prm_examples"]:
                f.write(json.dumps(ex) + "\n")
                prm_count += 1
    log.info(f"  PRM: {prm_path} ({prm_count} step examples)")

    # Stats
    scores = [r["overall_score"] for r in results]
    log.info(
        f"\n  📊 {len(results)} trajectories, avg score {sum(scores) / len(scores):.3f}"
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output_dir", default="./synthetic_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Verify env vars
    for var in ["MISTRAL_API_KEY", "BRAVE_API_KEY", "open_ai_api_key"]:
        if not os.environ.get(var):
            log.error(f"Missing env var: {var}")
            sys.exit(1)

    log.info("═══════════════════════════════════════════")
    log.info("  Real Trajectory Generator")
    log.info("═══════════════════════════════════════════")
    log.info("  Mistral Large → agent decisions")
    log.info("  Brave Search  → real search results")
    log.info("  Web scraping  → real page content")
    log.info("  OpenAI        → PRM scoring only")
    log.info(f"  Target: {args.num} trajectories")

    # Load
    samples = load_dataset_samples(args.num, args.seed)
    log.info(f"  Loaded {len(samples)} samples")

    # Run
    results = run_batch(samples, output_dir=args.output_dir, max_workers=args.workers)

    # Export
    export(results, args.output_dir)
    log.info(f"\n  ✓ Done! Files in {args.output_dir}/")


if __name__ == "__main__":
    main()
