#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Synthetic Multi-Turn Agentic Trajectory Generator

  Uses OpenAI API to generate 2000+ multi-turn browser agent
  trajectories with per-step PRM scores for:
    1. SFT training (multi-turn tool-calling)
    2. PRM training (step-level reward model)

  Pipeline:
    Phase 1: Generate diverse user queries
    Phase 2: Generate full trajectories (agent + simulated tool results)
    Phase 3: Score each step with a judge model
    Phase 4: Export to training formats
═══════════════════════════════════════════════════════════════════
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai --break-system-packages")
    exit(1)

# ── Tool Definitions (matching Chrome extension v2) ──────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search Google for information. Returns top search results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_page",
            "description": "Navigate to a URL and extract the page content including title, metadata, heading structure, and main text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to navigate to",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_content",
            "description": "Extract specific content from the current page using a CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector (e.g. 'h1', '.article-body', '#main-content')",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click an element on the current page identified by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to click",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_input",
            "description": "Type text into an input field or textarea on the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of the input field",
                    },
                    "value": {
                        "type": "string",
                        "description": "Text to type into the field",
                    },
                },
                "required": ["selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_links",
            "description": "List all links on the current page with their text and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max": {
                        "type": "integer",
                        "description": "Maximum links to return (default 20)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_interactive",
            "description": "List all interactive elements (buttons, inputs, links) with CSS selectors. Call before click_element or fill_input.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tables",
            "description": "Extract all HTML tables from the current page as structured data.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_page",
            "description": "Scroll the page up or down to see more content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Pixels to scroll (default 600)",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_text",
            "description": "Search for specific text on the current page. Returns matching passages with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_info",
            "description": "Get current page URL, title, scroll position, and page height.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]

# ── Query Templates ──────────────────────────────────────────────

QUERY_CATEGORIES = {
    "search_and_summarize": [
        "Search for the latest news about {topic} and give me a summary",
        "What are the most recent developments in {topic}?",
        "Find me information about {topic} and summarize the key points",
        "Research {topic} and tell me what you find",
        "Look up {topic} and give me an overview",
    ],
    "multi_hop_research": [
        "Search for {topic}, open the most relevant result, and extract the main points",
        "Find an article about {topic}, read it, and tell me the author's key arguments",
        "Search for {topic}, browse the top result, and find any statistics or data mentioned",
        "Look up {topic}, go to the best source, and summarize the methodology they describe",
        "Find a recent paper or article on {topic} and break down its conclusions",
    ],
    "specific_site_browse": [
        "Go to {url} and tell me about the top stories",
        "Browse {url} and summarize what you find",
        "Navigate to {url} and extract the main content",
        "Visit {url} and list the key links on the page",
        "Go to {url} and read the latest post or article",
    ],
    "data_extraction": [
        "Find a comparison table for {topic} and extract the data",
        "Search for {topic} pricing and find the pricing table",
        "Look up {topic} and extract any tables or structured data",
        "Find rankings or statistics about {topic}",
        "Search for {topic} benchmarks and extract the results",
    ],
    "form_interaction": [
        "Go to {url} and search for {topic} using their search box",
        "Navigate to {url}, find the search input, and look up {topic}",
        "Visit {url} and fill in the search field with {topic}",
    ],
    "multi_step_navigation": [
        "Go to {url}, find the link to {section}, and summarize that page",
        "Browse {url}, click on {section}, and tell me what it says",
        "Navigate to {url}, find and click the {section} link, then extract the content",
    ],
    "error_recovery": [
        "Search for {topic} on {url} — if the page doesn't load properly, try a Google search instead",
        "Try to browse {url} for {topic}. If you can't find it, search Google as a fallback",
        "Go to {url} and find {topic}. If the page structure is unexpected, use list_interactive to figure it out",
    ],
    "comparison_research": [
        "Compare {topic_a} vs {topic_b} — search for both and summarize the differences",
        "Find reviews of {topic_a} and {topic_b} and tell me which is better rated",
        "Research the pros and cons of {topic_a} versus {topic_b}",
    ],
}

TOPICS = [
    "artificial intelligence",
    "machine learning",
    "large language models",
    "transformer architecture",
    "CUDA programming",
    "GPU optimization",
    "Python web frameworks",
    "React vs Vue",
    "Rust programming",
    "quantum computing",
    "electric vehicles",
    "renewable energy",
    "cryptocurrency regulations",
    "space exploration",
    "climate change",
    "remote work trends",
    "cybersecurity threats",
    "5G technology",
    "gene editing CRISPR",
    "autonomous vehicles",
    "cloud computing pricing",
    "open source LLMs",
    "fine-tuning language models",
    "RAG systems",
    "vector databases",
    "MLOps best practices",
    "model quantization",
    "attention mechanisms",
    "reinforcement learning from human feedback",
    "neural network pruning",
    "federated learning",
    "edge AI deployment",
    "robotics advances",
    "natural language processing",
    "computer vision",
    "speech recognition",
    "recommendation systems",
    "time series forecasting",
    "graph neural networks",
    "diffusion models",
    "multimodal AI",
    "AI safety research",
    "AI alignment",
    "synthetic data generation",
    "knowledge distillation",
    "mixture of experts",
    "flash attention",
    "LoRA fine-tuning",
    "GGUF quantization",
    "vLLM serving",
    "Triton kernel programming",
    "DeepSpeed training",
    "PyTorch 2.0 compile",
]

URLS = [
    "news.ycombinator.com",
    "github.com/trending",
    "arxiv.org",
    "huggingface.co/models",
    "pytorch.org/blog",
    "openai.com/blog",
    "anthropic.com/research",
    "mistral.ai/news",
    "nvidia.com/blog",
    "techcrunch.com",
    "arstechnica.com",
    "theverge.com",
    "reddit.com/r/MachineLearning",
    "reddit.com/r/LocalLLaMA",
    "paperswithcode.com",
    "kaggle.com/competitions",
    "developer.nvidia.com/blog",
    "blog.google/technology/ai",
]

SECTIONS = [
    "about page",
    "pricing section",
    "documentation",
    "blog",
    "latest release",
    "getting started guide",
    "API reference",
    "FAQ section",
    "changelog",
    "community page",
]


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class Step:
    """A single step in a trajectory — either assistant text, tool call, or tool result."""

    role: str  # "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[list] = None  # for assistant turns with tool use
    tool_call_id: Optional[str] = None  # for tool result turns
    name: Optional[str] = None  # tool name (for tool results)
    # PRM fields
    score: Optional[float] = None  # 0.0 - 1.0 step quality
    score_reasoning: Optional[str] = None


@dataclass
class Trajectory:
    """A complete multi-turn trajectory with user query and agent steps."""

    id: str
    query: str
    category: str
    system_prompt: str
    steps: list = field(default_factory=list)
    # Overall quality
    overall_score: Optional[float] = None
    success: bool = False
    num_tool_calls: int = 0
    tools_used: list = field(default_factory=list)


# ── System Prompt ────────────────────────────────────────────────

from prompts import AGENT_SYSTEM_PROMPT, SCORING_SYSTEM, TRAJECTORY_GEN_SYSTEM

# ── Phase 1: Query Generation ───────────────────────────────────


def generate_queries(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate diverse queries from templates + topics."""
    rng = random.Random(seed)
    queries = []

    for i in range(n):
        cat = rng.choice(list(QUERY_CATEGORIES.keys()))
        template = rng.choice(QUERY_CATEGORIES[cat])

        topic = rng.choice(TOPICS)
        topic_a, topic_b = rng.sample(TOPICS, 2)
        url = rng.choice(URLS)
        section = rng.choice(SECTIONS)

        query = template.format(
            topic=topic, topic_a=topic_a, topic_b=topic_b, url=url, section=section
        )

        qid = hashlib.md5(f"{i}_{query}".encode()).hexdigest()[:12]
        queries.append({"id": qid, "query": query, "category": cat})

    return queries


# ── Phase 2: Trajectory Generation ──────────────────────────────

# TRAJECTORY_GEN_SYSTEM imported from prompts.py


def generate_single_trajectory(
    client: OpenAI, query_info: dict, model: str = "gpt-4o-mini"
) -> Optional[Trajectory]:
    """Generate one complete trajectory using OpenAI."""

    prompt = f"""Generate a realistic multi-turn browser agent trajectory for this query:

USER QUERY: "{query_info["query"]}"
CATEGORY: {query_info["category"]}

Generate the complete trajectory as a JSON array of steps. Include 2-6 tool calls with realistic results."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRAJECTORY_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,  # high diversity
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        # Parse — handle both {"steps": [...]} and bare [...]
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            steps_data = parsed.get("steps", parsed.get("trajectory", []))
        elif isinstance(parsed, list):
            steps_data = parsed
        else:
            return None

        if not steps_data or len(steps_data) < 3:
            return None

        # Build trajectory
        traj = Trajectory(
            id=query_info["id"],
            query=query_info["query"],
            category=query_info["category"],
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        tools_used = set()
        num_tool_calls = 0

        for s in steps_data:
            role = s.get("role", "assistant")

            if role == "assistant":
                tool_calls = s.get("tool_calls")
                if tool_calls:
                    # Ensure each has an id
                    for tc in tool_calls:
                        if "id" not in tc:
                            tc["id"] = (
                                f"call_{hashlib.md5(str(tc).encode()).hexdigest()[:8]}"
                            )
                        name = tc.get("name", tc.get("function", {}).get("name", ""))
                        if not name:
                            continue
                        tools_used.add(name)
                        num_tool_calls += 1

                traj.steps.append(
                    Step(
                        role="assistant",
                        content=s.get("content"),
                        tool_calls=tool_calls,
                    )
                )
            elif role == "tool":
                traj.steps.append(
                    Step(
                        role="tool",
                        content=s.get("content", ""),
                        tool_call_id=s.get("tool_call_id", ""),
                        name=s.get("name", ""),
                    )
                )

        traj.num_tool_calls = num_tool_calls
        traj.tools_used = list(tools_used)

        # Validate: must have at least 1 tool call and a final answer
        has_tool = any(s.tool_calls for s in traj.steps if s.role == "assistant")
        has_final = any(
            s.role == "assistant" and s.content and not s.tool_calls for s in traj.steps
        )

        if not has_tool or not has_final:
            return None

        return traj

    except Exception as e:
        print(f"  ✗ Error generating {query_info['id']}: {e}")
        return None


# ── Phase 3: Per-Step PRM Scoring ───────────────────────────────

# SCORING_SYSTEM imported from prompts.py


def score_trajectory(
    client: OpenAI, traj: Trajectory, model: str = "gpt-4o-mini"
) -> Trajectory:
    """Score each assistant step in a trajectory."""

    # Build the conversation for the judge to see
    conversation_text = f"USER QUERY: {traj.query}\n\n"
    assistant_indices = []

    for i, step in enumerate(traj.steps):
        if step.role == "assistant":
            assistant_indices.append(i)
            if step.tool_calls:
                tc_str = json.dumps(step.tool_calls, indent=2)
                conversation_text += f"[Step {i}] ASSISTANT (tool call):\n"
                if step.content:
                    conversation_text += f"  Reasoning: {step.content}\n"
                conversation_text += f"  Tool calls: {tc_str}\n\n"
            else:
                conversation_text += f"[Step {i}] ASSISTANT (final answer):\n"
                conversation_text += f"  {step.content}\n\n"
        elif step.role == "tool":
            # Truncate long tool results for the judge
            content = step.content or ""
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            conversation_text += f"[Step {i}] TOOL RESULT ({step.name}):\n"
            conversation_text += f"  {content}\n\n"

    prompt = f"""Score each ASSISTANT step in this browser agent trajectory.

{conversation_text}

Score each assistant step (indices: {assistant_indices}).
Output a JSON object with a "scores" array."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # low temperature for consistent scoring
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        result = json.loads(raw)
        scores = result.get("scores", [])

        # Apply scores to steps
        score_map = {s["step_index"]: s for s in scores}
        total_score = 0
        num_scored = 0

        for i, step in enumerate(traj.steps):
            if i in score_map:
                step.score = score_map[i].get("score", 0.5)
                step.score_reasoning = score_map[i].get("reasoning", "")
                total_score += step.score
                num_scored += 1

        if num_scored > 0:
            traj.overall_score = round(total_score / num_scored, 3)
            traj.success = traj.overall_score >= 0.6

    except Exception as e:
        print(f"  ✗ Error scoring {traj.id}: {e}")
        # Default scores
        for step in traj.steps:
            if step.role == "assistant":
                step.score = 0.5
                step.score_reasoning = "scoring failed"
        traj.overall_score = 0.5

    return traj


# ── Phase 4: Export to Training Formats ─────────────────────────


def trajectory_to_sft_messages(traj: Trajectory) -> list[dict]:
    """Convert trajectory to Mistral chat format for SFT training."""
    messages = [{"role": "system", "content": traj.system_prompt}]
    messages.append({"role": "user", "content": traj.query})

    for step in traj.steps:
        if step.role == "assistant":
            msg = {"role": "assistant"}
            if step.tool_calls:
                # Mistral format: content + tool_calls
                msg["content"] = step.content or ""
                msg["tool_calls"] = []
                for tc in step.tool_calls:
                    args = tc.get("arguments", tc.get("args", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    msg["tool_calls"].append(
                        {
                            "id": tc.get("id", f"call_{random.randint(1000, 9999)}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(args),
                            },
                        }
                    )
            else:
                msg["content"] = step.content or ""
            messages.append(msg)

        elif step.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step.tool_call_id or "",
                    "content": step.content or "",
                }
            )

    return messages


def trajectory_to_prm_examples(traj: Trajectory) -> list[dict]:
    """Convert trajectory to PRM training examples.

    Each example is: (prefix_messages, step_message, score)
    The PRM learns: given conversation so far, how good is this next step?
    """
    examples = []
    messages_so_far = [
        {"role": "system", "content": traj.system_prompt},
        {"role": "user", "content": traj.query},
    ]

    for step in traj.steps:
        if step.role == "assistant" and step.score is not None:
            # The step to score
            step_msg = {"role": "assistant"}
            if step.tool_calls:
                step_msg["content"] = step.content or ""
                step_msg["tool_calls"] = step.tool_calls
            else:
                step_msg["content"] = step.content or ""

            examples.append(
                {
                    "trajectory_id": traj.id,
                    "query": traj.query,
                    "prefix": list(messages_so_far),  # copy
                    "step": step_msg,
                    "score": step.score,
                    "reasoning": step.score_reasoning,
                    "is_final_answer": step.tool_calls is None,
                }
            )

        # Add step to prefix for next iteration
        if step.role == "assistant":
            msg = {"role": "assistant", "content": step.content or ""}
            if step.tool_calls:
                msg["tool_calls"] = step.tool_calls
            messages_so_far.append(msg)
        elif step.role == "tool":
            messages_so_far.append(
                {
                    "role": "tool",
                    "tool_call_id": step.tool_call_id or "",
                    "content": step.content or "",
                }
            )

    return examples


def export_datasets(
    trajectories: list[Trajectory], output_dir: str = "./synthetic_data"
):
    """Export trajectories to SFT and PRM training formats."""
    os.makedirs(output_dir, exist_ok=True)

    # ── SFT Dataset ──
    sft_examples = []
    for traj in trajectories:
        messages = trajectory_to_sft_messages(traj)
        sft_examples.append(
            {
                "id": traj.id,
                "messages": messages,
                "tools": TOOLS,
                "category": traj.category,
                "overall_score": traj.overall_score,
                "num_tool_calls": traj.num_tool_calls,
                "tools_used": traj.tools_used,
            }
        )

    sft_path = os.path.join(output_dir, "sft_multiturn.jsonl")
    with open(sft_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  ✓ SFT dataset: {sft_path} ({len(sft_examples)} examples)")

    # ── PRM Dataset ──
    prm_examples = []
    for traj in trajectories:
        prm_examples.extend(trajectory_to_prm_examples(traj))

    prm_path = os.path.join(output_dir, "prm_step_scores.jsonl")
    with open(prm_path, "w") as f:
        for ex in prm_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  ✓ PRM dataset: {prm_path} ({len(prm_examples)} step-level examples)")

    # ── High-quality SFT subset (score >= 0.8) ──
    hq_examples = [ex for ex in sft_examples if (ex["overall_score"] or 0) >= 0.8]
    hq_path = os.path.join(output_dir, "sft_multiturn_hq.jsonl")
    with open(hq_path, "w") as f:
        for ex in hq_examples:
            f.write(json.dumps(ex) + "\n")
    print(
        f"  ✓ High-quality SFT: {hq_path} ({len(hq_examples)} examples, score >= 0.8)"
    )

    # ── Full trajectories with scores (for analysis) ──
    full_path = os.path.join(output_dir, "trajectories_full.jsonl")
    with open(full_path, "w") as f:
        for traj in trajectories:
            obj = {
                "id": traj.id,
                "query": traj.query,
                "category": traj.category,
                "overall_score": traj.overall_score,
                "success": traj.success,
                "num_tool_calls": traj.num_tool_calls,
                "tools_used": traj.tools_used,
                "steps": [
                    {
                        "role": s.role,
                        "content": s.content,
                        "tool_calls": s.tool_calls,
                        "tool_call_id": s.tool_call_id,
                        "name": s.name,
                        "score": s.score,
                        "score_reasoning": s.score_reasoning,
                    }
                    for s in traj.steps
                ],
            }
            f.write(json.dumps(obj) + "\n")
    print(f"  ✓ Full trajectories: {full_path}")

    # ── Stats ──
    scores = [t.overall_score for t in trajectories if t.overall_score is not None]
    categories = {}
    tool_usage = {}
    for t in trajectories:
        categories[t.category] = categories.get(t.category, 0) + 1
        for tool in t.tools_used:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    stats = {
        "total_trajectories": len(trajectories),
        "total_sft_examples": len(sft_examples),
        "total_prm_examples": len(prm_examples),
        "high_quality_count": len(hq_examples),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "score_distribution": {
            "excellent (0.8-1.0)": sum(1 for s in scores if s >= 0.8),
            "good (0.6-0.8)": sum(1 for s in scores if 0.6 <= s < 0.8),
            "mediocre (0.4-0.6)": sum(1 for s in scores if 0.4 <= s < 0.6),
            "poor (0.0-0.4)": sum(1 for s in scores if s < 0.4),
        },
        "avg_tool_calls": round(
            sum(t.num_tool_calls for t in trajectories) / len(trajectories), 1
        ),
        "category_distribution": dict(sorted(categories.items(), key=lambda x: -x[1])),
        "tool_usage": dict(sorted(tool_usage.items(), key=lambda x: -x[1])),
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Stats: {stats_path}")
    print("\n  Summary:")
    print(f"    Trajectories: {stats['total_trajectories']}")
    print(f"    SFT examples: {stats['total_sft_examples']}")
    print(f"    PRM examples: {stats['total_prm_examples']}")
    print(f"    Avg score: {stats['avg_score']}")
    print(f"    Avg tool calls: {stats['avg_tool_calls']}")

    return stats


# ── Batch Generation with Concurrency ───────────────────────────


def generate_batch(
    client: OpenAI,
    queries: list[dict],
    gen_model: str = "gpt-4o-mini",
    score_model: str = "gpt-4o-mini",
    max_workers: int = 10,
    output_dir: str = "./synthetic_data",
    checkpoint_every: int = 50,
) -> list[Trajectory]:
    """Generate and score trajectories in parallel batches."""

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.jsonl")

    # Resume from checkpoint
    completed_ids = set()
    trajectories = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                obj = json.loads(line)
                completed_ids.add(obj["id"])
                # Rebuild trajectory from checkpoint
                traj = Trajectory(
                    id=obj["id"],
                    query=obj["query"],
                    category=obj["category"],
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    overall_score=obj.get("overall_score"),
                    success=obj.get("success", False),
                    num_tool_calls=obj.get("num_tool_calls", 0),
                    tools_used=obj.get("tools_used", []),
                )
                for s in obj.get("steps", []):
                    traj.steps.append(
                        Step(
                            **{
                                k: v
                                for k, v in s.items()
                                if k in Step.__dataclass_fields__
                            }
                        )
                    )
                trajectories.append(traj)
        print(f"  Resumed from checkpoint: {len(trajectories)} trajectories")

    remaining = [q for q in queries if q["id"] not in completed_ids]
    if not remaining:
        print("  All queries already processed!")
        return trajectories

    print(f"\n  Phase 2: Generating {len(remaining)} trajectories...")
    print(f"    Model: {gen_model}")
    print(f"    Workers: {max_workers}")

    generated = []
    failed = 0

    def gen_one(q):
        return generate_single_trajectory(client, q, model=gen_model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(gen_one, q): q for q in remaining}
        for i, future in enumerate(as_completed(futures)):
            q = futures[future]
            try:
                traj = future.result()
                if traj:
                    generated.append(traj)
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Exception for {q['id']}: {e}")
                failed += 1

            if (i + 1) % 10 == 0:
                print(f"    Generated: {len(generated)}/{i + 1} (failed: {failed})")

    print(f"  ✓ Generated {len(generated)} trajectories ({failed} failed)")

    # Phase 3: Score all generated trajectories
    print(f"\n  Phase 3: Scoring {len(generated)} trajectories...")
    print(f"    Model: {score_model}")

    scored = []

    def score_one(t):
        return score_trajectory(client, t, model=score_model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(score_one, t): t for t in generated}
        for i, future in enumerate(as_completed(futures)):
            try:
                traj = future.result()
                scored.append(traj)
            except Exception as e:
                print(f"  ✗ Scoring exception: {e}")

            if (i + 1) % 10 == 0:
                print(f"    Scored: {len(scored)}/{i + 1}")

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                with open(checkpoint_path, "a") as f:
                    for t in scored[-(checkpoint_every):]:
                        obj = {
                            "id": t.id,
                            "query": t.query,
                            "category": t.category,
                            "overall_score": t.overall_score,
                            "success": t.success,
                            "num_tool_calls": t.num_tool_calls,
                            "tools_used": t.tools_used,
                            "steps": [
                                {
                                    "role": s.role,
                                    "content": s.content,
                                    "tool_calls": s.tool_calls,
                                    "tool_call_id": s.tool_call_id,
                                    "name": s.name,
                                    "score": s.score,
                                    "score_reasoning": s.score_reasoning,
                                }
                                for s in t.steps
                            ],
                        }
                        f.write(json.dumps(obj) + "\n")
                print(f"    💾 Checkpoint saved ({len(scored)} scored)")

    print(f"  ✓ Scored {len(scored)} trajectories")
    trajectories.extend(scored)

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        for t in trajectories:
            obj = {
                "id": t.id,
                "query": t.query,
                "category": t.category,
                "overall_score": t.overall_score,
                "success": t.success,
                "num_tool_calls": t.num_tool_calls,
                "tools_used": t.tools_used,
                "steps": [
                    {
                        "role": s.role,
                        "content": s.content,
                        "tool_calls": s.tool_calls,
                        "tool_call_id": s.tool_call_id,
                        "name": s.name,
                        "score": s.score,
                        "score_reasoning": s.score_reasoning,
                    }
                    for s in t.steps
                ],
            }
            f.write(json.dumps(obj) + "\n")

    return trajectories


# ── Generate Example Trajectories (for preview) ────────────────


def generate_example_output() -> dict:
    """Return a hardcoded example showing what the output looks like."""
    return {
        "id": "abc123def456",
        "query": "Search for the latest news about open source LLMs and summarize the key points",
        "category": "search_and_summarize",
        "overall_score": 0.867,
        "success": True,
        "num_tool_calls": 3,
        "tools_used": ["web_search", "browse_page", "extract_content"],
        "steps": [
            {
                "role": "assistant",
                "content": "I'll search for the latest news about open source LLMs to find recent developments.",
                "tool_calls": [
                    {
                        "id": "call_8f3a2b1c",
                        "name": "web_search",
                        "arguments": {"query": "open source LLM news 2025"},
                    }
                ],
                "score": 0.9,
                "score_reasoning": "Good first step — web_search with a specific, time-bounded query. Including '2025' helps get recent results.",
            },
            {
                "role": "tool",
                "tool_call_id": "call_8f3a2b1c",
                "name": "web_search",
                "content": 'Search results for "open source LLM news 2025":\n\n1. Meta Releases Llama 4 Scout and Maverick Models\n   https://ai.meta.com/blog/llama-4-release\n   Meta has released Llama 4 Scout (17B active params, 109B total) and Llama 4 Maverick, featuring a new mixture-of-experts architecture...\n\n2. Mistral Launches Mistral Medium 3\n   https://mistral.ai/news/mistral-medium-3\n   Mistral AI announces Mistral Medium 3, their most capable open model yet, with 70B parameters and state-of-the-art reasoning...\n\n3. DeepSeek R2 Open-Sources Full Training Pipeline\n   https://www.deepseek.com/blog/r2-release\n   DeepSeek releases R2, a 236B MoE model with full training code, data recipes, and RLHF pipeline openly available...\n\n4. Hugging Face Open LLM Leaderboard February 2025 Update\n   https://huggingface.co/spaces/open-llm-leaderboard\n   The latest rankings show dramatic improvements in open models...',
                "score": None,
                "score_reasoning": None,
            },
            {
                "role": "assistant",
                "content": "Great results. Let me read the Meta Llama 4 article for more details since it seems like the biggest release.",
                "tool_calls": [
                    {
                        "id": "call_9e4b3c2d",
                        "name": "browse_page",
                        "arguments": {
                            "url": "https://ai.meta.com/blog/llama-4-release"
                        },
                    }
                ],
                "score": 0.85,
                "score_reasoning": "Good choice to dig deeper into the top result. Could arguably have browsed multiple articles, but prioritizing the most significant one is efficient.",
            },
            {
                "role": "tool",
                "tool_call_id": "call_9e4b3c2d",
                "name": "browse_page",
                "content": "Page: Introducing Llama 4: The Next Generation of Open AI Models\nURL: https://ai.meta.com/blog/llama-4-release\nAuthor: Meta AI Team\nDate: 2025-02-15\nDescription: Today we're releasing Llama 4, our most capable collection of open models.\n\nPage Structure:\n# Introducing Llama 4\n## Model Architecture\n## Scout: Efficient MoE\n## Maverick: Frontier Performance\n## Open Release Details\n\nContent:\nToday we are releasing the Llama 4 family of models, our most advanced open foundation models. The collection includes two models:\n\nLlama 4 Scout uses a mixture-of-experts architecture with 17 billion active parameters out of 109 billion total. This means it runs efficiently on a single GPU while maintaining strong performance across coding, math, and multilingual tasks. Scout achieves 82.3 on MMLU and 71.2 on HumanEval.\n\nLlama 4 Maverick is our flagship model with 400 billion total parameters (92B active). It matches or exceeds GPT-4o on most benchmarks while being fully open-weight. Maverick uses an improved attention mechanism we call Grouped Mixture Attention (GMA).\n\nBoth models are released under the Llama 4 Community License, which permits commercial use. Weights are available on Hugging Face and via our official GitHub repository.\n\n[... content truncated at 6000 chars]",
                "score": None,
                "score_reasoning": None,
            },
            {
                "role": "assistant",
                "content": "Let me also check the Mistral article to cover another major release.",
                "tool_calls": [
                    {
                        "id": "call_af5c4d3e",
                        "name": "browse_page",
                        "arguments": {
                            "url": "https://mistral.ai/news/mistral-medium-3"
                        },
                    }
                ],
                "score": 0.8,
                "score_reasoning": "Good to get a second source for a more comprehensive summary. The choice of the Mistral article makes sense given it's the second most significant result.",
            },
            {
                "role": "tool",
                "tool_call_id": "call_af5c4d3e",
                "name": "browse_page",
                "content": "Page: Mistral Medium 3 — Our Most Capable Open Model\nURL: https://mistral.ai/news/mistral-medium-3\nAuthor: Mistral AI\nDate: 2025-02-10\n\nContent:\nWe're excited to announce Mistral Medium 3, a 70B parameter model that sets new standards for open-weight models in reasoning, code generation, and instruction following.\n\nKey highlights:\n- 70B parameters with GQA attention and 128K context window\n- Scores 85.1 on MMLU, 74.8 on HumanEval, 92.3 on GSM8K\n- Native function calling with <tool_calls> format\n- Released under Apache 2.0 license\n- Available on Hugging Face, le Chat, and via API\n\nMistral Medium 3 was trained using a combination of supervised fine-tuning and a novel multi-stage RLHF process that improves both helpfulness and safety...",
                "score": None,
                "score_reasoning": None,
            },
            {
                "role": "assistant",
                "content": 'Here\'s a summary of the latest developments in open source LLMs:\n\n**1. Meta Llama 4 (Feb 15, 2025)**\nMeta released two new models:\n- **Llama 4 Scout**: A mixture-of-experts model with 17B active / 109B total parameters. Runs on a single GPU while scoring 82.3 on MMLU and 71.2 on HumanEval.\n- **Llama 4 Maverick**: The flagship 400B model (92B active) that matches GPT-4o on most benchmarks. Features a new "Grouped Mixture Attention" mechanism. Both are released under the Llama 4 Community License for commercial use.\n\n**2. Mistral Medium 3 (Feb 10, 2025)**\nMistral AI released a 70B model with:\n- 128K context window with grouped-query attention\n- Strong benchmarks: 85.1 MMLU, 74.8 HumanEval, 92.3 GSM8K\n- Native function calling support\n- Apache 2.0 license (fully permissive)\n\n**3. DeepSeek R2**\nDeepSeek released R2, a 236B MoE model notable for open-sourcing their complete training pipeline including data recipes and RLHF code.\n\nThe trend is clear: open models are rapidly closing the gap with proprietary ones, with MoE architectures becoming the dominant approach for balancing performance and efficiency.',
                "tool_calls": None,
                "score": 0.9,
                "score_reasoning": "Excellent final answer — well-organized with clear sections, includes specific numbers/benchmarks, covers multiple sources, and adds a synthesis observation about trends. Could have included the HuggingFace leaderboard data for completeness.",
            },
        ],
        "sft_format_preview": "This trajectory converts to 8 messages: system + user + 3x(assistant_tool_call + tool_result) + assistant_final",
        "prm_format_preview": "This trajectory produces 3 PRM examples (one per scored assistant step)",
    }


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn agentic trajectories"
    )
    parser.add_argument(
        "--num", type=int, default=2000, help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--gen_model", default="gpt-4o-mini", help="Model for trajectory generation"
    )
    parser.add_argument(
        "--score_model", default="gpt-4o-mini", help="Model for PRM scoring"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Concurrent API workers"
    )
    parser.add_argument(
        "--output_dir", default="./synthetic_data", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--preview", action="store_true", help="Show example output and exit"
    )
    parser.add_argument(
        "--api_key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env)"
    )
    args = parser.parse_args()

    if args.preview:
        print("\n═══ EXAMPLE TRAJECTORY OUTPUT ═══\n")
        example = generate_example_output()
        print(json.dumps(example, indent=2))
        print("\n═══ SFT FORMAT (Mistral chat) ═══\n")
        # Convert example to SFT
        traj = Trajectory(
            id=example["id"],
            query=example["query"],
            category=example["category"],
            system_prompt=AGENT_SYSTEM_PROMPT,
        )
        for s in example["steps"]:
            traj.steps.append(
                Step(
                    role=s["role"],
                    content=s.get("content"),
                    tool_calls=s.get("tool_calls"),
                    tool_call_id=s.get("tool_call_id"),
                    name=s.get("name"),
                    score=s.get("score"),
                )
            )
        sft_msgs = trajectory_to_sft_messages(traj)
        print(
            json.dumps({"messages": sft_msgs, "tools": ["...11 tools..."]}, indent=2)[
                :3000
            ]
        )
        print("\n═══ PRM FORMAT (step-level scores) ═══\n")
        prm_exs = trajectory_to_prm_examples(traj)
        for ex in prm_exs[:2]:
            print(
                json.dumps(
                    {
                        "query": ex["query"],
                        "prefix_length": len(ex["prefix"]),
                        "step_preview": str(ex["step"])[:200],
                        "score": ex["score"],
                        "is_final_answer": ex["is_final_answer"],
                    },
                    indent=2,
                )
            )
            print()
        return

    # Setup client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable or pass --api_key")
        return

    client = OpenAI(api_key=api_key)

    print("═══════════════════════════════════════════════════")
    print("  Synthetic Trajectory Generator")
    print("═══════════════════════════════════════════════════")
    print(f"  Target: {args.num} trajectories")
    print(f"  Gen model: {args.gen_model}")
    print(f"  Score model: {args.score_model}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output_dir}")

    # Phase 1
    print(f"\n  Phase 1: Generating {args.num} queries...")
    queries = generate_queries(args.num, seed=args.seed)
    cats = {}
    for q in queries:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"  ✓ Generated {len(queries)} queries across {len(cats)} categories")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Phase 2+3
    trajectories = generate_batch(
        client,
        queries,
        gen_model=args.gen_model,
        score_model=args.score_model,
        max_workers=args.workers,
        output_dir=args.output_dir,
    )

    # Phase 4
    print("\n  Phase 4: Exporting datasets...")
    stats = export_datasets(trajectories, output_dir=args.output_dir)

    # Cost estimate
    # gpt-4o-mini: ~$0.15/1M input, $0.60/1M output
    # Rough: ~2k input + ~2k output tokens per trajectory generation
    # + ~1.5k input + ~0.5k output per scoring
    est_gen_cost = args.num * (2000 * 0.15 + 2000 * 0.60) / 1_000_000
    est_score_cost = args.num * (1500 * 0.15 + 500 * 0.60) / 1_000_000
    print(f"\n  Estimated cost: ~${est_gen_cost + est_score_cost:.2f}")
    print(f"    Generation: ~${est_gen_cost:.2f}")
    print(f"    Scoring: ~${est_score_cost:.2f}")
    print(f"\n  Done! Files in {args.output_dir}/")


if __name__ == "__main__":
    main()
