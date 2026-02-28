#!/usr/bin/env python3
"""
Stage 2 SFT: Multi-dataset blending for enhanced agentic tool-calling.

Runs on top of a Stage 1 checkpoint (or base model) and blends multiple
tool-calling datasets to teach:
  - Precise function-calling schema adherence (xLAM)
  - Multi-turn agentic trajectories (Nemotron-Agentic)
  - Conversational tool use (Glaive)
  - Irrelevance detection / knowing when NOT to call tools (xLAM-Irrelevance)
  - Multi-step workplace workflows (Nemotron workplace_assistant, from Stage 1)

Usage:
    python train_sft_v2.py \
        --base_model ./ministral-3b-agent-sft-merged \
        --epochs 1 \
        --lr 5e-5 \
        --wandb_project mistral-hackathon
"""

import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# ──────────────────────────────────────────────
# 1. FORMAT CONVERTERS
#    Each dataset has a different schema.
#    All converters output: {"messages": [...], "tools": [...]}
#    where messages is OpenAI-style and tools is a list of
#    function schemas.
# ──────────────────────────────────────────────


def strip_nulls(tool: dict) -> dict:
    """Strip null-valued parameters from tool schemas."""
    tool = copy.deepcopy(tool)
    fn = tool.get("function", tool)  # handle both wrapped and unwrapped
    if "parameters" in fn and "properties" in fn["parameters"]:
        props = fn["parameters"]["properties"]
        cleaned = {k: v for k, v in props.items() if v is not None}
        fn["parameters"]["properties"] = cleaned
        if "required" in fn["parameters"] and fn["parameters"]["required"]:
            fn["parameters"]["required"] = [
                r for r in fn["parameters"]["required"] if r in cleaned
            ]
    return tool


# ── 1a. Nemotron workplace_assistant (same as Stage 1) ──


def convert_nemotron_workplace(example: dict) -> dict:
    """
    nvidia/Nemotron-RL-agent-workplace_assistant
    Schema: responses_create_params.input, .tools, ground_truth
    """
    params = example["responses_create_params"]
    ground_truth = example["ground_truth"]
    input_messages = params["input"]
    raw_tools = params.get("tools", [])
    cleaned_tools = [strip_nulls(t) for t in raw_tools]

    tool_calls = []
    for i, gt in enumerate(ground_truth):
        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": gt["name"],
                    "arguments": gt["arguments"]
                    if isinstance(gt["arguments"], str)
                    else json.dumps(gt["arguments"]),
                },
            }
        )

    messages = list(input_messages)
    if tool_calls:
        messages.append({"role": "assistant", "tool_calls": tool_calls})
    else:
        messages.append({"role": "assistant", "content": "I'll help you with that."})

    return {
        "messages": messages,
        "tools": cleaned_tools,
        "source": "nemotron_workplace",
    }


# ── 1b. Salesforce xLAM function-calling 60k ──


def convert_xlam(example: dict) -> dict:
    """
    Salesforce/xlam-function-calling-60k
    Schema: query (str), tools (JSON str), answers (JSON str)
    """
    query = example["query"]

    # Parse tools
    tools_raw = example.get("tools", "[]")
    if isinstance(tools_raw, str):
        try:
            tools_list = json.loads(tools_raw)
        except json.JSONDecodeError:
            tools_list = []
    else:
        tools_list = tools_raw if tools_raw else []

    # Normalize tool format to OpenAI style
    tools = []
    for t in tools_list:
        if "function" in t:
            tools.append(strip_nulls(t))
        else:
            # xlam stores tools flat: {name, description, parameters}
            wrapped = {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
            tools.append(strip_nulls(wrapped))

    # Parse answers
    answers_raw = example.get("answers", "[]")
    if isinstance(answers_raw, str):
        try:
            answers = json.loads(answers_raw)
        except json.JSONDecodeError:
            answers = []
    else:
        answers = answers_raw if answers_raw else []

    # Build messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use them when needed to answer the user's question.",
        },
        {"role": "user", "content": query},
    ]

    if answers:
        tool_calls = []
        for i, ans in enumerate(answers):
            args = ans.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": ans.get("name", ""),
                        "arguments": json.dumps(args)
                        if isinstance(args, dict)
                        else str(args),
                    },
                }
            )
        messages.append({"role": "assistant", "tool_calls": tool_calls})
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "I can help with that, but none of the available tools are needed for this request.",
            }
        )

    return {"messages": messages, "tools": tools, "source": "xlam"}


# ── 1c. Glaive function-calling v2 ──


def convert_glaive(example: dict) -> dict:
    """
    glaiveai/glaive-function-calling-v2
    Schema: system (str with tool defs), chat (str with USER/ASSISTANT/FUNCTION turns)
    """
    system_content = example.get("system", "")
    chat_raw = example.get("chat", "")

    # Extract tools from system prompt
    tools = []
    tool_match = re.search(r"(\[.*\])", system_content, re.DOTALL)
    if tool_match:
        try:
            tools_raw = json.loads(tool_match.group(1))
            for t in tools_raw:
                if "function" in t:
                    tools.append(strip_nulls(t))
                elif "name" in t:
                    tools.append(
                        strip_nulls(
                            {
                                "type": "function",
                                "function": {
                                    "name": t.get("name", ""),
                                    "description": t.get("description", ""),
                                    "parameters": t.get("parameters", {}),
                                },
                            }
                        )
                    )
        except json.JSONDecodeError:
            pass

    # Parse chat turns
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use them when needed.",
        }
    ]

    # Split on role markers
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_raw)
    parts = [p.strip() for p in parts if p.strip()]

    i = 0
    while i < len(parts):
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        if marker == "USER:":
            messages.append({"role": "user", "content": content})
        elif marker == "ASSISTANT:":
            # Check for function call
            fc_match = re.search(r"<functioncall>\s*(\{.*?\})", content, re.DOTALL)
            if fc_match:
                try:
                    fc = json.loads(fc_match.group(1))
                    args = fc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": f"call_{len(messages)}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": json.dumps(args)
                                        if isinstance(args, dict)
                                        else str(args),
                                    },
                                }
                            ],
                        }
                    )
                except json.JSONDecodeError:
                    messages.append({"role": "assistant", "content": content})
            else:
                if content:
                    messages.append({"role": "assistant", "content": content})
        elif marker == "FUNCTION RESPONSE:":
            messages.append({"role": "tool", "content": content})

    # Must have at least system + user + assistant
    if len(messages) < 3:
        return None

    return {"messages": messages, "tools": tools, "source": "glaive"}


# ── 1d. xLAM Irrelevance (teaches when NOT to call tools) ──


def convert_xlam_irrelevance(example: dict) -> dict:
    """
    MadeAgents/XLAM-7.5k-Irrelevance
    Same schema as xLAM but tools are irrelevant to query.
    Model should learn to NOT make a tool call.
    """
    query = example.get("query", "")

    tools_raw = example.get("tools", "[]")
    if isinstance(tools_raw, str):
        try:
            tools_list = json.loads(tools_raw)
        except json.JSONDecodeError:
            tools_list = []
    else:
        tools_list = tools_raw if tools_raw else []

    tools = []
    for t in tools_list:
        if "function" in t:
            tools.append(strip_nulls(t))
        else:
            tools.append(
                strip_nulls(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    }
                )
            )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Only use them when they are relevant to the user's question. If none of the tools are applicable, respond directly without calling any tools.",
        },
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": "I can answer this directly without using any of the available tools, as none of them are relevant to your question.",
        },
    ]

    return {"messages": messages, "tools": tools, "source": "xlam_irrelevance"}


# ── 1e. Nemotron-Agentic-v1 ──


def convert_nemotron_agentic(example: dict) -> dict:
    """
    nvidia/Nemotron-Agentic-v1
    Multi-turn trajectories with tool use.
    Schema: messages (list of dicts), tools (list of tool schemas)
    """
    messages = example.get("messages", [])
    tools = example.get("tools", [])

    if not messages:
        return None

    # Clean tools
    cleaned_tools = [strip_nulls(t) for t in tools] if tools else []

    return {"messages": messages, "tools": cleaned_tools, "source": "nemotron_agentic"}


# ──────────────────────────────────────────────
# 2. DATASET LOADING & BLENDING
# ──────────────────────────────────────────────

DATASET_REGISTRY = {
    "nemotron_workplace": {
        "hf_name": "nvidia/Nemotron-RL-agent-workplace_assistant",
        "converter": convert_nemotron_workplace,
        "default_samples": None,  # use all
        "default_weight": 3.0,  # oversample: core domain data
        "split_train": "train",
        "split_eval": "validation",
        "description": "Multi-step workplace tool chaining (Stage 1 domain)",
    },
    "xlam": {
        "hf_name": "Salesforce/xlam-function-calling-60k",
        "converter": convert_xlam,
        "default_samples": 5000,
        "default_weight": 1.0,
        "split_train": "train",
        "split_eval": None,  # no eval split; we'll carve out 5%
        "description": "Precise function-calling with diverse APIs",
    },
    "glaive": {
        "hf_name": "glaiveai/glaive-function-calling-v2",
        "converter": convert_glaive,
        "default_samples": 3000,
        "default_weight": 1.0,
        "split_train": "train",
        "split_eval": None,
        "description": "Multi-turn conversational tool use",
    },
    "xlam_irrelevance": {
        "hf_name": "MadeAgents/XLAM-7.5k-Irrelevance",
        "converter": convert_xlam_irrelevance,
        "default_samples": 2000,
        "default_weight": 0.5,
        "split_train": "train",
        "split_eval": None,
        "description": "When NOT to call tools (irrelevance detection)",
    },
    "nemotron_agentic": {
        "hf_name": "nvidia/Nemotron-Agentic-v1",
        "converter": convert_nemotron_agentic,
        "default_samples": 3000,
        "default_weight": 1.5,
        "split_train": "train",
        "split_eval": None,
        "description": "Multi-turn agentic trajectories with tool results",
    },
}


def load_and_convert_dataset(name: str, max_samples=None, split="train"):
    """Load a dataset from HF and convert to unified format."""
    config = DATASET_REGISTRY[name]
    print(f"\n  Loading {name}: {config['hf_name']}...")

    try:
        ds = load_dataset(config["hf_name"], split=split)
    except Exception as e:
        print(f"  ⚠ Failed to load {name}: {e}")
        return []

    # Sample if needed
    n = max_samples or config["default_samples"]
    if n and len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
        print(f"  Sampled {n} from {len(ds)} total")

    # Convert
    converter = config["converter"]
    converted = []
    skipped = 0
    for example in ds:
        try:
            result = converter(example)
            if result is not None:
                converted.append(result)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    print(f"  ✓ Converted {len(converted)} examples ({skipped} skipped)")
    return converted


def blend_datasets(datasets_dict: dict[str, list], weights: dict[str, float], seed=42):
    """
    Blend multiple converted datasets according to weights.
    Weight > 1 means oversample, < 1 means undersample.
    Returns a single shuffled list.
    """
    rng = random.Random(seed)
    blended = []

    for name, examples in datasets_dict.items():
        weight = weights.get(name, 1.0)
        if weight <= 0:
            continue

        if weight >= 1.0:
            # Oversample: repeat full dataset + sample remainder
            full_copies = int(weight)
            remainder = weight - full_copies
            for _ in range(full_copies):
                blended.extend(examples)
            if remainder > 0:
                n_extra = int(len(examples) * remainder)
                blended.extend(rng.sample(examples, min(n_extra, len(examples))))
        else:
            # Undersample
            n = int(len(examples) * weight)
            blended.extend(rng.sample(examples, min(n, len(examples))))

    rng.shuffle(blended)
    print(f"\n  Total blended examples: {len(blended)}")

    # Print composition
    source_counts = Counter(ex["source"] for ex in blended)
    for source, count in source_counts.most_common():
        pct = 100 * count / len(blended)
        print(f"    {source}: {count} ({pct:.1f}%)")

    return blended


def examples_to_hf_dataset(examples: list) -> Dataset:
    """Convert list of dicts to HuggingFace Dataset."""
    return Dataset.from_list(examples)


# ──────────────────────────────────────────────
# 3. CHAT TEMPLATE APPLICATION
# ──────────────────────────────────────────────


def apply_chat_template(example, tokenizer):
    """Apply chat template to a converted example."""
    messages = example["messages"]
    tools = example.get("tools", [])

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: embed tools in system prompt
        tool_desc = json.dumps(tools, indent=2) if tools else ""
        adjusted = []
        for msg in messages:
            if msg["role"] == "system" and tool_desc:
                adjusted.append(
                    {
                        "role": "system",
                        "content": msg["content"]
                        + f"\n\nAvailable tools:\n{tool_desc}",
                    }
                )
            elif msg.get("tool_calls"):
                # Convert tool_calls to text for models without native support
                tc_text = json.dumps(
                    [
                        {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"],
                        }
                        for tc in msg["tool_calls"]
                    ]
                )
                adjusted.append(
                    {"role": "assistant", "content": f"[TOOL_CALLS] {tc_text}"}
                )
            elif msg["role"] == "tool":
                adjusted.append(
                    {"role": "user", "content": f"[TOOL_RESULT] {msg['content']}"}
                )
            else:
                adjusted.append(msg)
        try:
            text = tokenizer.apply_chat_template(
                adjusted, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return {"text": None}

    return {"text": text}


def detect_response_template(tokenizer) -> str:
    """Auto-detect the boundary token(s) between user and assistant."""
    test_msgs = [
        {"role": "user", "content": "BOUNDARY_MARKER"},
        {"role": "assistant", "content": "RESPONSE_MARKER"},
    ]
    text = tokenizer.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=False
    )
    marker_pos = text.find("BOUNDARY_MARKER")
    response_pos = text.find("RESPONSE_MARKER")

    if marker_pos == -1 or response_pos == -1:
        return "[/INST]"

    between = text[marker_pos + len("BOUNDARY_MARKER") : response_pos]
    stripped = between.strip()
    if len(stripped) < 2:
        return "[/INST]"
    return stripped


# ──────────────────────────────────────────────
# 4. EVAL HARNESS (imported from v1 concepts)
# ──────────────────────────────────────────────


def parse_tool_calls_from_generation(text: str) -> list[dict]:
    """Parse tool calls from model output. Handles Mistral + JSON formats."""
    tool_calls = []

    # Strategy 1: [TOOL_CALLS] marker
    for marker in ["[TOOL_CALLS]", "[TOOL_CALL]"]:
        if marker in text:
            json_part = text.split(marker, 1)[1].strip()
            for line in json_part.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    items = parsed if isinstance(parsed, list) else [parsed]
                    for tc in items:
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        tool_calls.append(
                            {"name": tc.get("name", ""), "arguments": args}
                        )
                    return tool_calls
                except json.JSONDecodeError:
                    continue

    # Strategy 2: JSON arrays
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[" and brace_depth == 0:
            start = i
            brace_depth = 1
        elif ch == "[":
            brace_depth += 1
        elif ch == "]":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if (
                        isinstance(parsed, list)
                        and parsed
                        and isinstance(parsed[0], dict)
                        and "name" in parsed[0]
                    ):
                        for item in parsed:
                            args = item.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    pass
                            tool_calls.append({"name": item["name"], "arguments": args})
                        return tool_calls
                except:
                    pass
                start = None

    # Strategy 3: Individual objects
    idx = 0
    while idx < len(text):
        pos = text.find('{"name":', idx)
        if pos == -1:
            break
        depth = 0
        for j in range(pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[pos : j + 1])
                        if "name" in parsed:
                            args = parsed.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    pass
                            tool_calls.append(
                                {"name": parsed["name"], "arguments": args}
                            )
                    except:
                        pass
                    idx = j + 1
                    break
        else:
            break

    return tool_calls


def compute_tool_call_metrics(predicted, ground_truth):
    """Compute per-example metrics."""
    metrics = {
        "tool_name_exact_match": 0,
        "tool_name_set_match": 0,
        "first_tool_name_correct": 0,
        "full_exact_match": 0,
        "tool_name_precision": 0.0,
        "tool_name_recall": 0.0,
        "tool_name_f1": 0.0,
        "argument_accuracy": 0.0,
        "n_predicted_calls": len(predicted),
        "n_ground_truth_calls": len(ground_truth),
        "parse_success": 1 if len(predicted) > 0 or len(ground_truth) == 0 else 0,
    }
    if not ground_truth:
        metrics["full_exact_match"] = 1 if not predicted else 0
        metrics["tool_name_exact_match"] = 1 if not predicted else 0
        return metrics

    pred_names = [tc["name"] for tc in predicted]
    gt_names = [tc["name"] for tc in ground_truth]
    metrics["tool_name_exact_match"] = 1 if pred_names == gt_names else 0
    metrics["tool_name_set_match"] = 1 if set(pred_names) == set(gt_names) else 0
    if pred_names and gt_names:
        metrics["first_tool_name_correct"] = 1 if pred_names[0] == gt_names[0] else 0

    common = sum((Counter(pred_names) & Counter(gt_names)).values())
    metrics["tool_name_precision"] = common / max(len(pred_names), 1)
    metrics["tool_name_recall"] = common / max(len(gt_names), 1)
    p, r = metrics["tool_name_precision"], metrics["tool_name_recall"]
    metrics["tool_name_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    total_args, correct_args, full_match = 0, 0, True
    for i, gt_call in enumerate(ground_truth):
        gt_args = gt_call.get("arguments", {})
        if isinstance(gt_args, str):
            try:
                gt_args = json.loads(gt_args)
            except:
                gt_args = {}
        if i >= len(predicted):
            full_match = False
            total_args += len(gt_args)
            continue
        pred_call = predicted[i]
        if pred_call["name"] != gt_call["name"]:
            full_match = False
        pred_args = pred_call.get("arguments", {})
        if isinstance(pred_args, str):
            try:
                pred_args = json.loads(pred_args)
            except:
                pred_args = {}
        for key, value in gt_args.items():
            total_args += 1
            if key in pred_args and str(pred_args[key]) == str(value):
                correct_args += 1
            else:
                full_match = False
        for key in pred_args:
            if key not in gt_args:
                full_match = False

    metrics["argument_accuracy"] = correct_args / max(total_args, 1)
    metrics["full_exact_match"] = (
        1 if full_match and len(predicted) == len(ground_truth) else 0
    )
    return metrics


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    raw_dataset_split,
    max_examples=None,
    max_new_tokens=512,
    temperature=0.01,
    desc="eval",
):
    """Run generation-based eval on the Nemotron workplace validation set."""
    model.eval()
    examples = list(raw_dataset_split)
    if max_examples:
        examples = examples[:max_examples]

    all_metrics = defaultdict(list)
    per_example_results = []
    print(f"\n  Running {desc} on {len(examples)} examples...")

    for i, example in enumerate(examples):
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(examples)}...")

        params = example["responses_create_params"]
        ground_truth = example["ground_truth"]
        input_messages = params["input"]
        raw_tools = params.get("tools", [])
        cleaned_tools = [strip_nulls(t) for t in raw_tools]

        try:
            input_text = tokenizer.apply_chat_template(
                input_messages,
                tools=cleaned_tools if cleaned_tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except:
            tool_desc = json.dumps(cleaned_tools, indent=2)
            adjusted = [
                {
                    "role": m["role"],
                    "content": m["content"]
                    + (
                        f"\n\nAvailable tools:\n{tool_desc}"
                        if m["role"] == "system"
                        else ""
                    ),
                }
                if m["role"] == "system"
                else m
                for m in input_messages
            ]
            input_text = tokenizer.apply_chat_template(
                adjusted, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
        )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )

        predicted = parse_tool_calls_from_generation(generated)
        gt_calls = []
        for gt in ground_truth:
            args = gt["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            gt_calls.append({"name": gt["name"], "arguments": args})

        example_metrics = compute_tool_call_metrics(predicted, gt_calls)
        for k, v in example_metrics.items():
            all_metrics[k].append(v)

        user_msg = next(
            (m["content"][:150] for m in input_messages if m["role"] == "user"), ""
        )
        per_example_results.append(
            {
                "idx": i,
                "user_query": user_msg,
                "gt_tools": ", ".join(tc["name"] for tc in gt_calls),
                "pred_tools": ", ".join(tc["name"] for tc in predicted),
                "name_match": example_metrics["tool_name_exact_match"],
                "full_match": example_metrics["full_exact_match"],
                "arg_accuracy": example_metrics["argument_accuracy"],
                "generated_text": generated[:300],
            }
        )

    agg = {f"{desc}/{k}": np.mean(v) for k, v in all_metrics.items()}
    agg[f"{desc}/n_examples"] = len(examples)

    print(f"\n  ── {desc} Results ({len(examples)} examples) ──")
    for m in [
        "tool_name_exact_match",
        "tool_name_set_match",
        "first_tool_name_correct",
        "tool_name_f1",
        "argument_accuracy",
        "full_exact_match",
        "parse_success",
    ]:
        print(f"    {m:<30s}  {agg[f'{desc}/{m}']:.1%}")

    return agg, per_example_results


def log_eval_to_wandb(metrics, per_example, phase):
    """Log eval results to W&B."""
    import wandb

    wandb.log(metrics)
    table = wandb.Table(
        columns=[
            "idx",
            "user_query",
            "gt_tools",
            "pred_tools",
            "name_match",
            "full_match",
            "arg_accuracy",
            "generated_text",
        ]
    )
    for r in per_example:
        table.add_data(
            r["idx"],
            r["user_query"],
            r["gt_tools"],
            r["pred_tools"],
            r["name_match"],
            r["full_match"],
            r["arg_accuracy"],
            r["generated_text"],
        )
    wandb.log({f"{phase}/per_example_results": table})


# ──────────────────────────────────────────────
# 5. W&B CALLBACK (slimmed from v1)
# ──────────────────────────────────────────────


class WandbCallback(TrainerCallback):
    """Lightweight W&B callback for Stage 2."""

    def __init__(self):
        self.training_start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        import wandb

        step = state.global_step
        if step <= 1:
            return

        logs = {}
        elapsed = time.time() - self.training_start_time
        logs["perf/elapsed_min"] = elapsed / 60
        logs["perf/steps_per_sec"] = step / max(elapsed, 1)

        if torch.cuda.is_available():
            logs["gpu/vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            logs["gpu/vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        wandb.log(logs, step=step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb

        if logs is None:
            return
        enhanced = {}
        if "loss" in logs:
            enhanced["train/perplexity"] = math.exp(min(logs["loss"], 20))
        if "learning_rate" in logs:
            enhanced["train/lr_log10"] = math.log10(max(logs["learning_rate"], 1e-10))
        if enhanced:
            wandb.log(enhanced, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import wandb

        if metrics:
            eval_logs = {k.replace("eval_", "eval/"): v for k, v in metrics.items()}
            if "eval_loss" in metrics:
                eval_logs["eval/perplexity"] = math.exp(min(metrics["eval_loss"], 20))
            wandb.log(eval_logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        import wandb

        total = time.time() - self.training_start_time
        wandb.log({"summary/total_training_minutes": total / 60})


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Stage 2 SFT: Multi-dataset blending")

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ministral-3b-agent-sft-merged",
        help="Stage 1 merged model or base model path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Ministral-3-3B-Instruct",
        help="Tokenizer source (if different from base_model)",
    )
    parser.add_argument("--output_dir", type=str, default="./ministral-3b-agent-sft-v2")

    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "nemotron_workplace",
            "xlam",
            "glaive",
            "xlam_irrelevance",
            "nemotron_agentic",
        ],
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to include in blend",
    )
    parser.add_argument(
        "--samples",
        type=str,
        nargs="*",
        default=None,
        help="Per-dataset sample counts (same order as --datasets). E.g. --samples 0 5000 3000 2000 3000  (0 = use all)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="*",
        default=None,
        help="Per-dataset blend weights (same order as --datasets). E.g. --weights 3.0 1.0 1.0 0.5 1.5",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=1, help="Epochs (1 is usually enough for stage 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate (lower than stage 1)"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (lower than stage 1 to avoid overwriting)",
    )
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--full_finetune", action="store_true", default=False)

    # Eval
    parser.add_argument("--eval_max_examples", type=int, default=100)
    parser.add_argument("--skip_eval", action="store_true", default=False)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="mistral-hackathon")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    # Hub
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--hub_private", action="store_true", default=True)

    args = parser.parse_args()

    print("=" * 60)
    print("  STAGE 2 SFT: Multi-Dataset Blending")
    print("=" * 60)
    print(f"  Base model:  {args.base_model}")
    print(f"  Datasets:    {args.datasets}")
    print(f"  LR:          {args.lr} (lower for stage 2)")
    print(f"  LoRA rank:   {args.lora_r}")
    print(f"  Epochs:      {args.epochs}")
    print()

    # ── Load tokenizer ──
    print("[1/6] Loading tokenizer...")
    tokenizer_source = args.base_model
    # If base_model doesn't have tokenizer files, fall back to model_name
    if not os.path.exists(
        os.path.join(tokenizer_source, "tokenizer.json")
    ) and not os.path.exists(os.path.join(tokenizer_source, "tokenizer_config.json")):
        tokenizer_source = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load & convert datasets ──
    print("\n[2/6] Loading and converting datasets...")

    # Parse per-dataset samples and weights
    sample_overrides = {}
    weight_overrides = {}
    if args.samples:
        for name, s in zip(args.datasets, args.samples):
            val = int(s)
            sample_overrides[name] = val if val > 0 else None
    if args.weights:
        for name, w in zip(args.datasets, args.weights):
            weight_overrides[name] = float(w)

    all_converted = {}
    for ds_name in args.datasets:
        max_s = sample_overrides.get(
            ds_name, DATASET_REGISTRY[ds_name]["default_samples"]
        )
        split = DATASET_REGISTRY[ds_name]["split_train"]
        converted = load_and_convert_dataset(ds_name, max_samples=max_s, split=split)
        if converted:
            all_converted[ds_name] = converted

    if not all_converted:
        print("ERROR: No datasets loaded. Exiting.")
        return

    # Blend
    weights = {}
    for name in all_converted:
        weights[name] = weight_overrides.get(
            name, DATASET_REGISTRY[name]["default_weight"]
        )

    blended = blend_datasets(all_converted, weights)

    # ── Apply chat template ──
    print("\n[3/6] Applying chat template...")
    hf_dataset = examples_to_hf_dataset(blended)
    hf_dataset = hf_dataset.map(
        lambda ex: apply_chat_template(ex, tokenizer), desc="Applying chat template"
    )

    # Filter nulls and too-long
    n_before = len(hf_dataset)
    hf_dataset = hf_dataset.filter(lambda x: x["text"] is not None)
    hf_dataset = hf_dataset.filter(
        lambda x: len(tokenizer.encode(x["text"])) <= args.max_seq_length
    )
    n_after = len(hf_dataset)
    print(f"  Filtered: {n_before} → {n_after} ({n_before - n_after} removed)")

    # Train/eval split
    split = hf_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Quick stats
    lengths = [
        len(tokenizer.encode(x["text"]))
        for x in train_dataset.select(range(min(500, len(train_dataset))))
    ]
    print(
        f"  Token lengths (sample) - mean: {np.mean(lengths):.0f}, max: {max(lengths)}, min: {min(lengths)}"
    )

    # Also load the workplace validation set for eval if available
    eval_raw_dataset = None
    if "nemotron_workplace" in args.datasets and not args.skip_eval:
        try:
            eval_raw_dataset = load_dataset(
                "nvidia/Nemotron-RL-agent-workplace_assistant", split="validation"
            )
            print(
                f"  Loaded workplace validation set: {len(eval_raw_dataset)} examples"
            )
        except Exception as e:
            print(f"  ⚠ Could not load workplace validation: {e}")

    # ── Load model ──
    print("\n[4/6] Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "device_map": "auto",
    }

    if not args.full_finetune:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── LoRA ──
    peft_config = None
    if not args.full_finetune:
        print("\n[4.5/6] Setting up LoRA (stage 2)...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_rslora=True,
        )
        print(
            f"  rsLoRA: alpha/√r = {args.lora_alpha}/{args.lora_r**0.5:.1f} = {args.lora_alpha / args.lora_r**0.5:.2f}"
        )

    # ── Completion-only masking ──
    print("\n[4.6/6] Setting up completion-only loss masking...")
    response_template = detect_response_template(tokenizer)
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    print(
        f"  Response template: {repr(response_template)} → IDs: {response_template_ids}"
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tokenizer
    )

    # ── Training args ──
    run_name = (
        args.wandb_run_name
        or f"stage2-blend-r{args.lora_r}-lr{args.lr}-ep{args.epochs}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ── W&B ──
    wandb_callback = None
    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "stage": 2,
                "base_model": args.base_model,
                "datasets": args.datasets,
                "blend_weights": weights,
                "total_examples": len(train_dataset),
                "lr": args.lr,
                "epochs": args.epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_seq_length": args.max_seq_length,
            },
            tags=["stage2", "multi-dataset", "sft"],
        )

        # Log dataset composition
        source_counts = Counter(ex["source"] for ex in blended)
        comp_table = wandb.Table(
            columns=["source", "count", "weight", "pct"],
            data=[
                [src, cnt, weights.get(src, 1.0), round(100 * cnt / len(blended), 1)]
                for src, cnt in source_counts.most_common()
            ],
        )
        wandb.log({"dataset/composition": comp_table})
        wandb_callback = WandbCallback()

    # ── Trainer ──
    callbacks = [wandb_callback] if wandb_callback else []
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if not args.full_finetune:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"\n  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    # ── Pre-training eval ──
    pre_eval_metrics = None
    if eval_raw_dataset and not args.skip_eval:
        print("\n[5a/6] PRE-TRAINING eval (stage 2 baseline)...")
        pre_eval_metrics, pre_eval_examples = run_eval(
            trainer.model,
            tokenizer,
            eval_raw_dataset,
            max_examples=args.eval_max_examples,
            desc="pre_s2_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(pre_eval_metrics, pre_eval_examples, "pre_s2_eval")

    # ── Train ──
    print("\n[5/6] Training stage 2...")
    trainer.train()

    # ── Save ──
    print("\n[6/6] Saving...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    merged_dir = args.output_dir
    if not args.full_finetune:
        merged_dir = args.output_dir + "-merged"
        print("  Merging LoRA weights...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"  Merged model saved to: {merged_dir}")

    # ── Post-training eval ──
    post_eval_metrics = None
    if eval_raw_dataset and not args.skip_eval:
        print("\nPOST-TRAINING eval (stage 2)...")
        post_eval_metrics, post_eval_examples = run_eval(
            trainer.model,
            tokenizer,
            eval_raw_dataset,
            max_examples=args.eval_max_examples,
            desc="post_s2_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(post_eval_metrics, post_eval_examples, "post_s2_eval")

        # Comparison
        if pre_eval_metrics and post_eval_metrics:
            print("\n  ══════════════════════════════════════════════")
            print("  ██ STAGE 2: PRE vs POST COMPARISON ██")
            print("  ══════════════════════════════════════════════")
            comparison_data = []
            for m in [
                "tool_name_exact_match",
                "tool_name_set_match",
                "first_tool_name_correct",
                "tool_name_f1",
                "argument_accuracy",
                "full_exact_match",
                "parse_success",
            ]:
                pre_val = pre_eval_metrics.get(f"pre_s2_eval/{m}", 0)
                post_val = post_eval_metrics.get(f"post_s2_eval/{m}", 0)
                delta = post_val - pre_val
                print(f"    {m:<30s}  {pre_val:.1%} → {post_val:.1%}  ({delta:+.1%})")
                comparison_data.append(
                    [
                        m,
                        round(pre_val * 100, 1),
                        round(post_val * 100, 1),
                        round(delta * 100, 1),
                    ]
                )

            if not args.no_wandb:
                import wandb

                wandb.log(
                    {
                        "s2_eval_comparison/summary": wandb.Table(
                            columns=["metric", "pre_%", "post_%", "delta_%"],
                            data=comparison_data,
                        )
                    }
                )

    # ── Hub upload ──
    if args.hub_repo:
        from huggingface_hub import HfApi, create_repo

        print(f"\n  Pushing to HuggingFace Hub: {args.hub_repo}")
        api = HfApi()
        try:
            create_repo(args.hub_repo, private=args.hub_private, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        upload_dir = merged_dir if not args.full_finetune else args.output_dir
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=args.hub_repo,
            commit_message=f"Stage 2 SFT: multi-dataset blend, rsLoRA r={args.lora_r}, {args.epochs} ep",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{args.hub_repo}")

        if not args.full_finetune:
            lora_repo = args.hub_repo + "-lora-v2"
            try:
                create_repo(lora_repo, private=args.hub_private, exist_ok=True)
                api.upload_folder(
                    folder_path=args.output_dir,
                    repo_id=lora_repo,
                    commit_message=f"Stage 2 LoRA: r={args.lora_r}, alpha={args.lora_alpha}",
                )
                print(f"  ✓ LoRA adapter: https://huggingface.co/{lora_repo}")
            except Exception as e:
                print(f"  LoRA upload: {e}")

    # ── Finalize ──
    print("\n" + "=" * 60)
    print("Stage 2 training complete!")
    print(f"  Checkpoint: {args.output_dir}")
    if not args.full_finetune:
        print(f"  Merged:     {merged_dir}")
    if args.hub_repo:
        print(f"  Hub:        https://huggingface.co/{args.hub_repo}")
    ds_summary = ", ".join(
        f"{name}({len(all_converted.get(name, []))})"
        for name in args.datasets
        if name in all_converted
    )
    print(f"  Datasets:   {ds_summary}")
    print(f"  Total train: {len(train_dataset)}")
    print("=" * 60)

    if not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
