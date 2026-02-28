#!/usr/bin/env python3
"""
SFT Fine-tuning script for Ministral-3-3B-Instruct on Nemotron workplace_assistant dataset.
Uses LoRA for efficient fine-tuning + TRL SFTTrainer.

Usage:
    python train_sft.py [--full_finetune] [--epochs 2] [--lr 2e-4] [--batch_size 2]
"""

import argparse
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
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
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# ──────────────────────────────────────────────
# 1. DATA PREPROCESSING
# ──────────────────────────────────────────────


def strip_nulls_from_tool(tool: dict) -> dict:
    """
    Strip null-valued parameters from tool schemas to reduce sequence length.
    The original dataset has ~30 null fields per tool which wastes tokens.
    """
    tool = copy.deepcopy(tool)
    if "parameters" in tool and "properties" in tool["parameters"]:
        props = tool["parameters"]["properties"]
        # Remove null-valued properties
        cleaned_props = {k: v for k, v in props.items() if v is not None}
        tool["parameters"]["properties"] = cleaned_props
        # Also clean required list to only include keys that exist
        if "required" in tool["parameters"]:
            req = tool["parameters"]["required"]
            if req:
                tool["parameters"]["required"] = [r for r in req if r in cleaned_props]
    return tool


def format_example_to_messages(example: dict) -> dict:
    """
    Convert a dataset row into a chat messages format suitable for SFT.

    Input format (from dataset):
        responses_create_params: {input: [{role, content}, ...], tools: [...], ...}
        ground_truth: [{"name": "tool_name", "arguments": "{...}"}, ...]

    Output format:
        messages: [
            {role: "system", content: "..."},
            {role: "user", content: "..."},
            {role: "assistant", tool_calls: [{id, type, function: {name, arguments}}]}
        ]
        tools: [cleaned tool schemas]
    """
    params = example["responses_create_params"]
    ground_truth = example["ground_truth"]

    # Extract messages from input
    input_messages = params["input"]

    # Clean tool schemas
    raw_tools = params.get("tools", [])
    cleaned_tools = [strip_nulls_from_tool(t) for t in raw_tools]

    # Build the assistant response with tool calls
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

    # Construct full message sequence
    messages = list(input_messages)  # system + user messages
    if tool_calls:
        messages.append({"role": "assistant", "tool_calls": tool_calls})
    else:
        # Edge case: no tool calls in ground truth
        messages.append({"role": "assistant", "content": "I'll help you with that."})

    return {"messages": messages, "tools": cleaned_tools}


def format_dataset(dataset):
    """Apply formatting to entire dataset."""
    formatted = dataset.map(
        format_example_to_messages,
        remove_columns=dataset.column_names,
        desc="Formatting examples",
    )
    return formatted


# ──────────────────────────────────────────────
# 2. TOKENIZATION / CHAT TEMPLATE
# ──────────────────────────────────────────────


def detect_response_template(tokenizer) -> str:
    """
    Auto-detect the token sequence that marks the start of assistant responses
    in the model's chat template. This is used by DataCollatorForCompletionOnlyLM
    to mask loss on everything before the assistant's generated content.
    """
    # Create a minimal example with a known assistant response
    marker = "___ASSISTANT_START_MARKER___"
    dummy_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": marker},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            dummy_messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback for models that need special handling
        rendered = tokenizer.apply_chat_template(dummy_messages, tokenize=False)

    # Find what comes immediately before our marker
    marker_idx = rendered.find(marker)
    if marker_idx == -1:
        raise ValueError(
            f"Could not find marker in rendered template: {rendered[:200]}"
        )

    # Walk backwards from marker to find a good boundary
    # Take the last 20 chars before the marker as potential template
    prefix = rendered[max(0, marker_idx - 50) : marker_idx]

    # Try progressively shorter suffixes until we find one that tokenizes cleanly
    # We want the shortest unique sequence that marks assistant turn start
    for start in range(len(prefix)):
        candidate = prefix[start:]
        if candidate.strip():
            # Verify this appears only at assistant turn boundaries
            # by checking it appears exactly once before our marker
            before_marker = rendered[:marker_idx]
            if before_marker.count(candidate) == 1:
                return candidate

    # Fallback: use the last 10 chars before marker
    fallback = prefix[-10:].strip()
    print(f"Warning: Using fallback response template: {repr(fallback)}")
    return fallback


def apply_chat_template(example, tokenizer):
    """
    Apply the tokenizer's chat template to format messages with tool definitions.
    Mistral models have native tool-calling chat templates.
    """
    messages = example["messages"]
    tools = example.get("tools", None)

    try:
        # Try with tools parameter (Mistral tokenizers support this)
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: inject tools into system message
        tool_desc = json.dumps(tools, indent=2) if tools else ""
        adjusted_messages = []
        for msg in messages:
            if msg["role"] == "system" and tool_desc:
                adjusted_messages.append(
                    {
                        "role": "system",
                        "content": msg["content"]
                        + f"\n\nAvailable tools:\n{tool_desc}",
                    }
                )
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                # Convert tool calls to text representation
                tc_text = json.dumps(msg["tool_calls"], indent=2)
                adjusted_messages.append(
                    {"role": "assistant", "content": f"[TOOL_CALLS]{tc_text}"}
                )
            else:
                adjusted_messages.append(msg)

        text = tokenizer.apply_chat_template(
            adjusted_messages, tokenize=False, add_generation_prompt=False
        )

    return {"text": text}


# ──────────────────────────────────────────────
# 3. EVALUATION HARNESS
# ──────────────────────────────────────────────


def parse_tool_calls_from_generation(text: str) -> list[dict]:
    """
    Parse tool calls from model-generated text.
    Handles Mistral native format, JSON arrays, and individual objects.
    Returns list of {"name": str, "arguments": dict}
    """
    tool_calls = []

    # Strategy 1: Mistral [TOOL_CALLS] marker
    for marker in ["[TOOL_CALLS]", "[TOOL_CALL]"]:
        if marker in text:
            json_part = text.split(marker, 1)[1].strip()
            # Find the JSON array/object
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
                            except json.JSONDecodeError:
                                pass
                        tool_calls.append(
                            {"name": tc.get("name", ""), "arguments": args}
                        )
                    return tool_calls
                except json.JSONDecodeError:
                    continue

    # Strategy 2: Find JSON arrays containing tool call objects
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
                                except json.JSONDecodeError:
                                    pass
                            tool_calls.append({"name": item["name"], "arguments": args})
                        return tool_calls
                except json.JSONDecodeError:
                    pass
                start = None

    # Strategy 3: Find individual {"name": ...} objects
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
                                except json.JSONDecodeError:
                                    pass
                            tool_calls.append(
                                {"name": parsed["name"], "arguments": args}
                            )
                    except json.JSONDecodeError:
                        pass
                    idx = j + 1
                    break
        else:
            break

    return tool_calls


def normalize_arguments(args: dict) -> dict:
    """Normalize argument values for comparison."""
    normalized = {}
    for k, v in args.items():
        if isinstance(v, str):
            normalized[k] = v.strip()
        else:
            normalized[k] = v
    return normalized


def compute_tool_call_metrics(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """
    Compute fine-grained metrics for a single example.
    """
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
        metrics["tool_name_set_match"] = 1 if not predicted else 0
        return metrics

    pred_names = [tc["name"] for tc in predicted]
    gt_names = [tc["name"] for tc in ground_truth]

    metrics["tool_name_exact_match"] = 1 if pred_names == gt_names else 0
    metrics["tool_name_set_match"] = 1 if set(pred_names) == set(gt_names) else 0
    if pred_names and gt_names:
        metrics["first_tool_name_correct"] = 1 if pred_names[0] == gt_names[0] else 0

    pred_name_counter = Counter(pred_names)
    gt_name_counter = Counter(gt_names)
    common = sum((pred_name_counter & gt_name_counter).values())
    metrics["tool_name_precision"] = common / max(len(pred_names), 1)
    metrics["tool_name_recall"] = common / max(len(gt_names), 1)
    if metrics["tool_name_precision"] + metrics["tool_name_recall"] > 0:
        metrics["tool_name_f1"] = (
            2
            * metrics["tool_name_precision"]
            * metrics["tool_name_recall"]
            / (metrics["tool_name_precision"] + metrics["tool_name_recall"])
        )

    # Argument-level accuracy for matched tool pairs
    total_args = 0
    correct_args = 0
    full_match = True

    for i, gt_call in enumerate(ground_truth):
        gt_args = gt_call.get("arguments", {})
        if isinstance(gt_args, str):
            try:
                gt_args = json.loads(gt_args)
            except json.JSONDecodeError:
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
            except json.JSONDecodeError:
                pred_args = {}

        gt_args = normalize_arguments(gt_args)
        pred_args = normalize_arguments(pred_args)

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
    """
    Run generation-based evaluation on a dataset split.
    Returns aggregate metrics dict and per-example results for W&B tables.
    """
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
        cleaned_tools = [strip_nulls_from_tool(t) for t in raw_tools]

        try:
            input_text = tokenizer.apply_chat_template(
                input_messages,
                tools=cleaned_tools if cleaned_tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            tool_desc = json.dumps(cleaned_tools, indent=2) if cleaned_tools else ""
            adjusted = []
            for msg in input_messages:
                if msg["role"] == "system" and tool_desc:
                    adjusted.append(
                        {
                            "role": "system",
                            "content": msg["content"]
                            + f"\n\nAvailable tools:\n{tool_desc}",
                        }
                    )
                else:
                    adjusted.append(msg)
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

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Parse predictions
        predicted_calls = parse_tool_calls_from_generation(generated_text)

        # Format ground truth
        gt_calls = []
        for gt in ground_truth:
            args = gt["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            gt_calls.append({"name": gt["name"], "arguments": args})

        example_metrics = compute_tool_call_metrics(predicted_calls, gt_calls)
        for k, v in example_metrics.items():
            all_metrics[k].append(v)

        user_msg = ""
        for msg in input_messages:
            if msg["role"] == "user":
                user_msg = msg["content"][:150]

        per_example_results.append(
            {
                "idx": i,
                "user_query": user_msg,
                "gt_tools": ", ".join(tc["name"] for tc in gt_calls),
                "pred_tools": ", ".join(tc["name"] for tc in predicted_calls),
                "name_match": example_metrics["tool_name_exact_match"],
                "full_match": example_metrics["full_exact_match"],
                "arg_accuracy": example_metrics["argument_accuracy"],
                "generated_text": generated_text[:300],
            }
        )

    # Aggregate
    agg = {}
    for k, values in all_metrics.items():
        agg[f"{desc}/{k}"] = np.mean(values)
    agg[f"{desc}/n_examples"] = len(examples)

    print(f"\n  ── {desc} Results ({len(examples)} examples) ──")
    print(f"    Tool name exact match:   {agg[f'{desc}/tool_name_exact_match']:.1%}")
    print(f"    Tool name set match:     {agg[f'{desc}/tool_name_set_match']:.1%}")
    print(f"    First tool correct:      {agg[f'{desc}/first_tool_name_correct']:.1%}")
    print(f"    Tool name F1:            {agg[f'{desc}/tool_name_f1']:.1%}")
    print(f"    Argument accuracy:       {agg[f'{desc}/argument_accuracy']:.1%}")
    print(f"    Full exact match:        {agg[f'{desc}/full_exact_match']:.1%}")
    print(f"    Parse success rate:      {agg[f'{desc}/parse_success']:.1%}")

    return agg, per_example_results


def log_eval_to_wandb(metrics: dict, per_example: list[dict], phase: str):
    """Log eval results to W&B with tables and bar charts."""
    import wandb

    wandb.log(metrics)

    # Per-example results table
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

    # Error analysis: which tools are most often wrong
    tool_errors = Counter()
    tool_correct = Counter()
    for r in per_example:
        gt_set = set(r["gt_tools"].split(", ")) if r["gt_tools"] else set()
        pred_set = set(r["pred_tools"].split(", ")) if r["pred_tools"] else set()
        for tool in gt_set:
            if tool in pred_set:
                tool_correct[tool] += 1
            else:
                tool_errors[tool] += 1

    if tool_errors or tool_correct:
        all_tools = set(list(tool_errors.keys()) + list(tool_correct.keys()))
        error_table = wandb.Table(
            columns=["tool_name", "correct", "missed", "accuracy"],
            data=[
                [
                    tool,
                    tool_correct.get(tool, 0),
                    tool_errors.get(tool, 0),
                    round(
                        tool_correct.get(tool, 0)
                        / max(tool_correct.get(tool, 0) + tool_errors.get(tool, 0), 1)
                        * 100,
                        1,
                    ),
                ]
                for tool in sorted(all_tools)
            ],
        )
        wandb.log({f"{phase}/per_tool_accuracy": error_table})


# ──────────────────────────────────────────────
# 4. WEIGHTS & BIASES RICH LOGGING
# ──────────────────────────────────────────────


def log_dataset_analysis(train_dataset, eval_dataset, tokenizer, raw_dataset):
    """
    Log comprehensive dataset analysis to W&B at the start of training.
    Creates tables, histograms, and summary stats that look great on the dashboard.
    """
    import wandb

    # ── Token length distributions ──
    train_lengths = [len(tokenizer.encode(x["text"])) for x in train_dataset]
    eval_lengths = [len(tokenizer.encode(x["text"])) for x in eval_dataset]

    wandb.log(
        {
            "dataset/train_token_lengths": wandb.Histogram(train_lengths, num_bins=50),
            "dataset/eval_token_lengths": wandb.Histogram(eval_lengths, num_bins=50),
            "dataset/train_size": len(train_dataset),
            "dataset/eval_size": len(eval_dataset),
            "dataset/train_mean_tokens": np.mean(train_lengths),
            "dataset/train_median_tokens": np.median(train_lengths),
            "dataset/train_max_tokens": max(train_lengths),
            "dataset/train_min_tokens": min(train_lengths),
            "dataset/train_std_tokens": np.std(train_lengths),
            "dataset/train_total_tokens": sum(train_lengths),
        }
    )

    # ── Tool usage distribution ──
    tool_counts = Counter()
    tools_per_example = []
    category_counts = Counter()

    for example in raw_dataset["train"]:
        gt = example["ground_truth"]
        n_tools = len(gt)
        tools_per_example.append(n_tools)
        for call in gt:
            tool_counts[call["name"]] += 1
        if "category" in example and example["category"]:
            category_counts[example["category"]] += 1

    # Tool frequency bar chart as a W&B Table
    tool_table = wandb.Table(
        columns=["tool_name", "count", "percentage"],
        data=[
            [name, count, round(100 * count / sum(tool_counts.values()), 1)]
            for name, count in tool_counts.most_common()
        ],
    )
    wandb.log(
        {
            "dataset/tool_usage": tool_table,
            "dataset/tools_per_example": wandb.Histogram(
                tools_per_example, num_bins=10
            ),
            "dataset/unique_tools": len(tool_counts),
            "dataset/avg_tools_per_example": np.mean(tools_per_example),
        }
    )

    # ── Category distribution ──
    if category_counts:
        cat_table = wandb.Table(
            columns=["category", "count", "percentage"],
            data=[
                [cat, count, round(100 * count / sum(category_counts.values()), 1)]
                for cat, count in category_counts.most_common()
            ],
        )
        wandb.log({"dataset/category_distribution": cat_table})

    # ── Null stripping effectiveness ──
    original_tool_tokens = 0
    cleaned_tool_tokens = 0
    for example in raw_dataset["train"]:
        raw_tools = example["responses_create_params"].get("tools", [])
        cleaned_tools = [strip_nulls_from_tool(t) for t in raw_tools]
        original_tool_tokens += len(json.dumps(raw_tools))
        cleaned_tool_tokens += len(json.dumps(cleaned_tools))

    reduction_pct = 100 * (1 - cleaned_tool_tokens / max(original_tool_tokens, 1))
    wandb.log(
        {
            "dataset/null_strip_original_chars": original_tool_tokens,
            "dataset/null_strip_cleaned_chars": cleaned_tool_tokens,
            "dataset/null_strip_reduction_pct": reduction_pct,
        }
    )

    # ── Sample examples table ──
    sample_table = wandb.Table(
        columns=["idx", "user_query", "tool_calls", "n_tokens", "category"]
    )
    for i, example in enumerate(raw_dataset["train"]):
        if i >= 20:  # Log first 20 examples
            break
        user_msg = ""
        for msg in example["responses_create_params"]["input"]:
            if msg["role"] == "user":
                user_msg = msg["content"][:200]
        tool_calls_str = ", ".join(call["name"] for call in example["ground_truth"])
        n_tokens = train_lengths[i] if i < len(train_lengths) else 0
        category = example.get("category", "")
        sample_table.add_data(i, user_msg, tool_calls_str, n_tokens, category)

    wandb.log({"dataset/sample_examples": sample_table})


class WandbRichCallback(TrainerCallback):
    """
    Custom callback for rich W&B logging during training.
    Tracks gradients, LoRA weights, GPU utilization, loss components, and more.
    """

    def __init__(self, model, tokenizer, log_grad_every_n=10):
        self.model = model
        self.tokenizer = tokenizer
        self.log_grad_every_n = log_grad_every_n
        self.step_times = []
        self.last_step_time = None
        self.training_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        self.last_step_time = time.time()

        import wandb

        # Log model architecture summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        wandb.log(
            {
                "model/total_params_M": total_params / 1e6,
                "model/trainable_params_M": trainable_params / 1e6,
                "model/frozen_params_M": frozen_params / 1e6,
                "model/trainable_pct": 100 * trainable_params / total_params,
            }
        )

        # Log LoRA layer breakdown
        lora_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_layers.append(
                    [name, list(param.shape), param.numel(), f"{param.dtype}"]
                )

        if lora_layers:
            lora_table = wandb.Table(
                columns=["layer_name", "shape", "params", "dtype"], data=lora_layers
            )
            wandb.log({"model/lora_layers": lora_table})

        # Log GPU info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                wandb.log(
                    {
                        f"system/gpu_{i}_name": props.name,
                        f"system/gpu_{i}_vram_GB": props.total_mem / 1e9,
                    }
                )

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        import wandb

        now = time.time()
        step_time = now - self.last_step_time
        self.last_step_time = now
        self.step_times.append(step_time)

        step = state.global_step
        logs = {}

        # ── Throughput metrics ──
        logs["perf/step_time_seconds"] = step_time
        logs["perf/steps_per_second"] = 1.0 / max(step_time, 1e-6)
        if len(self.step_times) > 1:
            logs["perf/avg_step_time"] = np.mean(self.step_times[-50:])
            logs["perf/rolling_steps_per_sec"] = 1.0 / np.mean(self.step_times[-50:])

        elapsed = now - self.training_start_time
        logs["perf/elapsed_minutes"] = elapsed / 60
        if state.max_steps > 0:
            remaining_steps = state.max_steps - step
            est_remaining = remaining_steps * np.mean(self.step_times[-50:])
            logs["perf/eta_minutes"] = est_remaining / 60
            logs["perf/progress_pct"] = 100 * step / state.max_steps

        # ── GPU memory tracking ──
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_mem / 1e9
                logs[f"gpu/gpu_{i}_allocated_GB"] = mem_alloc
                logs[f"gpu/gpu_{i}_reserved_GB"] = mem_reserved
                logs[f"gpu/gpu_{i}_utilization_pct"] = 100 * mem_alloc / mem_total
                logs[f"gpu/gpu_{i}_fragmentation_pct"] = (
                    100 * (mem_reserved - mem_alloc) / max(mem_reserved, 1e-6)
                )

        # ── Gradient statistics (every N steps) ──
        if step % self.log_grad_every_n == 0:
            grad_norms = {}
            lora_A_norms = {}
            lora_B_norms = {}
            all_grad_norms = []

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    all_grad_norms.append(grad_norm)

                    # Track per-module-type gradient norms
                    for module_type in [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]:
                        if module_type in name:
                            key = f"gradients/{module_type}"
                            if key not in grad_norms:
                                grad_norms[key] = []
                            grad_norms[key].append(grad_norm)

                    # Track LoRA A vs B weight magnitudes
                    if "lora_A" in name:
                        for mod in [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]:
                            if mod in name:
                                lora_A_norms[mod] = param.data.norm(2).item()
                    elif "lora_B" in name:
                        for mod in [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]:
                            if mod in name:
                                lora_B_norms[mod] = param.data.norm(2).item()

            # Aggregate gradient stats
            if all_grad_norms:
                logs["gradients/global_norm"] = np.sqrt(
                    sum(g**2 for g in all_grad_norms)
                )
                logs["gradients/mean_norm"] = np.mean(all_grad_norms)
                logs["gradients/max_norm"] = max(all_grad_norms)
                logs["gradients/min_norm"] = min(all_grad_norms)
                logs["gradients/std_norm"] = np.std(all_grad_norms)

                # Detect vanishing/exploding gradients
                logs["gradients/pct_below_1e-7"] = (
                    100
                    * sum(1 for g in all_grad_norms if g < 1e-7)
                    / len(all_grad_norms)
                )
                logs["gradients/pct_above_10"] = (
                    100 * sum(1 for g in all_grad_norms if g > 10) / len(all_grad_norms)
                )

            # Per-module gradient norms (averaged)
            for key, norms in grad_norms.items():
                logs[f"{key}_mean"] = np.mean(norms)
                logs[f"{key}_max"] = max(norms)

            # LoRA weight magnitude tracking (A vs B)
            for mod in lora_A_norms:
                if mod in lora_B_norms:
                    logs[f"lora_weights/{mod}_A_norm"] = lora_A_norms[mod]
                    logs[f"lora_weights/{mod}_B_norm"] = lora_B_norms[mod]
                    logs[f"lora_weights/{mod}_effective_norm"] = (
                        lora_A_norms[mod] * lora_B_norms[mod]
                    )

        wandb.log(logs, step=step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import wandb

        if metrics:
            # Log eval metrics with cleaner names
            eval_logs = {}
            for k, v in metrics.items():
                clean_key = k.replace("eval_", "eval/")
                eval_logs[clean_key] = v

            # Perplexity from eval loss
            if "eval_loss" in metrics:
                eval_logs["eval/perplexity"] = math.exp(min(metrics["eval_loss"], 20))

            wandb.log(eval_logs, step=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhance default training logs with derived metrics."""
        import wandb

        if logs is None:
            return

        enhanced = {}
        # Training perplexity
        if "loss" in logs:
            enhanced["train/perplexity"] = math.exp(min(logs["loss"], 20))
        if "learning_rate" in logs:
            enhanced["train/lr_log10"] = math.log10(max(logs["learning_rate"], 1e-10))
        # Effective batch size
        enhanced["train/effective_batch_size"] = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(1, args.world_size if hasattr(args, "world_size") else 1)
        )
        if enhanced:
            wandb.log(enhanced, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        import wandb

        total_time = time.time() - self.training_start_time
        wandb.log(
            {
                "summary/total_training_minutes": total_time / 60,
                "summary/total_steps": state.global_step,
                "summary/avg_step_time": np.mean(self.step_times)
                if self.step_times
                else 0,
                "summary/final_train_loss": state.log_history[-1].get("loss", 0)
                if state.log_history
                else 0,
                "summary/best_eval_loss": min(
                    (h.get("eval_loss", float("inf")) for h in state.log_history),
                    default=0,
                ),
            }
        )

        # Step time distribution
        if self.step_times:
            wandb.log(
                {
                    "summary/step_time_histogram": wandb.Histogram(
                        self.step_times, num_bins=30
                    )
                }
            )


def init_wandb(args, model, tokenizer, train_dataset, eval_dataset, raw_dataset):
    """Initialize W&B run with comprehensive config and dataset analysis."""
    import wandb

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        # Model
        "model_name": args.model_name,
        "model_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "trainable_pct": 100 * trainable_params / total_params,
        # LoRA
        "lora_enabled": not args.full_finetune,
        "lora_r": args.lora_r if not args.full_finetune else None,
        "lora_alpha": args.lora_alpha if not args.full_finetune else None,
        "lora_dropout": args.lora_dropout if not args.full_finetune else None,
        "rslora": True if not args.full_finetune else False,
        "rslora_scaling": (
            args.lora_alpha / (args.lora_r**0.5) if not args.full_finetune else None
        ),
        "lora_target_modules": "q,k,v,o,gate,up,down",
        # Training
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lr_scheduler": "cosine",
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "packing": False,
        # Data
        "dataset": args.dataset_name,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "completion_only_masking": True,
        "null_stripping": True,
        # System
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
    }

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"ministral3b-sft-r{args.lora_r}-e{args.epochs}",
        config=config,
        tags=[
            "sft",
            "ministral-3b",
            "tool-calling",
            "agentic",
            f"lora-r{args.lora_r}" if not args.full_finetune else "full-ft",
            "rslora",
            "completion-masking",
            "hackathon",
        ],
    )

    # Log dataset analysis charts
    print("\n  Logging dataset analysis to W&B...")
    log_dataset_analysis(train_dataset, eval_dataset, tokenizer, raw_dataset)

    return wandb.run


# ──────────────────────────────────────────────
# 4. TRAINING SETUP
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SFT training for Ministral-3B")
    parser.add_argument(
        "--model_name", type=str, default="mistralai/Ministral-3-3B-Instruct-2512-BF16"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nvidia/Nemotron-RL-agent-workplace_assistant",
    )
    parser.add_argument("--output_dir", type=str, default="./ministral-3b-agent-sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Do full fine-tuning instead of LoRA",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="HuggingFace Hub repo to push to (e.g. 'your-username/ministral-3b-agent')",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        default=False,
        help="Make the Hub repo private",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ministral-3b-agent",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", default=False, help="Disable W&B logging"
    )
    parser.add_argument(
        "--eval_max_examples",
        type=int,
        default=100,
        help="Max examples for pre/post-training generation eval (set lower for speed)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Skip pre/post-training generation eval",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ministral-3B SFT Training")
    print("=" * 60)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset_name}")
    print(f"LoRA:     {not args.full_finetune}")
    print(f"Epochs:   {args.epochs}")
    print(f"LR:       {args.lr}")
    print(f"Batch:    {args.batch_size} x {args.gradient_accumulation_steps} accum")
    print(f"Max Seq:  {args.max_seq_length}")
    print("=" * 60)

    # ── Load tokenizer ──
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load and format dataset ──
    print("\n[2/5] Loading and formatting dataset...")
    dataset = load_dataset(args.dataset_name)

    train_dataset = format_dataset(dataset["train"])
    eval_dataset = format_dataset(dataset["validation"])

    # Apply chat template
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        desc="Applying chat template (train)",
    )
    eval_dataset = eval_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        desc="Applying chat template (eval)",
    )

    # Print a sample
    print("\n── Sample formatted text (first 500 chars) ──")
    print(train_dataset[0]["text"][:500])
    print("...")

    # Quick stats
    train_lengths = [len(tokenizer.encode(x["text"])) for x in train_dataset]
    print(f"\nTrain set: {len(train_dataset)} examples")
    print(f"Eval set:  {len(eval_dataset)} examples")
    print(
        f"Token lengths - mean: {sum(train_lengths) / len(train_lengths):.0f}, "
        f"max: {max(train_lengths)}, min: {min(train_lengths)}"
    )

    # Filter out examples that exceed max_seq_length
    n_before = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer.encode(x["text"])) <= args.max_seq_length
    )
    n_filtered = n_before - len(train_dataset)
    if n_filtered > 0:
        print(f"Filtered {n_filtered} examples exceeding {args.max_seq_length} tokens")

    # ── Load model ──
    print("\n[3/5] Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "device_map": "auto",
    }

    # If not doing full finetune, we can load in 4-bit for memory efficiency
    if not args.full_finetune:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Setup LoRA ──
    peft_config = None
    if not args.full_finetune:
        print("\n[3.5/5] Setting up LoRA...")
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
            use_rslora=True,  # Rank-stabilized LoRA: scales by alpha/√r instead of alpha/r
        )
        print(
            f"  rsLoRA enabled: scaling factor = alpha/√r = {args.lora_alpha}/{args.lora_r**0.5:.1f} = {args.lora_alpha / args.lora_r**0.5:.2f}"
        )

    # ── Setup completion-only loss masking ──
    print("\n[3.5b/5] Setting up completion-only loss masking...")
    response_template = detect_response_template(tokenizer)
    print(f"  Detected response template: {repr(response_template)}")

    # Tokenize the response template to get token IDs
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    print(f"  Response template token IDs: {response_template_ids}")

    # Verify by decoding back
    decoded_back = tokenizer.decode(response_template_ids)
    print(f"  Decoded back: {repr(decoded_back)}")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tokenizer
    )

    # ── Training arguments ──
    print("\n[4/5] Setting up training...")
    training_args = SFTConfig(
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
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if args.gradient_checkpointing
        else None,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none" if args.no_wandb else "wandb",
        # NOTE: packing=False required when using DataCollatorForCompletionOnlyLM
        # Packing and completion-only masking are mutually exclusive in TRL.
        packing=False,
    )

    # ── Initialize W&B ──
    wandb_callback = None
    if not args.no_wandb:
        print("\n[4.5/5] Initializing Weights & Biases...")
        init_wandb(args, model, tokenizer, train_dataset, eval_dataset, dataset)
        wandb_callback = WandbRichCallback(model, tokenizer, log_grad_every_n=5)

    # ── Create trainer ──
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

    # Print trainable params
    if not args.full_finetune:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    # ── Pre-training eval (baseline) ──
    pre_eval_metrics = None
    if not args.skip_eval:
        print("\n[4.8/5] Running PRE-TRAINING evaluation (baseline)...")
        pre_eval_metrics, pre_eval_examples = run_eval(
            trainer.model,
            tokenizer,
            dataset["validation"],
            max_examples=args.eval_max_examples,
            desc="pre_train_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(pre_eval_metrics, pre_eval_examples, "pre_train_eval")

    # ── Train ──
    print("\n[5/5] Starting training...")
    trainer.train()

    # ── Save ──
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    merged_dir = args.output_dir  # default for full finetune
    # If LoRA, also save merged model
    if not args.full_finetune:
        print("Merging LoRA weights and saving full model...")
        merged_dir = args.output_dir + "-merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    # ── Post-training eval ──
    post_eval_metrics = None
    if not args.skip_eval:
        print("\n[POST] Running POST-TRAINING evaluation...")
        post_eval_metrics, post_eval_examples = run_eval(
            trainer.model,
            tokenizer,
            dataset["validation"],
            max_examples=args.eval_max_examples,
            desc="post_train_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(post_eval_metrics, post_eval_examples, "post_train_eval")

        # ── Comparison summary ──
        if pre_eval_metrics and post_eval_metrics:
            print("\n  ══════════════════════════════════════════════")
            print("  ██ PRE vs POST TRAINING COMPARISON ██")
            print("  ══════════════════════════════════════════════")

            comparison_data = []
            key_metrics = [
                "tool_name_exact_match",
                "tool_name_set_match",
                "first_tool_name_correct",
                "tool_name_f1",
                "argument_accuracy",
                "full_exact_match",
                "parse_success",
            ]
            for metric in key_metrics:
                pre_val = pre_eval_metrics.get(f"pre_train_eval/{metric}", 0)
                post_val = post_eval_metrics.get(f"post_train_eval/{metric}", 0)
                delta = post_val - pre_val
                delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
                print(
                    f"    {metric:<30s}  {pre_val:.1%} → {post_val:.1%}  ({delta_str})"
                )
                comparison_data.append(
                    [
                        metric,
                        round(pre_val * 100, 1),
                        round(post_val * 100, 1),
                        round(delta * 100, 1),
                    ]
                )

            if not args.no_wandb:
                import wandb

                comparison_table = wandb.Table(
                    columns=["metric", "pre_train_%", "post_train_%", "delta_%"],
                    data=comparison_data,
                )
                wandb.log({"eval_comparison/summary": comparison_table})

                # Log individual deltas as scalars for easy dashboard charts
                for metric in key_metrics:
                    pre_val = pre_eval_metrics.get(f"pre_train_eval/{metric}", 0)
                    post_val = post_eval_metrics.get(f"post_train_eval/{metric}", 0)
                    wandb.log(
                        {
                            f"eval_comparison/{metric}_pre": pre_val,
                            f"eval_comparison/{metric}_post": post_val,
                            f"eval_comparison/{metric}_delta": post_val - pre_val,
                        }
                    )

    # ── Push to HuggingFace Hub ──
    if args.hub_repo:
        from huggingface_hub import HfApi, create_repo

        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            create_repo(args.hub_repo, private=args.hub_private, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        # Decide what to upload: merged model if LoRA, otherwise the output_dir
        upload_dir = merged_dir if (not args.full_finetune) else args.output_dir

        print(f"  Uploading from: {upload_dir}")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=args.hub_repo,
            commit_message=f"SFT fine-tuned Ministral-3B on Nemotron workplace_assistant (rsLoRA r={args.lora_r}, {args.epochs} epochs)",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{args.hub_repo}")

        # Also upload LoRA adapter separately if applicable
        if not args.full_finetune:
            lora_repo = args.hub_repo + "-lora"
            try:
                create_repo(lora_repo, private=args.hub_private, exist_ok=True)
                api.upload_folder(
                    folder_path=args.output_dir,
                    repo_id=lora_repo,
                    commit_message=f"LoRA adapter: rsLoRA r={args.lora_r}, alpha={args.lora_alpha}, {args.epochs} epochs",
                )
                print(
                    f"  ✓ LoRA adapter uploaded to https://huggingface.co/{lora_repo}"
                )
            except Exception as e:
                print(f"  LoRA upload skipped: {e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoint: {args.output_dir}")
    if not args.full_finetune:
        print(f"Merged:     {args.output_dir}-merged")
    if args.hub_repo:
        print(f"Hub:        https://huggingface.co/{args.hub_repo}")
    print("=" * 60)

    # ── Finalize W&B ──
    if not args.no_wandb:
        import wandb

        # Log final model as W&B artifact
        print("\nLogging model artifact to W&B...")
        artifact_dir = (
            merged_dir
            if (not args.full_finetune and os.path.exists(merged_dir))
            else args.output_dir
        )
        model_artifact = wandb.Artifact(
            name="ministral-3b-agent-sft",
            type="model",
            description=f"SFT fine-tuned Ministral-3B | rsLoRA r={args.lora_r} | {args.epochs} epochs",
            metadata={
                "base_model": args.model_name,
                "dataset": args.dataset_name,
                "lora_r": args.lora_r,
                "epochs": args.epochs,
                "final_loss": trainer.state.log_history[-1].get("loss", None)
                if trainer.state.log_history
                else None,
                "pre_train_full_exact_match": pre_eval_metrics.get(
                    "pre_train_eval/full_exact_match"
                )
                if pre_eval_metrics
                else None,
                "post_train_full_exact_match": post_eval_metrics.get(
                    "post_train_eval/full_exact_match"
                )
                if post_eval_metrics
                else None,
            },
        )
        model_artifact.add_dir(artifact_dir)
        wandb.log_artifact(model_artifact)

        # Log training summary table
        summary_data = [
            ["Base Model", args.model_name],
            ["Dataset", args.dataset_name],
            ["Train Examples", str(len(train_dataset))],
            ["Eval Examples", str(len(eval_dataset))],
            ["LoRA Rank", str(args.lora_r)],
            ["rsLoRA Scaling", f"{args.lora_alpha / args.lora_r**0.5:.2f}"],
            ["Epochs", str(args.epochs)],
            ["Learning Rate", str(args.lr)],
            [
                "Effective Batch Size",
                str(args.batch_size * args.gradient_accumulation_steps),
            ],
            ["Completion Masking", "Yes"],
            ["Null Stripping", "Yes"],
        ]
        if pre_eval_metrics:
            summary_data.append(
                [
                    "Pre-Train Full Exact Match",
                    f"{pre_eval_metrics.get('pre_train_eval/full_exact_match', 0):.1%}",
                ]
            )
            summary_data.append(
                [
                    "Pre-Train Tool Name F1",
                    f"{pre_eval_metrics.get('pre_train_eval/tool_name_f1', 0):.1%}",
                ]
            )
        if post_eval_metrics:
            summary_data.append(
                [
                    "Post-Train Full Exact Match",
                    f"{post_eval_metrics.get('post_train_eval/full_exact_match', 0):.1%}",
                ]
            )
            summary_data.append(
                [
                    "Post-Train Tool Name F1",
                    f"{post_eval_metrics.get('post_train_eval/tool_name_f1', 0):.1%}",
                ]
            )
        if pre_eval_metrics and post_eval_metrics:
            delta = post_eval_metrics.get(
                "post_train_eval/full_exact_match", 0
            ) - pre_eval_metrics.get("pre_train_eval/full_exact_match", 0)
            summary_data.append(["Δ Full Exact Match", f"{delta:+.1%}"])

        summary_table = wandb.Table(columns=["metric", "value"], data=summary_data)
        wandb.log({"summary/training_config": summary_table})
        wandb.finish()
        print("  ✓ W&B run finalized")


if __name__ == "__main__":
    main()
