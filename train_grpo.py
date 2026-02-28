#!/usr/bin/env python3
"""
Stage 3 GRPO: Reinforcement Learning for tool-calling with verifiable rewards.

Runs on top of the Stage 2 SFT model. Uses Group Relative Policy Optimization
with multiple decomposed reward functions that verify tool-calling correctness
without any human labels — pure rule-based rewards.

Reward signals:
  1. format_reward:     Valid, parseable tool-call JSON format        (0 or 1)
  2. tool_name_reward:  Correct tool names selected                   (0 to 1)
  3. argument_reward:   Correct argument key-value pairs              (0 to 1)
  4. completeness_reward: Right number of tool calls                  (0 or 1)
  5. ordering_reward:    Tools called in correct order                (0 or 1)

3-stage pipeline:
    Stage 1 (train_sft.py)    → Learn tool-calling format on workplace data
    Stage 2 (train_sft_v2.py) → Broaden with multi-dataset blend
    Stage 3 (train_grpo.py)   → RL-sharpen with verifiable rewards ← this file

Usage:
    python train_grpo.py \
        --base_model ./ministral-3b-agent-sft-v2-merged \
        --wandb_project mistral-hackathon

    # With vLLM generation server (faster, needs 2nd GPU or enough VRAM):
    python train_grpo.py \
        --base_model ./ministral-3b-agent-sft-v2-merged \
        --use_vllm
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
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────
# 1. TOOL-CALL PARSING (shared with eval harness)
# ──────────────────────────────────────────────


def parse_tool_calls(text: str) -> list[dict]:
    """
    Parse tool calls from model generation. Handles:
      - Mistral [TOOL_CALLS] [...] format
      - Raw JSON arrays of {name, arguments}
      - Individual JSON objects with "name" key
    Returns list of {"name": str, "arguments": dict}.
    """
    tool_calls = []

    # Strategy 1: [TOOL_CALLS] marker
    for marker in ["[TOOL_CALLS]", "[TOOL_CALL]"]:
        if marker in text:
            json_part = text.split(marker, 1)[1].strip()
            # Try the first line, or everything until a non-JSON marker
            candidates = [json_part.split("\n")[0].strip(), json_part.strip()]
            for candidate in candidates:
                # Find the JSON array
                bracket_start = candidate.find("[")
                if bracket_start != -1:
                    depth = 0
                    for i in range(bracket_start, len(candidate)):
                        if candidate[i] == "[":
                            depth += 1
                        elif candidate[i] == "]":
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(
                                        candidate[bracket_start : i + 1]
                                    )
                                    for tc in (
                                        parsed if isinstance(parsed, list) else [parsed]
                                    ):
                                        args = tc.get("arguments", {})
                                        if isinstance(args, str):
                                            try:
                                                args = json.loads(args)
                                            except:
                                                pass
                                        tool_calls.append(
                                            {
                                                "name": tc.get("name", ""),
                                                "arguments": args,
                                            }
                                        )
                                    return tool_calls
                                except json.JSONDecodeError:
                                    pass
                                break
                # Try as single object
                brace_start = candidate.find("{")
                if brace_start != -1:
                    depth = 0
                    for i in range(brace_start, len(candidate)):
                        if candidate[i] == "{":
                            depth += 1
                        elif candidate[i] == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(candidate[brace_start : i + 1])
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
                                        return tool_calls
                                except:
                                    pass
                                break

    # Strategy 2: Find JSON array with "name" field anywhere in text
    arr_match = re.search(r'\[\s*\{[^}]*"name"[^}]*\}.*?\]', text, re.DOTALL)
    if arr_match:
        try:
            parsed = json.loads(arr_match.group(0))
            for tc in parsed if isinstance(parsed, list) else [parsed]:
                if "name" in tc:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    tool_calls.append({"name": tc["name"], "arguments": args})
            if tool_calls:
                return tool_calls
        except json.JSONDecodeError:
            pass

    # Strategy 3: Individual {"name": ...} objects
    idx = 0
    while idx < len(text):
        pos = text.find('"name"', idx)
        if pos == -1:
            break
        # Walk backwards to find opening brace
        brace_pos = text.rfind("{", max(0, pos - 20), pos)
        if brace_pos == -1:
            idx = pos + 6
            continue
        depth = 0
        for j in range(brace_pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[brace_pos : j + 1])
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


def normalize_value(v):
    """Normalize argument values for comparison."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    # Normalize booleans
    if s in ("true", "yes", "1"):
        return "true"
    if s in ("false", "no", "0"):
        return "false"
    return s


# ──────────────────────────────────────────────
# 2. REWARD FUNCTIONS
#    Each takes (completions, **kwargs) per TRL spec.
#    kwargs contains dataset columns like ground_truth_names,
#    ground_truth_arguments, tools, etc.
# ──────────────────────────────────────────────


def format_reward(completions, **kwargs):
    """
    Reward 1: Is the output valid, parseable tool-call JSON?

    +1.0 if we can parse at least one valid tool call (or if ground truth
         expects zero calls and model output has no tool calls)
    +0.5 if output contains tool-call-like structure but parsing is partial
    +0.0 if output is unparseable garbage
    """
    ground_truth_names = kwargs.get("ground_truth_names", [None] * len(completions))
    rewards = []

    for completion, gt_names_str in zip(completions, ground_truth_names):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        # Parse ground truth expected count
        gt_names = []
        if gt_names_str:
            try:
                gt_names = (
                    json.loads(gt_names_str)
                    if isinstance(gt_names_str, str)
                    else gt_names_str
                )
            except:
                pass

        parsed = parse_tool_calls(text)

        if len(gt_names) == 0:
            # Ground truth expects no tool call
            if len(parsed) == 0:
                rewards.append(1.0)
            else:
                # Called tools when shouldn't have — minor penalty
                rewards.append(0.2)
        else:
            if len(parsed) > 0:
                rewards.append(1.0)
            elif "[TOOL_CALLS]" in text or '"name"' in text:
                # Attempted but malformed
                rewards.append(0.3)
            else:
                rewards.append(0.0)

    return rewards


def tool_name_reward(completions, **kwargs):
    """
    Reward 2: Did the model call the RIGHT tools?

    Computed as F1 between predicted and ground-truth tool name multisets.
    """
    ground_truth_names = kwargs.get("ground_truth_names", [None] * len(completions))
    rewards = []

    for completion, gt_names_str in zip(completions, ground_truth_names):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        gt_names = []
        if gt_names_str:
            try:
                gt_names = (
                    json.loads(gt_names_str)
                    if isinstance(gt_names_str, str)
                    else gt_names_str
                )
            except:
                pass

        if not gt_names:
            parsed = parse_tool_calls(text)
            # No tools expected; reward if none called
            rewards.append(1.0 if len(parsed) == 0 else 0.0)
            continue

        parsed = parse_tool_calls(text)
        pred_names = [tc["name"] for tc in parsed]

        if not pred_names:
            rewards.append(0.0)
            continue

        # F1 score on tool name multisets
        common = sum((Counter(pred_names) & Counter(gt_names)).values())
        precision = common / len(pred_names)
        recall = common / len(gt_names)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rewards.append(f1)

    return rewards


def argument_reward(completions, **kwargs):
    """
    Reward 3: Did the model pass the correct arguments?

    For each ground-truth tool call, checks fraction of argument key-value
    pairs that match. Averaged across all expected calls.
    """
    ground_truth_names = kwargs.get("ground_truth_names", [None] * len(completions))
    ground_truth_args = kwargs.get("ground_truth_arguments", [None] * len(completions))
    rewards = []

    for completion, gt_names_str, gt_args_str in zip(
        completions, ground_truth_names, ground_truth_args
    ):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        gt_names = []
        if gt_names_str:
            try:
                gt_names = (
                    json.loads(gt_names_str)
                    if isinstance(gt_names_str, str)
                    else gt_names_str
                )
            except:
                pass

        gt_arguments = []
        if gt_args_str:
            try:
                gt_arguments = (
                    json.loads(gt_args_str)
                    if isinstance(gt_args_str, str)
                    else gt_args_str
                )
            except:
                pass

        if not gt_names or not gt_arguments:
            rewards.append(0.0)
            continue

        parsed = parse_tool_calls(text)

        if not parsed:
            rewards.append(0.0)
            continue

        # Match predicted calls to ground truth by name
        total_kv, correct_kv = 0, 0

        for i, (gt_name, gt_args) in enumerate(zip(gt_names, gt_arguments)):
            if isinstance(gt_args, str):
                try:
                    gt_args = json.loads(gt_args)
                except:
                    gt_args = {}

            # Find matching predicted call
            matched = None
            for pred in parsed:
                if pred["name"] == gt_name:
                    matched = pred
                    break

            if matched is None:
                total_kv += max(len(gt_args), 1)
                continue

            pred_args = matched.get("arguments", {})
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except:
                    pred_args = {}

            for key, gt_val in gt_args.items():
                total_kv += 1
                if key in pred_args:
                    if normalize_value(pred_args[key]) == normalize_value(gt_val):
                        correct_kv += 1

        reward = correct_kv / max(total_kv, 1)
        rewards.append(reward)

    return rewards


def completeness_reward(completions, **kwargs):
    """
    Reward 4: Did the model call the right NUMBER of tools?

    +1.0 if count matches exactly
    +0.5 if within ±1
    +0.0 otherwise
    """
    ground_truth_names = kwargs.get("ground_truth_names", [None] * len(completions))
    rewards = []

    for completion, gt_names_str in zip(completions, ground_truth_names):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        gt_names = []
        if gt_names_str:
            try:
                gt_names = (
                    json.loads(gt_names_str)
                    if isinstance(gt_names_str, str)
                    else gt_names_str
                )
            except:
                pass

        parsed = parse_tool_calls(text)
        expected = len(gt_names)
        actual = len(parsed)

        if actual == expected:
            rewards.append(1.0)
        elif abs(actual - expected) == 1:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


def ordering_reward(completions, **kwargs):
    """
    Reward 5: Are the tools called in the correct ORDER?

    Uses longest common subsequence ratio to handle partial ordering.
    """
    ground_truth_names = kwargs.get("ground_truth_names", [None] * len(completions))
    rewards = []

    for completion, gt_names_str in zip(completions, ground_truth_names):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        gt_names = []
        if gt_names_str:
            try:
                gt_names = (
                    json.loads(gt_names_str)
                    if isinstance(gt_names_str, str)
                    else gt_names_str
                )
            except:
                pass

        if not gt_names:
            rewards.append(1.0)
            continue

        parsed = parse_tool_calls(text)
        pred_names = [tc["name"] for tc in parsed]

        if not pred_names:
            rewards.append(0.0)
            continue

        # Exact order match
        if pred_names == gt_names:
            rewards.append(1.0)
            continue

        # LCS ratio for partial credit
        def lcs_len(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        lcs = lcs_len(pred_names, gt_names)
        ratio = lcs / len(gt_names)
        rewards.append(ratio)

    return rewards


# ──────────────────────────────────────────────
# 3. DATASET PREPARATION
# ──────────────────────────────────────────────


def strip_nulls(tool: dict) -> dict:
    """Strip null-valued parameters from tool schemas."""
    tool = copy.deepcopy(tool)
    if "parameters" in tool and "properties" in tool["parameters"]:
        props = tool["parameters"]["properties"]
        cleaned = {k: v for k, v in props.items() if v is not None}
        tool["parameters"]["properties"] = cleaned
        if "required" in tool["parameters"] and tool["parameters"]["required"]:
            tool["parameters"]["required"] = [
                r for r in tool["parameters"]["required"] if r in cleaned
            ]
    return tool


def prepare_grpo_dataset(raw_dataset, tokenizer, max_prompt_len=3072):
    """
    Convert the Nemotron workplace_assistant dataset into GRPO format.

    GRPOTrainer expects:
      - "prompt": list of messages (or formatted string)
      - Extra columns for reward function kwargs

    We store ground truth as JSON strings so reward functions can parse them.
    """
    prompts = []

    for example in raw_dataset:
        params = example["responses_create_params"]
        ground_truth = example["ground_truth"]
        input_messages = params["input"]
        raw_tools = params.get("tools", [])
        cleaned_tools = [strip_nulls(t) for t in raw_tools]

        # Build prompt (everything up to assistant turn)
        try:
            prompt_text = tokenizer.apply_chat_template(
                input_messages,
                tools=cleaned_tools if cleaned_tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback
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
            try:
                prompt_text = tokenizer.apply_chat_template(
                    adjusted, tokenize=False, add_generation_prompt=True
                )
            except:
                continue

        # Skip if too long
        tokens = tokenizer.encode(prompt_text)
        if len(tokens) > max_prompt_len:
            continue

        # Extract ground truth
        gt_names = [gt["name"] for gt in ground_truth]
        gt_arguments = []
        for gt in ground_truth:
            args = gt["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            gt_arguments.append(args)

        prompts.append(
            {
                "prompt": prompt_text,
                "ground_truth_names": json.dumps(gt_names),
                "ground_truth_arguments": json.dumps(gt_arguments),
            }
        )

    print(f"  Prepared {len(prompts)} prompts for GRPO")
    return Dataset.from_list(prompts)


# ──────────────────────────────────────────────
# 4. W&B LOGGING
# ──────────────────────────────────────────────


def log_reward_analysis(trainer, dataset, tokenizer, n_samples=50):
    """Generate and score a batch of samples for reward analysis."""
    import wandb

    model = trainer.model
    model.eval()

    sample_indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    results = []

    for idx in sample_indices:
        example = dataset[idx]
        prompt = example["prompt"]
        gt_names = json.loads(example["ground_truth_names"])

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        parsed = parse_tool_calls(generated)
        pred_names = [tc["name"] for tc in parsed]

        results.append(
            {
                "gt_tools": ", ".join(gt_names),
                "pred_tools": ", ".join(pred_names),
                "exact_match": 1 if pred_names == gt_names else 0,
                "generated": generated[:300],
            }
        )

    table = wandb.Table(
        columns=["gt_tools", "pred_tools", "exact_match", "generated"],
        data=[
            [r["gt_tools"], r["pred_tools"], r["exact_match"], r["generated"]]
            for r in results
        ],
    )
    exact_match_rate = np.mean([r["exact_match"] for r in results])
    wandb.log(
        {
            "grpo_eval/sample_results": table,
            "grpo_eval/exact_match_rate": exact_match_rate,
            "grpo_eval/n_samples": len(results),
        }
    )
    print(f"  GRPO eval: exact_match = {exact_match_rate:.1%} ({len(results)} samples)")

    model.train()
    return exact_match_rate


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Stage 3 GRPO: RL for tool-calling")

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ministral-3b-agent-sft-v2-merged",
        help="Stage 2 merged model (or any SFT checkpoint)",
    )
    parser.add_argument("--output_dir", type=str, default="./ministral-3b-agent-grpo")

    # GRPO
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt for group comparison",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )

    # Reward weights (sum of decomposed rewards)
    parser.add_argument(
        "--w_format", type=float, default=1.0, help="Weight for format reward"
    )
    parser.add_argument(
        "--w_tool_name",
        type=float,
        default=3.0,
        help="Weight for tool name reward (highest — most important)",
    )
    parser.add_argument(
        "--w_argument",
        type=float,
        default=2.0,
        help="Weight for argument accuracy reward",
    )
    parser.add_argument(
        "--w_completeness",
        type=float,
        default=1.5,
        help="Weight for completeness reward",
    )
    parser.add_argument(
        "--w_ordering", type=float, default=1.0, help="Weight for ordering reward"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Max steps (-1 for full epochs)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-7,
        help="Learning rate (very low for RL — 10x below SFT)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (small due to generation overhead)",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=3072)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (very low for RL to prevent mode collapse)",
    )
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--full_finetune", action="store_true", default=False)

    # vLLM
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=False,
        help="Use vLLM for faster generation (needs extra GPU or memory)",
    )

    # Dataset
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit training samples"
    )

    # W&B
    parser.add_argument("--wandb_project", type=str, default="mistral-hackathon")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    # Hub
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--hub_private", action="store_true", default=True)

    # Eval
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Run reward analysis every N steps"
    )
    parser.add_argument("--eval_samples", type=int, default=50)

    args = parser.parse_args()

    print("=" * 60)
    print("  STAGE 3 GRPO: Reinforcement Learning for Tool-Calling")
    print("=" * 60)
    print(f"  Base model:      {args.base_model}")
    print(f"  Generations/prompt: {args.num_generations}")
    print(f"  LR:              {args.lr}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(
        f"  Reward weights:  format={args.w_format}, name={args.w_tool_name}, "
        f"args={args.w_argument}, complete={args.w_completeness}, order={args.w_ordering}"
    )
    print()

    # Lazy import TRL (might need specific version)
    from trl import GRPOConfig, GRPOTrainer

    # ── Load tokenizer ──
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left-padding for generation

    # ── Load dataset ──
    print("\n[2/5] Loading Nemotron workplace_assistant dataset...")
    raw_dataset = load_dataset(
        "nvidia/Nemotron-RL-agent-workplace_assistant", split="train"
    )
    if args.max_samples and len(raw_dataset) > args.max_samples:
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(args.max_samples))
    print(f"  Raw examples: {len(raw_dataset)}")

    grpo_dataset = prepare_grpo_dataset(
        raw_dataset, tokenizer, max_prompt_len=args.max_prompt_len
    )
    print(f"  GRPO dataset: {len(grpo_dataset)} prompts")

    # Also load validation set for periodic eval
    val_raw = None
    try:
        val_raw = load_dataset(
            "nvidia/Nemotron-RL-agent-workplace_assistant", split="validation"
        )
        val_dataset = prepare_grpo_dataset(
            val_raw, tokenizer, max_prompt_len=args.max_prompt_len
        )
        print(f"  Validation: {len(val_dataset)} prompts")
    except:
        val_dataset = None
        print("  No validation split available")

    # ── Setup reward functions ──
    print("\n[3/5] Setting up reward functions...")
    reward_funcs = [
        format_reward,
        tool_name_reward,
        argument_reward,
        completeness_reward,
        ordering_reward,
    ]
    reward_weights = [
        args.w_format,
        args.w_tool_name,
        args.w_argument,
        args.w_completeness,
        args.w_ordering,
    ]
    total_w = sum(reward_weights)
    print(f"  Rewards ({len(reward_funcs)}):")
    for fn, w in zip(reward_funcs, reward_weights):
        print(f"    {fn.__name__:<25s} weight={w:.1f} ({100 * w / total_w:.0f}%)")

    # ── Configure training ──
    print("\n[4/5] Configuring GRPO training...")
    run_name = (
        args.wandb_run_name
        or f"grpo-r{args.lora_r}-lr{args.lr}-g{args.num_generations}"
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_prompt_len,
        temperature=args.temperature,
        # Use vLLM for generation if requested
        use_vllm=args.use_vllm,
    )

    # LoRA config
    peft_config = None
    if not args.full_finetune:
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
            f"  rsLoRA: rank={args.lora_r}, alpha={args.lora_alpha}, "
            f"scaling={args.lora_alpha / args.lora_r**0.5:.2f}"
        )

    # ── W&B ──
    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "stage": 3,
                "method": "GRPO",
                "base_model": args.base_model,
                "num_generations": args.num_generations,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "reward_weights": {
                    fn.__name__: w for fn, w in zip(reward_funcs, reward_weights)
                },
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "dataset_size": len(grpo_dataset),
            },
            tags=["stage3", "grpo", "rl", "tool-calling"],
        )

    # ── Initialize trainer ──
    print("\n[5/5] Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=args.base_model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        processing_class=tokenizer,
        train_dataset=grpo_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Train ──
    print("\n" + "=" * 60)
    print("  Starting GRPO training...")
    print("=" * 60)
    trainer.train()

    # ── Save ──
    print("\n  Saving...")
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

    # ── Final eval ──
    if not args.no_wandb and val_dataset:
        print("\n  Running final GRPO evaluation...")
        log_reward_analysis(
            trainer, val_dataset, tokenizer, n_samples=args.eval_samples
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
            commit_message=f"Stage 3 GRPO: tool-calling RL, rsLoRA r={args.lora_r}, "
            f"g={args.num_generations}, {args.epochs} ep",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{args.hub_repo}")

        if not args.full_finetune:
            lora_repo = args.hub_repo + "-lora-grpo"
            try:
                create_repo(lora_repo, private=args.hub_private, exist_ok=True)
                api.upload_folder(
                    folder_path=args.output_dir,
                    repo_id=lora_repo,
                    commit_message=f"Stage 3 GRPO LoRA: r={args.lora_r}",
                )
                print(f"  ✓ LoRA adapter: https://huggingface.co/{lora_repo}")
            except Exception as e:
                print(f"  LoRA upload: {e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  STAGE 3 GRPO COMPLETE!")
    print("=" * 60)
    print(f"  Base model:       {args.base_model}")
    print(f"  Output:           {args.output_dir}")
    if not args.full_finetune:
        print(f"  Merged:           {merged_dir}")
    print(f"  Generations/prompt: {args.num_generations}")
    print(f"  Dataset size:     {len(grpo_dataset)}")
    print(f"  Reward functions: {len(reward_funcs)}")
    print()
    print("  Full 3-stage pipeline:")
    print("    Stage 1 (SFT):  train_sft.py      → Core tool-calling format")
    print("    Stage 2 (SFT):  train_sft_v2.py   → Multi-dataset broadening")
    print("    Stage 3 (GRPO): train_grpo.py      → RL sharpening ✓")
    print()
    print("  Next: Deploy with vLLM + Chrome extension")
    print("    python -m vllm.entrypoints.openai.api_server \\")
    print(f"        --model {merged_dir} \\")
    print("        --port 8000 \\")
    print("        --enable-auto-tool-choice \\")
    print("        --tool-call-parser mistral")
    print("=" * 60)

    if not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
