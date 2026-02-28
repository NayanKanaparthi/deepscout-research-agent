#!/usr/bin/env python3
"""
Stage 4 QAT: Quantization-Aware Training for NVFP4 deployment.

Uses NVIDIA ModelOpt to insert fake-quantization nodes into the model,
then fine-tunes with the standard HF Trainer so the weights learn to
compensate for NVFP4 rounding/clipping errors.

The output is a quantized checkpoint that can be exported for deployment
on Blackwell GPUs via vLLM or TensorRT-LLM with near-lossless accuracy.

QAT does NOT require Blackwell hardware — it uses simulated (fake)
quantization and can run on any GPU (Ampere, Hopper, etc).

4-stage pipeline:
    Stage 1 (train_sft.py)    → Learn tool-calling format
    Stage 2 (train_sft_v2.py) → Broaden with multi-dataset blend
    Stage 3 (train_grpo.py)   → RL-sharpen with verifiable rewards
    Stage 4 (train_qat.py)    → Quantization-aware training for NVFP4 ← this file

Usage:
    # Basic QAT
    python train_qat.py \
        --base_model ./ministral-3b-agent-grpo-merged \
        --output_dir ./ministral-3b-agent-nvfp4

    # QAT with custom settings
    python train_qat.py \
        --base_model ./ministral-3b-agent-grpo-merged \
        --quant_cfg nvfp4 \
        --epochs 1 \
        --lr 1e-5 \
        --calib_size 256 \
        --wandb_project mistral-hackathon

Dependencies:
    pip install nvidia-modelopt
"""

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ──────────────────────────────────────────────
# 1. QUANTIZATION CONFIG REGISTRY
# ──────────────────────────────────────────────


def get_quant_config(name: str):
    """
    Get ModelOpt quantization config by name.

    Available configs:
      nvfp4          - NVFP4 dynamic block weight & activation (recommended for Blackwell)
      nvfp4_mlp_only - NVFP4 only on MLP layers (lighter, preserves attention precision)
      nvfp4_awq      - NVFP4 with AWQ-Lite calibration
      fp8            - FP8 per-tensor weight & activation (Hopper/Blackwell)
      fp8_pc_pt      - FP8 per-channel weight, per-token activation
    """
    import modelopt.torch.quantization as mtq

    configs = {
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
        "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
        "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        "int8": mtq.INT8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
    }

    if name not in configs:
        raise ValueError(
            f"Unknown quant config: {name}. Available: {list(configs.keys())}"
        )

    return configs[name]


# ──────────────────────────────────────────────
# 2. DATASET PREPARATION
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


def prepare_qat_dataset(raw_dataset, tokenizer, max_seq_length=4096):
    """
    Prepare dataset for QAT training.
    Same format as SFT — full conversation with tool calls.
    """
    examples = []

    for example in raw_dataset:
        params = example["responses_create_params"]
        ground_truth = example["ground_truth"]
        input_messages = params["input"]
        raw_tools = params.get("tools", [])
        cleaned_tools = [strip_nulls(t) for t in raw_tools]

        # Build assistant response with tool calls
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
            messages.append(
                {"role": "assistant", "content": "I'll help you with that."}
            )

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=cleaned_tools if cleaned_tools else None,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            tool_desc = json.dumps(cleaned_tools, indent=2) if cleaned_tools else ""
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
                else:
                    adjusted.append(msg)
            try:
                text = tokenizer.apply_chat_template(
                    adjusted, tokenize=False, add_generation_prompt=False
                )
            except:
                continue

        tokens = tokenizer.encode(text)
        if len(tokens) > max_seq_length:
            continue

        examples.append({"text": text})

    print(f"  Prepared {len(examples)} examples for QAT")
    return Dataset.from_list(examples)


def prepare_calibration_dataloader(
    dataset, tokenizer, calib_size=256, max_seq_length=2048, batch_size=4
):
    """
    Prepare calibration dataloader for PTQ step before QAT.
    ModelOpt needs a forward_loop to calibrate scale factors.
    """
    from torch.utils.data import DataLoader

    # Sample calibration set
    indices = random.sample(range(len(dataset)), min(calib_size, len(dataset)))
    calib_texts = [dataset[i]["text"] for i in indices]

    # Tokenize
    encodings = tokenizer(
        calib_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )

    # Create simple dataset
    class CalibDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }

    calib_dataset = CalibDataset(encodings["input_ids"], encodings["attention_mask"])
    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"  Calibration set: {len(calib_dataset)} samples, {len(calib_dataloader)} batches"
    )
    return calib_dataloader


# ──────────────────────────────────────────────
# 3. QAT CORE
# ──────────────────────────────────────────────


def insert_quantizers(model, quant_cfg_name, calib_dataloader, device):
    """
    Insert fake-quantization nodes into model using ModelOpt.

    This is the key step:
    1. mtq.quantize() inserts quantize/dequantize ops into the forward pass
    2. These ops simulate NVFP4 rounding/clipping in BF16
    3. Backward pass uses straight-through estimation (STE)
    4. The model learns to compensate for quantization errors

    Returns the quantized model (modified in-place).
    """
    import modelopt.torch.quantization as mtq

    quant_cfg = get_quant_config(quant_cfg_name)

    print(f"\n  Inserting fake quantizers ({quant_cfg_name})...")
    print(
        f"  This calibrates activation scales using {len(calib_dataloader)} batches..."
    )

    # Define calibration forward loop
    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
                if (i + 1) % 10 == 0:
                    print(f"    Calibration batch {i + 1}/{len(calib_dataloader)}")

    # Insert quantizers and calibrate
    model = mtq.quantize(model, quant_cfg, forward_loop)

    # Count quantized modules
    n_quantized = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer") or hasattr(module, "input_quantizer"):
            n_quantized += 1

    print(f"  ✓ Inserted quantizers into {n_quantized} modules")
    print(
        f"  Model is now in fake-quantized mode (BF16 with simulated {quant_cfg_name})"
    )

    return model


def save_modelopt_state(model, save_dir):
    """Save ModelOpt quantizer states for resuming QAT."""
    import modelopt.torch.opt as mto

    os.makedirs(save_dir, exist_ok=True)
    state_path = os.path.join(save_dir, "modelopt_quantizer_states.pt")
    torch.save(mto.modelopt_state(model), state_path)
    print(f"  Saved ModelOpt state to: {state_path}")
    return state_path


def export_quantized_checkpoint(model, tokenizer, export_dir):
    """
    Export the QAT model to a quantized HuggingFace checkpoint.

    The exported checkpoint contains:
      - Quantized weights
      - Scale factors
      - hf_quant_config.json (detected by vLLM/TRT-LLM)

    Can be directly loaded by:
      vllm serve <export_dir> --quantization modelopt_fp4
    """
    from modelopt.torch.export import export_hf_checkpoint

    os.makedirs(export_dir, exist_ok=True)
    print(f"\n  Exporting quantized checkpoint to: {export_dir}")

    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir)

    # Also save tokenizer for convenience
    tokenizer.save_pretrained(export_dir)

    # Verify export
    config_path = os.path.join(export_dir, "hf_quant_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            quant_config = json.load(f)
        algo = quant_config.get("quantization", {}).get("quant_algo", "unknown")
        print(f"  ✓ Exported with quant_algo: {algo}")
    else:
        # Check config.json for quantization_config
        model_config_path = os.path.join(export_dir, "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path) as f:
                model_config = json.load(f)
            qc = model_config.get("quantization_config", {})
            if qc:
                print(
                    f"  ✓ Quantization config in config.json: {qc.get('quant_algo', 'set')}"
                )

    print("  ✓ Checkpoint ready for deployment")
    return export_dir


# ──────────────────────────────────────────────
# 4. EVAL (lightweight, reuses Stage 1-3 pattern)
# ──────────────────────────────────────────────


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model output (simplified)."""
    import re

    tool_calls = []

    for marker in ["[TOOL_CALLS]", "[TOOL_CALL]"]:
        if marker in text:
            json_part = text.split(marker, 1)[1].strip()
            try:
                parsed = json.loads(json_part.split("\n")[0].strip())
                items = parsed if isinstance(parsed, list) else [parsed]
                for tc in items:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    tool_calls.append({"name": tc.get("name", ""), "arguments": args})
                return tool_calls
            except json.JSONDecodeError:
                pass

    # Fallback: find JSON objects with "name"
    idx = 0
    while idx < len(text):
        pos = text.find('"name"', idx)
        if pos == -1:
            break
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


@torch.no_grad()
def run_qat_eval(model, tokenizer, raw_dataset, max_examples=50, desc="qat_eval"):
    """Quick eval to verify QAT hasn't degraded tool-calling ability."""
    model.eval()
    examples = list(raw_dataset)[:max_examples]

    correct_names = 0
    correct_full = 0
    parse_ok = 0
    total = len(examples)

    print(f"\n  Running {desc} on {total} examples...")

    for i, example in enumerate(examples):
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
            continue

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=3072
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.01, do_sample=False
        )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )

        predicted = parse_tool_calls(generated)
        gt_names = [gt["name"] for gt in ground_truth]
        pred_names = [tc["name"] for tc in predicted]

        if len(predicted) > 0 or len(gt_names) == 0:
            parse_ok += 1
        if pred_names == gt_names:
            correct_names += 1

        # Check arguments too
        all_args_match = True
        for j, gt in enumerate(ground_truth):
            if j >= len(predicted) or predicted[j]["name"] != gt["name"]:
                all_args_match = False
                break
            gt_args = gt["arguments"]
            if isinstance(gt_args, str):
                try:
                    gt_args = json.loads(gt_args)
                except:
                    gt_args = {}
            pred_args = predicted[j].get("arguments", {})
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except:
                    pred_args = {}
            for k, v in gt_args.items():
                if (
                    k not in pred_args
                    or str(pred_args[k]).strip().lower() != str(v).strip().lower()
                ):
                    all_args_match = False
                    break
            if not all_args_match:
                break
        if all_args_match and len(predicted) == len(ground_truth):
            correct_full += 1

    metrics = {
        f"{desc}/parse_success": parse_ok / max(total, 1),
        f"{desc}/tool_name_exact_match": correct_names / max(total, 1),
        f"{desc}/full_exact_match": correct_full / max(total, 1),
        f"{desc}/n_examples": total,
    }

    print(f"  ── {desc} Results ──")
    print(f"    parse_success:       {metrics[f'{desc}/parse_success']:.1%}")
    print(f"    tool_name_match:     {metrics[f'{desc}/tool_name_exact_match']:.1%}")
    print(f"    full_exact_match:    {metrics[f'{desc}/full_exact_match']:.1%}")

    model.train()
    return metrics


# ──────────────────────────────────────────────
# 5. W&B CALLBACK
# ──────────────────────────────────────────────


class QATCallback(TrainerCallback):
    """W&B callback for QAT monitoring."""

    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb

        if logs is None:
            return
        enhanced = {}
        if "loss" in logs:
            enhanced["qat/perplexity"] = math.exp(min(logs["loss"], 20))
        if "learning_rate" in logs:
            enhanced["qat/lr"] = logs["learning_rate"]
        if torch.cuda.is_available():
            enhanced["qat/vram_gb"] = torch.cuda.memory_allocated() / 1e9
        if self.start_time:
            enhanced["qat/elapsed_min"] = (time.time() - self.start_time) / 60
        if enhanced:
            wandb.log(enhanced, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        import wandb

        if self.start_time:
            wandb.log({"qat/total_minutes": (time.time() - self.start_time) / 60})


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 QAT: NVFP4 Quantization-Aware Training"
    )

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="./ministral-3b-agent-grpo-merged",
        help="Input model (Stage 3 output or any HF model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ministral-3b-agent-nvfp4-qat",
        help="QAT training checkpoint dir",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="./ministral-3b-agent-nvfp4",
        help="Final exported quantized checkpoint for deployment",
    )

    # Quantization
    parser.add_argument(
        "--quant_cfg",
        type=str,
        default="nvfp4",
        choices=[
            "nvfp4",
            "nvfp4_mlp_only",
            "nvfp4_awq",
            "fp8",
            "fp8_pc_pt",
            "int8",
            "int4_awq",
        ],
        help="Quantization config (default: nvfp4 for Blackwell)",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=256,
        help="Number of calibration samples for scale factor estimation",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=1, help="QAT epochs (1 is typically sufficient)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (very low — just adapting to quantization noise)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Eval
    parser.add_argument("--eval_max_examples", type=int, default=50)
    parser.add_argument("--skip_eval", action="store_true", default=False)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="mistral-hackathon")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    # Hub
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--hub_private", action="store_true", default=True)

    # Skip phases
    parser.add_argument(
        "--ptq_only",
        action="store_true",
        default=False,
        help="Only do PTQ calibration + export, skip QAT training",
    )
    parser.add_argument(
        "--resume_from_quantized",
        type=str,
        default=None,
        help="Resume QAT from a saved modelopt state",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  STAGE 4 QAT: Quantization-Aware Training for NVFP4")
    print("=" * 60)
    print(f"  Base model:    {args.base_model}")
    print(f"  Quant config:  {args.quant_cfg}")
    print(f"  Calib samples: {args.calib_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  PTQ only:      {args.ptq_only}")
    print()

    # ── Import ModelOpt ──
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq

        print(
            f"  ModelOpt version: {mtq.__version__ if hasattr(mtq, '__version__') else 'installed'}"
        )
    except ImportError:
        print("ERROR: nvidia-modelopt not installed.")
        print("  Install with: pip install nvidia-modelopt")
        print("  Or use the ModelOpt docker image.")
        return

    # ── Load tokenizer ──
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model in BF16 (full precision for QAT) ──
    print("\n[2/7] Loading model in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"  Model: {model.config.architectures}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load dataset ──
    print("\n[3/7] Loading dataset...")
    raw_train = load_dataset(
        "nvidia/Nemotron-RL-agent-workplace_assistant", split="train"
    )
    raw_val = None
    try:
        raw_val = load_dataset(
            "nvidia/Nemotron-RL-agent-workplace_assistant", split="validation"
        )
    except:
        pass

    train_dataset = prepare_qat_dataset(
        raw_train, tokenizer, max_seq_length=args.max_seq_length
    )

    # ── Prepare calibration data ──
    print("\n[4/7] Preparing calibration data...")
    calib_dataloader = prepare_calibration_dataloader(
        train_dataset,
        tokenizer,
        calib_size=args.calib_size,
        max_seq_length=min(args.max_seq_length, 2048),
        batch_size=args.batch_size,
    )

    # ── Pre-QAT eval (BF16 baseline) ──
    pre_metrics = None
    if raw_val and not args.skip_eval:
        print("\n[4.5/7] Pre-QAT eval (BF16 baseline)...")
        pre_metrics = run_qat_eval(
            model,
            tokenizer,
            raw_val,
            max_examples=args.eval_max_examples,
            desc="pre_qat",
        )

    # ── Insert fake quantizers (PTQ calibration) ──
    print("\n[5/7] Inserting fake quantizers...")

    if args.resume_from_quantized:
        # Resume from saved modelopt state
        print(f"  Resuming from: {args.resume_from_quantized}")
        state = torch.load(args.resume_from_quantized, weights_only=False)
        mto.restore_from_modelopt_state(model, state)
        print("  ✓ Restored quantizer states")
    else:
        device = next(model.parameters()).device
        model = insert_quantizers(model, args.quant_cfg, calib_dataloader, device)

        # Save quantizer states for potential resume
        save_modelopt_state(model, args.output_dir)

    # ── PTQ-only mode: just export and exit ──
    if args.ptq_only:
        print("\n  PTQ-only mode: skipping QAT training, exporting directly...")

        if raw_val and not args.skip_eval:
            print("\n  Post-PTQ eval...")
            ptq_metrics = run_qat_eval(
                model,
                tokenizer,
                raw_val,
                max_examples=args.eval_max_examples,
                desc="ptq_eval",
            )
            if pre_metrics:
                print("\n  ── PTQ Impact ──")
                for key in [
                    "parse_success",
                    "tool_name_exact_match",
                    "full_exact_match",
                ]:
                    pre = pre_metrics.get(f"pre_qat/{key}", 0)
                    post = ptq_metrics.get(f"ptq_eval/{key}", 0)
                    print(
                        f"    {key:<30s}  {pre:.1%} → {post:.1%}  ({post - pre:+.1%})"
                    )

        export_quantized_checkpoint(model, tokenizer, args.export_dir)
        print(f"\n  ✓ PTQ checkpoint exported to: {args.export_dir}")
        print(f"  Deploy: vllm serve {args.export_dir} --quantization modelopt_fp4")
        return

    # ── QAT Training ──
    print("\n[6/7] QAT Training...")

    run_name = (
        args.wandb_run_name or f"qat-{args.quant_cfg}-lr{args.lr}-ep{args.epochs}"
    )

    # Split dataset
    split = train_dataset.train_test_split(test_size=0.05, seed=42)
    qat_train = split["train"]
    qat_eval = split["test"]
    print(f"  Train: {len(qat_train)}, Eval: {len(qat_eval)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
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

    # W&B
    qat_callback = None
    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "stage": 4,
                "method": "QAT",
                "base_model": args.base_model,
                "quant_cfg": args.quant_cfg,
                "lr": args.lr,
                "epochs": args.epochs,
                "calib_size": args.calib_size,
                "dataset_size": len(qat_train),
            },
            tags=["stage4", "qat", args.quant_cfg],
        )
        qat_callback = QATCallback()

    # Data collator for standard LM training
    def data_collator(examples):
        texts = [ex["text"] for ex in examples]
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        batch["labels"] = batch["input_ids"].clone()
        # Mask padding tokens in labels
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=qat_train,
        eval_dataset=qat_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[qat_callback] if qat_callback else [],
    )

    # Train!
    trainer.train()

    # Save QAT weights + modelopt state
    print("\n  Saving QAT checkpoint...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_modelopt_state(model, args.output_dir)

    # ── Post-QAT eval ──
    post_metrics = None
    if raw_val and not args.skip_eval:
        print("\n  Post-QAT eval...")
        post_metrics = run_qat_eval(
            model,
            tokenizer,
            raw_val,
            max_examples=args.eval_max_examples,
            desc="post_qat",
        )

        if pre_metrics:
            print("\n  ══════════════════════════════════════════════")
            print("  ██ QAT: BF16 vs NVFP4 COMPARISON ██")
            print("  ══════════════════════════════════════════════")
            comparison_data = []
            for key in ["parse_success", "tool_name_exact_match", "full_exact_match"]:
                pre = pre_metrics.get(f"pre_qat/{key}", 0)
                post = post_metrics.get(f"post_qat/{key}", 0)
                delta = post - pre
                print(f"    {key:<30s}  {pre:.1%} → {post:.1%}  ({delta:+.1%})")
                comparison_data.append(
                    [
                        key,
                        round(pre * 100, 1),
                        round(post * 100, 1),
                        round(delta * 100, 1),
                    ]
                )

            if not args.no_wandb:
                import wandb

                wandb.log(
                    {
                        "qat_comparison": wandb.Table(
                            columns=["metric", "bf16_%", "nvfp4_%", "delta_%"],
                            data=comparison_data,
                        )
                    }
                )

    # ── Export quantized checkpoint ──
    print("\n[7/7] Exporting quantized checkpoint...")
    export_quantized_checkpoint(model, tokenizer, args.export_dir)

    # ── Hub upload ──
    if args.hub_repo:
        from huggingface_hub import HfApi, create_repo

        print(f"\n  Pushing to HuggingFace Hub: {args.hub_repo}")
        api = HfApi()
        try:
            create_repo(args.hub_repo, private=args.hub_private, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        api.upload_folder(
            folder_path=args.export_dir,
            repo_id=args.hub_repo,
            commit_message=f"Stage 4 QAT: {args.quant_cfg}, lr={args.lr}, {args.epochs} ep",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{args.hub_repo}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  STAGE 4 QAT COMPLETE!")
    print("=" * 60)
    print(f"  QAT checkpoint: {args.output_dir}")
    print(f"  Exported model: {args.export_dir}")
    print(f"  Quant format:   {args.quant_cfg}")
    if args.hub_repo:
        print(f"  Hub:            https://huggingface.co/{args.hub_repo}")
    print()
    print("  Full 4-stage pipeline:")
    print("    Stage 1 (SFT):  train_sft.py      → Core tool-calling format")
    print("    Stage 2 (SFT):  train_sft_v2.py   → Multi-dataset broadening")
    print("    Stage 3 (GRPO): train_grpo.py      → RL sharpening")
    print("    Stage 4 (QAT):  train_qat.py       → NVFP4 quantization ✓")
    print()
    print("  Deploy on Blackwell:")
    print(f"    vllm serve {args.export_dir} \\")
    print("        --quantization modelopt_fp4 \\")
    print("        --enable-auto-tool-choice \\")
    print("        --tool-call-parser mistral \\")
    print("        --port 8000")
    print()
    print("  Deploy on Hopper (if using fp8 config):")
    print(f"    vllm serve {args.export_dir} \\")
    print("        --quantization modelopt \\")
    print("        --port 8000")
    print()
    print("  Memory savings (approx for 3B model):")
    print("    BF16:   ~6.0 GB")
    print("    FP8:    ~3.0 GB")
    print("    NVFP4:  ~1.7 GB  (3.5x reduction)")
    print("=" * 60)

    if not args.no_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
