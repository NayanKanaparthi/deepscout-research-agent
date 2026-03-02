#!/usr/bin/env python3
"""
SFT Fine-tuning script for Ministral-3B-Instruct: MCQA -> Search Query generation.
JSON-in / JSON-out SFT with LoRA/QLoRA, completion-only loss masking.

Matches system_prompt.md format:
  Input:  {"user_input": "<MCQA question>"}
  Output: {"query": "<search query>"}

Usage:
    python search_query_agent/train.py
    python search_query_agent/train.py --data_path path/to/data.jsonl --epochs 3 --lr 1e-4
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

def _load_model(model_name, bnb_config):
    """Load model. For multimodal models like Mistral3, extracts the language model as a CausalLM."""
    load_kwargs = dict(
        quantization_config=bnb_config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        return model
    except (ValueError, KeyError) as e:
        print(f"  AutoModelForCausalLM failed: {e}")
        print("  Loading VLM and extracting text-only CausalLM backbone...")
        from transformers import AutoModelForImageTextToText, MistralForCausalLM

        full_model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)

        text_config = full_model.config.text_config
        text_config.tie_word_embeddings = False

        with torch.device("meta"):
            model = MistralForCausalLM(text_config)

        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        model.generation_config = full_model.generation_config

        del full_model
        torch.cuda.empty_cache()
        print("  Extracted MistralForCausalLM from VLM successfully.")
        return model

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_SYSTEM_PROMPT_PATH = SCRIPT_DIR / "system_prompt.md"
DEFAULT_DATA_PATH = SCRIPT_DIR / "search_query_data.jsonl" / "search_query_data.jsonl"


# ──────────────────────────────────────────────
# 1. DATA LOADING & FORMATTING
# ──────────────────────────────────────────────


def load_system_prompt(path):
    """Load system prompt from markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_data_path(data_path):
    """Resolve the data path, handling the nested directory case."""
    p = Path(data_path)
    if p.is_file():
        return p
    # Handle case where path is a directory containing a file with the same name
    if p.is_dir():
        nested = p / p.name
        if nested.is_file():
            return nested
        jsonl_files = list(p.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
    raise FileNotFoundError(f"Could not find data file at: {data_path}")


def load_jsonl_dataset(data_path, val_split=0.1, seed=42):
    """
    Load a JSONL file with search query training data.
    Supports two formats:
      - {"input": ..., "output": ...}
      - {"user_prompt": ..., "output": "{\"query\": ...}", "system_prompt": ...}

    The output field is kept as-is to match system_prompt.md JSON format.
    If the output is a plain string (not JSON), it gets wrapped as {"query":"..."}.
    Returns (train_dataset, val_dataset) as HuggingFace Datasets.
    """
    resolved = resolve_data_path(data_path)
    print(f"  Resolved data path: {resolved}")

    records = []
    with open(resolved, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                normalized = {}

                # Handle input field: "input" or "user_prompt"
                if "input" in record:
                    normalized["input"] = record["input"]
                elif "user_prompt" in record:
                    normalized["input"] = record["user_prompt"]
                else:
                    raise ValueError("Missing 'input' or 'user_prompt' key")

                # Handle output field — keep as JSON to match system_prompt.md
                if "output" not in record:
                    raise ValueError("Missing 'output' key")
                output_raw = record["output"]
                if isinstance(output_raw, str):
                    try:
                        parsed = json.loads(output_raw)
                        if isinstance(parsed, dict) and "query" in parsed:
                            normalized["output"] = output_raw.strip()
                        else:
                            normalized["output"] = json.dumps({"query": output_raw})
                    except json.JSONDecodeError:
                        normalized["output"] = json.dumps({"query": output_raw})
                else:
                    normalized["output"] = json.dumps({"query": str(output_raw)})

                records.append(normalized)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  Warning: Skipping line {line_num}: {e}")

    if not records:
        raise ValueError(f"No valid records found in {resolved}")

    print(f"  Loaded {len(records)} records from {resolved}")
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    print(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]


def format_example_to_messages(example, system_prompt):
    """
    Convert a normalized record into chat messages matching system_prompt.md:
      - User input is wrapped as JSON: {"user_input": "<question text>"}
      - Assistant output is JSON: {"query": "..."}
      - System prompt always comes from system_prompt.md
    """
    user_content = json.dumps({"user_input": example["input"]})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"messages": messages}


def format_example_to_prompt_completion(example, system_prompt, tokenizer):
    """
    Convert a record into prompt-completion format for SFTTrainer.
    The prompt includes system + user with generation prompt appended by the chat template.
    The completion is the assistant's JSON response.
    """
    user_content = json.dumps({"user_input": example["input"]})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "completion": example["output"]}


def format_dataset(dataset, system_prompt, tokenizer=None):
    """Apply formatting to entire dataset. Uses prompt-completion format if tokenizer is provided."""
    if tokenizer is not None:
        return dataset.map(
            lambda x: format_example_to_prompt_completion(x, system_prompt, tokenizer),
            desc="Formatting to prompt-completion",
        )
    return dataset.map(
        lambda x: format_example_to_messages(x, system_prompt),
        desc="Formatting to chat messages",
    )


# ──────────────────────────────────────────────
# 2. TOKENIZATION / CHAT TEMPLATE
# ──────────────────────────────────────────────


def apply_chat_template(example, tokenizer):
    """
    Apply tokenizer's chat template to format messages.
    No tools parameter needed -- simple system/user/assistant messages.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ──────────────────────────────────────────────
# 3. EVALUATION
# ──────────────────────────────────────────────


def _extract_query(text):
    """Extract the plain query string from a JSON {"query":"..."} response, or return as-is."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "query" in parsed:
            return parsed["query"]
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def compute_query_metrics(predicted, reference):
    """
    Compute text-level metrics for a single predicted vs reference search query.
    """
    pred_clean = predicted.strip().lower()
    ref_clean = reference.strip().lower()

    # Exact match
    exact_match = int(pred_clean == ref_clean)

    # Token-level overlap (word-level F1)
    pred_tokens = set(pred_clean.split())
    ref_tokens = set(ref_clean.split())

    if not ref_tokens:
        token_precision = 1.0 if not pred_tokens else 0.0
        token_recall = 1.0
        token_f1 = 1.0 if not pred_tokens else 0.0
    elif not pred_tokens:
        token_precision = 0.0
        token_recall = 0.0
        token_f1 = 0.0
    else:
        common = pred_tokens & ref_tokens
        token_precision = len(common) / len(pred_tokens)
        token_recall = len(common) / len(ref_tokens)
        token_f1 = (
            2 * token_precision * token_recall / (token_precision + token_recall)
            if (token_precision + token_recall) > 0
            else 0.0
        )

    # Query length stats
    pred_word_count = len(predicted.strip().split())
    ref_word_count = len(reference.strip().split())
    length_diff = pred_word_count - ref_word_count

    # Length compliance (5-12 words as per system prompt)
    in_length_range = int(5 <= pred_word_count <= 12)

    return {
        "exact_match": exact_match,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "pred_word_count": pred_word_count,
        "ref_word_count": ref_word_count,
        "length_diff": length_diff,
        "in_length_range": in_length_range,
    }


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    eval_records,
    system_prompt,
    max_examples=None,
    max_new_tokens=64,
    temperature=0.01,
    desc="eval",
):
    """
    Run generation-based evaluation.
    eval_records: list of {"input": ..., "output": ...} dicts.
    Returns (aggregate_metrics, per_example_results).
    """
    model.eval()

    examples = eval_records
    if max_examples:
        examples = examples[:max_examples]

    all_metrics = defaultdict(list)
    per_example_results = []

    print(f"\n  Running {desc} on {len(examples)} examples...")

    for i, example in enumerate(examples):
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(examples)}...")

        user_content = json.dumps({"user_input": example["input"]})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        pred_query = _extract_query(generated_text)
        ref_query = _extract_query(example["output"])

        example_metrics = compute_query_metrics(pred_query, ref_query)
        for k, v in example_metrics.items():
            all_metrics[k].append(v)

        per_example_results.append(
            {
                "idx": i,
                "input": example["input"][:200],
                "reference": ref_query,
                "predicted": generated_text,
                "exact_match": example_metrics["exact_match"],
                "token_f1": example_metrics["token_f1"],
                "pred_words": example_metrics["pred_word_count"],
                "in_range": example_metrics["in_length_range"],
            }
        )

    # Aggregate
    agg = {}
    for k, values in all_metrics.items():
        agg[f"{desc}/{k}"] = np.mean(values)
    agg[f"{desc}/n_examples"] = len(examples)

    print(f"\n  -- {desc} Results ({len(examples)} examples) --")
    print(f"    Exact match:       {agg[f'{desc}/exact_match']:.1%}")
    print(f"    Token F1:          {agg[f'{desc}/token_f1']:.1%}")
    print(f"    Token precision:   {agg[f'{desc}/token_precision']:.1%}")
    print(f"    Token recall:      {agg[f'{desc}/token_recall']:.1%}")
    print(f"    In length range:   {agg[f'{desc}/in_length_range']:.1%}")
    print(f"    Avg pred words:    {agg[f'{desc}/pred_word_count']:.1f}")
    print(f"    Avg ref words:     {agg[f'{desc}/ref_word_count']:.1f}")

    return agg, per_example_results


# ──────────────────────────────────────────────
# 4. WEIGHTS & BIASES LOGGING
# ──────────────────────────────────────────────


def log_eval_to_wandb(metrics, per_example, phase):
    """Log eval results to W&B with a per-example table."""
    import wandb

    wandb.log(metrics)

    table = wandb.Table(
        columns=[
            "idx",
            "input",
            "reference",
            "predicted",
            "exact_match",
            "token_f1",
            "pred_words",
            "in_range",
        ]
    )
    for r in per_example:
        table.add_data(
            r["idx"],
            r["input"],
            r["reference"],
            r["predicted"],
            r["exact_match"],
            r["token_f1"],
            r["pred_words"],
            r["in_range"],
        )
    wandb.log({f"{phase}/per_example_results": table})


def log_dataset_analysis(train_dataset, eval_dataset, tokenizer):
    """Log dataset statistics to W&B."""
    import wandb

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
            "dataset/train_total_tokens": sum(train_lengths),
        }
    )


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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        wandb.log(
            {
                "model/total_params_M": total_params / 1e6,
                "model/trainable_params_M": trainable_params / 1e6,
                "model/frozen_params_M": (total_params - trainable_params) / 1e6,
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

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                wandb.log(
                    {
                        f"system/gpu_{i}_name": props.name,
                        f"system/gpu_{i}_vram_GB": props.total_memory / 1e9,
                    }
                )

    def on_step_end(self, args, state, control, **kwargs):
        import wandb

        now = time.time()
        step_time = now - self.last_step_time
        self.last_step_time = now
        self.step_times.append(step_time)

        step = state.global_step
        logs = {}

        # Throughput metrics
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

        # GPU memory tracking
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                logs[f"gpu/gpu_{i}_allocated_GB"] = mem_alloc
                logs[f"gpu/gpu_{i}_reserved_GB"] = mem_reserved
                logs[f"gpu/gpu_{i}_utilization_pct"] = 100 * mem_alloc / mem_total

        # Gradient statistics (every N steps)
        if step % self.log_grad_every_n == 0:
            all_grad_norms = []
            grad_norms = {}
            lora_A_norms = {}
            lora_B_norms = {}

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    all_grad_norms.append(grad_norm)

                    for module_type in [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]:
                        if module_type in name:
                            key = f"gradients/{module_type}"
                            if key not in grad_norms:
                                grad_norms[key] = []
                            grad_norms[key].append(grad_norm)

                    if "lora_A" in name:
                        for mod in ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]:
                            if mod in name:
                                lora_A_norms[mod] = param.data.norm(2).item()
                    elif "lora_B" in name:
                        for mod in ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]:
                            if mod in name:
                                lora_B_norms[mod] = param.data.norm(2).item()

            if all_grad_norms:
                logs["gradients/global_norm"] = np.sqrt(
                    sum(g**2 for g in all_grad_norms)
                )
                logs["gradients/mean_norm"] = np.mean(all_grad_norms)
                logs["gradients/max_norm"] = max(all_grad_norms)
                logs["gradients/min_norm"] = min(all_grad_norms)

            for key, norms in grad_norms.items():
                logs[f"{key}_mean"] = np.mean(norms)
                logs[f"{key}_max"] = max(norms)

            for mod in lora_A_norms:
                if mod in lora_B_norms:
                    logs[f"lora_weights/{mod}_A_norm"] = lora_A_norms[mod]
                    logs[f"lora_weights/{mod}_B_norm"] = lora_B_norms[mod]

        wandb.log(logs, step=step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import wandb

        if metrics:
            eval_logs = {}
            for k, v in metrics.items():
                clean_key = k.replace("eval_", "eval/")
                eval_logs[clean_key] = v

            if "eval_loss" in metrics:
                eval_logs["eval/perplexity"] = math.exp(
                    min(metrics["eval_loss"], 20)
                )

            wandb.log(eval_logs, step=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb

        if logs is None:
            return

        enhanced = {}
        if "loss" in logs:
            enhanced["train/perplexity"] = math.exp(min(logs["loss"], 20))
        if "learning_rate" in logs:
            enhanced["train/lr_log10"] = math.log10(max(logs["learning_rate"], 1e-10))
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


def init_wandb(args, model, tokenizer, train_dataset, eval_dataset):
    """Initialize W&B with config and dataset analysis."""
    import wandb

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        "model_name": args.model_name,
        "model_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "trainable_pct": 100 * trainable_params / total_params,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "rslora": True,
        "rslora_scaling": args.lora_alpha / (args.lora_r**0.5),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "data_path": args.data_path,
        "val_split": args.val_split,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "task": "mcqa_to_search_query",
    }

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"search-query-r{args.lora_r}-e{args.epochs}",
        config=config,
        tags=[
            "sft",
            "ministral",
            "search-query",
            "mcqa",
            f"lora-r{args.lora_r}",
            "rslora",
            "hackathon",
        ],
    )

    # Define metrics so W&B shows proper loss curves and charts
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/global_step")
    wandb.define_metric("eval/*", step_metric="eval/global_step")
    wandb.define_metric("loss", step_metric="train/global_step")
    wandb.define_metric("eval_loss", step_metric="eval/global_step")
    wandb.define_metric("learning_rate", step_metric="train/global_step")

    print("\n  Logging dataset analysis to W&B...")
    log_dataset_analysis(train_dataset, eval_dataset, tokenizer)

    return wandb.run


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SFT training: MCQA -> Search Query (Ministral-3B)"
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to JSONL file with {input, output} records",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to system_prompt.md",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Ministral-3-3B-Instruct-2512-BF16",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./search-query-agent-sft"
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision / efficiency
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="search-query-agent")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False)

    # Eval
    parser.add_argument("--eval_max_examples", type=int, default=100)
    parser.add_argument("--skip_eval", action="store_true", default=False)

    # Hub
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--hub_private", action="store_true", default=False)

    args = parser.parse_args()

    print("=" * 60)
    print("Search Query Agent SFT Training")
    print("=" * 60)
    print(f"Model:    {args.model_name}")
    print(f"Data:     {args.data_path}")
    print(f"Val split:{args.val_split}")
    print(f"LoRA:     r={args.lora_r}, alpha={args.lora_alpha}, rsLoRA")
    print(f"Epochs:   {args.epochs}")
    print(f"LR:       {args.lr}")
    print(f"Batch:    {args.batch_size} x {args.gradient_accumulation_steps} accum")
    print(f"Max Seq:  {args.max_seq_length}")
    print("=" * 60)

    # ── [1/5] Load tokenizer ──
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── [2/5] Load and format dataset ──
    print("\n[2/5] Loading and formatting dataset...")
    system_prompt = load_system_prompt(args.system_prompt_path)
    print(f"  System prompt loaded: {len(system_prompt)} chars")

    raw_train, raw_val = load_jsonl_dataset(args.data_path, args.val_split, args.seed)
    # Keep raw records for eval (need input/output pairs)
    raw_val_records = list(raw_val)

    train_dataset = format_dataset(raw_train, system_prompt)
    eval_dataset = format_dataset(raw_val, system_prompt)

    # Apply chat template to create "text" field
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        desc="Applying chat template (train)",
    )
    eval_dataset = eval_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        desc="Applying chat template (eval)",
    )

    # Print a sample
    print("\n-- Sample formatted text (first 500 chars) --")
    print(train_dataset[0]["text"][:500])
    print("...")

    # Quick stats on token lengths
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

    # ── [3/5] Load model ──
    print("\n[3/5] Loading model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = _load_model(args.model_name, bnb_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Setup LoRA ──
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
        use_rslora=True,
    )
    scaling = args.lora_alpha / (args.lora_r**0.5)
    print(
        f"  rsLoRA enabled: scaling = alpha/sqrt(r) = {args.lora_alpha}/{args.lora_r**0.5:.1f} = {scaling:.2f}"
    )

    # ── [4/5] Training arguments ──
    print("\n[4/5] Setting up training...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
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
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none" if args.no_wandb else "wandb",
        packing=False,
    )

    # ── Initialize W&B ──
    wandb_callback = None
    if not args.no_wandb:
        print("\n[4.5/5] Initializing Weights & Biases...")
        init_wandb(args, model, tokenizer, train_dataset, eval_dataset)
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
        callbacks=callbacks,
    )

    # Print trainable params
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
            raw_val_records,
            system_prompt,
            max_examples=args.eval_max_examples,
            desc="pre_train_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(pre_eval_metrics, pre_eval_examples, "pre_train_eval")

    # ── [5/5] Train ──
    print("\n[5/5] Starting training...")
    trainer.train()

    # ── Save ──
    print("\nSaving LoRA adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")

    # ── Post-training eval ──
    post_eval_metrics = None
    if not args.skip_eval:
        print("\n[POST] Running POST-TRAINING evaluation...")
        post_eval_metrics, post_eval_examples = run_eval(
            trainer.model,
            tokenizer,
            raw_val_records,
            system_prompt,
            max_examples=args.eval_max_examples,
            desc="post_train_eval",
        )
        if not args.no_wandb:
            log_eval_to_wandb(
                post_eval_metrics, post_eval_examples, "post_train_eval"
            )

        # Comparison summary
        if pre_eval_metrics and post_eval_metrics:
            print("\n  ==============================================")
            print("  == PRE vs POST TRAINING COMPARISON ==")
            print("  ==============================================")

            comparison_data = []
            key_metrics = [
                "exact_match",
                "token_f1",
                "token_precision",
                "token_recall",
                "in_length_range",
            ]
            for metric in key_metrics:
                pre_val = pre_eval_metrics.get(f"pre_train_eval/{metric}", 0)
                post_val = post_eval_metrics.get(f"post_train_eval/{metric}", 0)
                delta = post_val - pre_val
                delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
                print(
                    f"    {metric:<25s}  {pre_val:.1%} -> {post_val:.1%}  ({delta_str})"
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

        try:
            create_repo(args.hub_repo, private=args.hub_private, exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

        # Fix adapter metadata so HF validation passes (base_model must be non-empty)
        upload_dir = os.path.abspath(args.output_dir)
        base_model_name = args.model_name or "mistralai/Ministral-3-14B-Instruct-2512-BF16"

        adapter_config_path = os.path.join(upload_dir, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            adapter_config["base_model_name_or_path"] = base_model_name
            with open(adapter_config_path, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2)

        # README with valid YAML: base_model must be non-empty (quoted for safety)
        readme_path = os.path.join(upload_dir, "README.md")
        readme_content = f"""---
license: apache-2.0
base_model: "{base_model_name}"
tags:
  - lora
  - search-query
  - ministral
---

# Search Query Agent (LoRA)

LoRA adapter for `{base_model_name}` fine-tuned for MCQA to search query generation.

## Usage

Load with PEFT:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base_model_name}")
model = PeftModel.from_pretrained(base, "{args.hub_repo}")
```
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"  Uploading LoRA adapter from: {upload_dir}")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=args.hub_repo,
            commit_message=f"SFT search query agent | rsLoRA r={args.lora_r} | {args.epochs} epochs",
        )
        print(f"  Uploaded to https://huggingface.co/{args.hub_repo}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter: {args.output_dir}")
    if args.hub_repo:
        print(f"Hub: https://huggingface.co/{args.hub_repo}")
    print("=" * 60)

    # ── Finalize W&B ──
    if not args.no_wandb:
        try:
            import wandb

            print("\nLogging model artifact to W&B...")
            artifact_dir = args.output_dir
            model_artifact = wandb.Artifact(
                name="search-query-agent-sft",
                type="model",
                description=f"SFT search query agent | rsLoRA r={args.lora_r} | {args.epochs} epochs",
                metadata={
                    "base_model": args.model_name,
                    "data_path": args.data_path,
                    "lora_r": args.lora_r,
                    "epochs": args.epochs,
                    "task": "mcqa_to_search_query",
                    "pre_train_exact_match": pre_eval_metrics.get(
                        "pre_train_eval/exact_match"
                    )
                    if pre_eval_metrics
                    else None,
                    "post_train_exact_match": post_eval_metrics.get(
                        "post_train_eval/exact_match"
                    )
                    if post_eval_metrics
                    else None,
                },
            )
            model_artifact.add_dir(artifact_dir)
            wandb.log_artifact(model_artifact)
            wandb.finish()
            print("  W&B run finalized")
        except Exception as e:
            print(f"  W&B finalize skipped: {e}")


if __name__ == "__main__":
    main()
