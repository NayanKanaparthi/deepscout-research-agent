# Ministral-3B Agent SFT Training

Fine-tune Ministral-3-3B-Instruct for agentic tool-calling using NVIDIA's Nemotron workplace_assistant dataset.

## Authentication

Both HuggingFace and W&B tokens are read automatically from environment variables or cached CLI logins.

**Option 1: CLI login (persists across sessions)**
```bash
huggingface-cli login
wandb login
```

**Option 2: Environment variables (inline)**
```bash
HF_TOKEN=hf_xxx WANDB_API_KEY=xxx python train_sft.py \
    --wandb_project mistral-hackathon \
    --hub_repo your-username/ministral-3b-agent
```

| Service     | Env Variable              | CLI Login              | Cache Location                |
|-------------|---------------------------|------------------------|-------------------------------|
| HuggingFace | `HF_TOKEN`                | `huggingface-cli login`| `~/.cache/huggingface/token`  |
| W&B         | `WANDB_API_KEY`           | `wandb login`          | `~/.netrc`                    |

No code changes needed — the HF Trainer, `huggingface_hub.HfApi`, and `wandb.init()` all pick them up transparently.

## Quickstart

```bash
# Install deps
pip install -r requirements.txt

# Login to HuggingFace and W&B
huggingface-cli login
wandb login

# LoRA training (recommended - fast, memory efficient)
# ~20-30 min on a single RTX 6000 Blackwell (96GB)
python train_sft.py \
    --epochs 2 \
    --lr 2e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 4096 \
    --lora_r 64 \
    --lora_alpha 128 \
    --wandb_project mistral-hackathon \
    --hub_repo your-username/ministral-3b-agent

# Full fine-tune (if you want maximum quality and have VRAM)
python train_sft.py \
    --full_finetune \
    --epochs 2 \
    --lr 5e-6 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --wandb_project mistral-hackathon

# Test the model
python test_inference.py --model_path ./ministral-3b-agent-sft-merged
```

## What the training does

1. **Loads** the NVIDIA Nemotron-RL-agent-workplace_assistant dataset (1.26k train / 545 val)
2. **Preprocesses** by stripping null fields from tool schemas (reduces ~60% of tool definition tokens)
3. **Formats** each example into Mistral's native chat+tool_call template
4. **Trains** with SFT using sequence packing for efficiency
5. **Saves** both LoRA adapter and merged model

## Key training choices

- **Completion-only loss masking**: Only computes loss on assistant-generated tokens (tool calls). System prompts, user messages, and tool schemas are masked out — the model learns *what to generate*, not *how to repeat the prompt*.
- **rsLoRA (rank-stabilized)**: At r=64, standard LoRA scales by `α/r = 2.0`. rsLoRA scales by `α/√r = 16.0`, which keeps gradients stable at high ranks and allows scaling r without retuning alpha.
- **Targets all attention + MLP projections**: q/k/v/o + gate/up/down for maximum expressiveness
- **4-bit QLoRA**: Loads base model in NF4 for memory efficiency during LoRA training
- **Packing=False**: Required when using completion-only masking (they're mutually exclusive in TRL)
- **Cosine LR schedule**: Standard for SFT, with 5% warmup

## Post-training pipeline

After SFT, the hackathon plan is:
1. ✅ SFT (this script)
2. ⬜ GRPO (optional - only if time permits and you have a clear reward signal)
3. ⬜ QAT + nvFP4 quantization via ModelOpt

## File overview

```
train_sft.py        - Main SFT training script
test_inference.py   - Inference testing with browser agent tools
requirements.txt    - Python dependencies
```
