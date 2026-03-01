# Baseline Evaluation Guide

## Overview

`mcqa_exploration.ipynb` runs the Mistral API against the Nemotron MCQA web-search dataset and measures accuracy, parse rate, latency, and accuracy-by-difficulty.

## Required Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MISTRAL_API_KEY` | **Yes** | Your Mistral API key |
| `MISTRAL_MODEL` | No | Model name (default: `mistral-medium-latest`) |
| `MISTRAL_API_BASE` | No | API base URL (default: `https://api.mistral.ai/v1`) |

## Quick Start

```bash
export MISTRAL_API_KEY="your-key-here"
cd mistral_hackathon_2026
jupyter notebook mcqa_exploration.ipynb
```

Then run all cells. Edit the `CONFIG` dict in cell 2 to change settings.

## Configuration (CONFIG dict)

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `"answer_only"` | `"answer_only"` or `"tool_aware_prompt"` |
| `max_examples` | `25` | Number of examples to evaluate |
| `seed` | `42` | Random seed for deterministic sampling |
| `start_idx` | `0` | Offset into shuffled indices |
| `save_every` | `10` | Checkpoint every N examples |
| `output_dir` | `"outputs"` | Where result files are saved |

## Modes

- **`answer_only`** — Asks the model to answer the MCQA question directly.
- **`tool_aware_prompt`** — Tells the model it has `search`/`browse` tools (conceptually) and should reason about what it would search. Tools are NOT executed.

## Output Files

After a run, `outputs/` contains:

| File | Description |
|------|-------------|
| `baseline_results_<mode>.json` | Full results + metrics |
| `baseline_results_<mode>.csv` | Per-example results table |
| `baseline_summary_<mode>.md` | Human-readable summary |

## Comparing Modes

1. Run the notebook with `mode = "answer_only"`
2. Change `CONFIG["mode"]` to `"tool_aware_prompt"` and re-run
3. The final "Compare Modes" cell loads both result files and prints a side-by-side table
