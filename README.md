# Deepscout

**AI-powered deep research agent that searches the web, scrapes pages, and reasons over real-time information to answer any question — all running locally on a fine-tuned 3B parameter model.**

Built at the [Mistral AI Hackathon 2026](https://mistral.ai).

---

## What is Deepscout?

Deepscout is a Chrome extension that acts as a deep research assistant. Ask it any question — from current events to complex science MCQs — and it will:

1. **Generate a search query** using a fine-tuned `search-query-agent` model
2. **Search the web** via DuckDuckGo directly in your browser tab
3. **Scrape full page content** from the top 10 results
4. **Reason over the evidence** using a fine-tuned `search-reasoner` model with chain-of-thought
5. **Present a cited answer** with confidence scores, supporting evidence, and clickable source links

All inference runs on **Ministral-3-3B-Instruct** with LoRA adapters served via vLLM — no cloud API calls, fully local.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  search-query-agent  │  Fine-tuned LoRA → generates optimal search query
│  (Ministral-3B)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  DuckDuckGo Search   │  Searches via browser tab (visible to user)
│  + Page Scraping     │  Scrapes content from top 10 results
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  search-reasoner     │  Fine-tuned LoRA → analyzes results, ranks relevance,
│  (Ministral-3B)      │  synthesizes answer with chain-of-thought reasoning
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deepscout UI        │  Rich answer card with confidence bar,
│  (Chrome Side Panel) │  numbered citations, and clickable sources
└─────────────────────┘
```

## Key Features

- **Two-model pipeline**: Separate fine-tuned models for query generation and reasoning
- **Real-time web search**: Searches and scrapes live web pages in the user's active tab
- **Chain-of-thought reasoning**: Model shows its work — toggle to see the full reasoning trace
- **Cited answers**: Every answer comes with numbered source citations and clickable links
- **Confidence scoring**: Visual confidence bar shows how certain the model is
- **MCQ + Freeform**: Handles both multiple-choice questions and open-ended queries
- **Fully local inference**: All models run on your own GPU via vLLM — no data leaves your machine

## Project Structure

```
deepscout-research-agent/
├── search_query_agent/        # Model 1: MCQ → search query generation
│   ├── train.py               # SFT training script
│   ├── system_prompt.md
│   └── data.jsonl
├── search_reasoner/           # Model 2: Reason over search results
│   ├── train.py               # QLoRA fine-tuning
│   ├── generate_cot_dataset.py # CoT data via teacher distillation
│   └── eval.py                # Evaluation (base vs fine-tuned)
├── pipeline/                  # Data collection
│   ├── search_scrape_pipeline.py # Brave Search + scrape → JSONL
│   ├── brave_crawler.py
│   └── search_client.py       # Brave API client
├── chrome-extension/          # Deepscout Chrome extension
│   ├── manifest.json
│   ├── sidepanel.html
│   ├── sidepanel.js
│   ├── background.js
│   └── content.js
├── data/                      # Datasets (see data/README.md)
├── outputs/                   # Eval reports, baseline results
├── docs/
├── baseline_100.py            # 100-question benchmark
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, etc.)
- Google Chrome
- [vLLM](https://docs.vllm.ai/) installed

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models (or use pre-trained checkpoints)

```bash
# Train search-query-agent
python search_query_agent/train.py \
    --epochs 3 --lr 2e-4 --lora_r 64 \
    --output_dir ./search-query-agent-sft

# Generate CoT training data (requires MISTRAL_API_KEY)
python search_reasoner/generate_cot_dataset.py

# Train search-reasoner with chain-of-thought
python search_reasoner/train.py \
    --epochs 3 --lr 2e-4 --lora_r 64 \
    --output_dir ./search-reasoner-3b-sft
```

### 3. Serve models with vLLM

```bash
vllm serve mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --port 8001 \
    --max-model-len 4096 \
    --enable-lora \
    --lora-modules \
        search-query-agent=./search-query-agent-sft \
        search-reasoner=./search-reasoner-3b-sft
```

### 4. Install the Chrome extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder
5. Click the Deepscout icon in the toolbar to open the side panel

### 5. Use it

Type any question in the side panel and hit Enter. Watch as Deepscout searches the web in your active tab, scrapes the results, and delivers a researched answer with citations.

## Training Data

- **search-query-agent**: Trained on 500+ MCQ-to-search-query pairs generated from academic questions across physics, chemistry, biology, medicine, and computer science
- **search-reasoner**: Trained on chain-of-thought reasoning traces over real web search results, teaching the model to rank result relevance, extract key information, and synthesize accurate answers

## Model Details

| Model | Base | Adapter | Context | Purpose |
|-------|------|---------|---------|---------|
| `search-query-agent` | Ministral-3-3B-Instruct | LoRA (r=64) | 4096 | Generate optimal search queries from questions |
| `search-reasoner` | Ministral-3-3B-Instruct | LoRA (r=64) | 4096 | Reason over search results and synthesize answers |

## Tech Stack

- **Model**: [Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) by Mistral AI
- **Training**: HuggingFace TRL + PEFT (LoRA/QLoRA)
- **Serving**: vLLM with multi-LoRA support
- **Frontend**: Chrome Extension (Manifest V3) with side panel API
- **Search**: DuckDuckGo (via browser tab navigation)
- **Scraping**: Chrome `scripting` API for content extraction

## Team

Built at the Mistral AI Hackathon 2026.

## License

MIT
