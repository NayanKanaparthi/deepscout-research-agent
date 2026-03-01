# Ministral Agent — Chrome Extension

AI-powered browser agent using fine-tuned Ministral-3B. Opens as a Chrome side panel that connects to a vLLM backend and can search, browse, and extract information from the web.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  Chrome Side Panel   │  HTTP   │   vLLM Backend        │
│                      │────────▶│                       │
│  Chat UI             │         │  /v1/chat/completions │
│  Tool Executor       │◀────────│  Ministral-3B-SFT     │
│    web_search()      │         │  (OpenAI-compatible)  │
│    browse_page()     │         └──────────────────────┘
│    extract_content() │
│    click_element()   │
└─────────────────────┘
```

The agentic loop runs in the extension:
1. User sends query
2. Model returns tool call(s)
3. Extension executes them (navigates tabs, scrapes pages)
4. Results fed back to model
5. Model either calls more tools or gives final answer

## Quick Start

### 1. Start the vLLM Backend

```bash
# Serve the fine-tuned model (merged version)
python -m vllm.entrypoints.openai.api_server \
    --model ./ministral-3b-agent-sft-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral

# Or from HuggingFace Hub
python -m vllm.entrypoints.openai.api_server \
    --model your-username/ministral-3b-agent \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral
```

Key flags:
- `--enable-auto-tool-choice`: Enables tool/function calling
- `--tool-call-parser mistral`: Uses Mistral's native tool call format

Verify it's running:
```bash
curl http://localhost:8000/v1/models
```

### 2. Install the Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder
5. Click the extension icon in the toolbar — the side panel opens

### 3. Configure & Use

1. Click ⚙ in the side panel header
2. Set the vLLM endpoint (default: `http://localhost:8000`)
3. Model name auto-detects from server
4. Start chatting — try the example queries

## Tools

| Tool | What it does |
|------|-------------|
| `web_search(query)` | Opens Google search in active tab, scrapes top results |
| `browse_page(url)` | Navigates to URL, extracts page text content |
| `extract_content(selector)` | Pulls specific DOM elements by CSS selector |
| `click_element(selector)` | Clicks an element on the current page |

## Agent Behavior

The model was fine-tuned on multi-step tool-chaining workflows (workplace assistant tasks), so it naturally:
- Plans sequences of tool calls
- Chains results (search → browse → extract)
- Stops when it has enough information to answer

Max steps defaults to 8 (configurable in settings).

## Files

```
chrome-extension/
├── manifest.json      # Extension manifest v3
├── background.js      # Service worker (opens side panel)
├── content.js         # Injected into pages (DOM interaction)
├── sidepanel.html     # Chat UI + agent loop + vLLM client
├── icons/
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── README.md
```

## Demo Tips

For the hackathon demo:
1. Pre-start vLLM so the model is warm
2. Use queries that showcase multi-step reasoning:
   - "Find who won the latest F1 race and tell me about the winning team"
   - "Search for Mistral AI's latest blog post and summarize it"
   - "What are the top trending repos on GitHub right now?"
3. The side panel shows tool calls executing in real-time — visually impressive
4. The agent loop is visible: tool badges animate, results stream in
