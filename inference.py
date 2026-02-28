#!/usr/bin/env python3
"""
Quick inference test for the fine-tuned Ministral-3B agent model.
Tests tool-calling ability with sample queries.

Usage:
    python test_inference.py --model_path ./ministral-3b-agent-sft-merged
    python test_inference.py --model_path ./ministral-3b-agent-sft  # LoRA adapter
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEARCH_TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for information. Returns a list of search results with titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "browse_page",
        "description": "Navigate to a URL and return the page content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "click_element",
        "description": "Click on an element on the current page.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click",
                }
            },
            "required": ["selector"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fill_input",
        "description": "Fill in a form input field on the current page.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input element",
                },
                "value": {
                    "type": "string",
                    "description": "Value to type into the input",
                },
            },
            "required": ["selector", "value"],
            "additionalProperties": False,
        },
    },
]

TEST_QUERIES = [
    "Search for the latest news about AI regulation in the EU",
    "Find me the best rated Italian restaurants in San Francisco",
    "Look up the current stock price of NVIDIA",
    "Search for how to train a LoRA adapter and open the first result",
    "What's the weather forecast for New York this weekend?",
]


def load_model(model_path, device="auto"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Try loading as a full model first, fall back to LoRA
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception:
        print("Loading as LoRA adapter...")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Ministral-3-3B-Instruct-2512-BF16",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, query, tools=None, max_new_tokens=512):
    """Generate a response for a given query."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful browser assistant. You can search the web, "
                "browse pages, click elements, and fill in forms to help the user. "
                "Use the available tools to accomplish the user's request. "
                "Issue one tool call at a time."
            ),
        },
        {"role": "user", "content": query},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback
        tool_desc = json.dumps(tools, indent=2) if tools else ""
        messages[0]["content"] += f"\n\nAvailable tools:\n{tool_desc}"
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated, skip_special_tokens=False)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query (if not provided, runs test suite)",
    )
    parser.add_argument(
        "--no_tools", action="store_true", help="Don't pass tool definitions"
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    tools = None if args.no_tools else SEARCH_TOOLS

    if args.query:
        queries = [args.query]
    else:
        queries = TEST_QUERIES

    print("\n" + "=" * 60)
    print("Testing fine-tuned model")
    print("=" * 60)

    for i, query in enumerate(queries):
        print(f"\n{'─' * 60}")
        print(f"Query {i + 1}: {query}")
        print(f"{'─' * 60}")
        response = generate_response(model, tokenizer, query, tools)
        print(f"Response:\n{response}")

    # Also run a comparison with base model if not custom query
    if not args.query:
        print("\n\n" + "=" * 60)
        print("Quick comparison: try the same with base model:")
        print(
            "  python test_inference.py --model_path mistralai/Ministral-3-3B-Instruct-2512-BF16"
        )
        print("=" * 60)


if __name__ == "__main__":
    main()
