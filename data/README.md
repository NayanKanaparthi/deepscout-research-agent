# Data Directory

Place dataset files here. Many are gitignored due to size.

| File | Description |
|------|-------------|
| `mcqa_search.jsonl` | MCQ questions with ground truth answers |
| `search_query_data.jsonl` | Question → search query mappings |
| `search_dataset.jsonl` | Output of pipeline: query → Brave results + scraped pages |
| `cot_training_data.jsonl` | Output of generate_cot_dataset: CoT traces for reasoner training |

Generate `search_dataset.jsonl` via `python pipeline/search_scrape_pipeline.py`.
Generate `cot_training_data.jsonl` via `python search_reasoner/generate_cot_dataset.py`.
