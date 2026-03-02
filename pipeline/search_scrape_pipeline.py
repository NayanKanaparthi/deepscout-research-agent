#!/usr/bin/env python3
"""
Search + Scrape Pipeline → JSONL Dataset (append-only)

Reads search queries (one per line from a file, or from CLI args),
runs Brave Search + BeautifulSoup scraping for each, and appends
each result as one line to a .jsonl file.

Output format: one JSON object per line (JSONL)
  {"id":0,"query":"...","search_results":[...],"scraped_pages":[...]}
  {"id":1,"query":"...","search_results":[...],"scraped_pages":[...]}

Resume: at startup, streams the file line-by-line to collect existing
IDs. Already-processed queries are skipped. No full-file rewrites.

Safety: uses a lockfile to prevent concurrent runs writing to the
same output file. The lock is released on exit or crash.

Usage:
    pip install requests beautifulsoup4

    # From a queries file (one query per line)
    python search_scrape_pipeline.py --queries-file queries.txt \
        --api-key YOUR_BRAVE_KEY --count 10

    # Inline queries
    python search_scrape_pipeline.py \
        -q "chronic allograft vasculopathy predictors" \
        -q "Berry phase gauge invariance" \
        --api-key YOUR_BRAVE_KEY

    # Single query (backward compatible)
    python search_scrape_pipeline.py "your search query" \
        --api-key YOUR_BRAVE_KEY --count 10
"""

import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Lockfile — prevents concurrent pipeline runs on the same output
# ---------------------------------------------------------------------------

_lock_fd = None


def acquire_lock(output_path: Path):
    """Acquire an exclusive lock. Exits if another instance is running."""
    global _lock_fd
    lock_path = output_path.parent / f".{output_path.stem}.lock"
    _lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Error: Another pipeline instance is already writing to {output_path}")
        print(f"  Lock file: {lock_path}")
        print("  Kill the other process first, or remove the lock file if stale.")
        sys.exit(1)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
    atexit.register(release_lock)


def release_lock():
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None


def _signal_handler(signum, frame):
    release_lock()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def brave_search(query: str, api_key: str, count: int = 10) -> dict:
    resp = requests.get(
        BRAVE_API_URL,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
        params={"q": query, "count": count},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_and_extract(url: str, max_chars: int = 50000, timeout: int = 15) -> dict:
    """Fetch a URL and extract main text content with BeautifulSoup."""
    result = {"url": url, "title": "", "text": "", "success": False, "error": None}

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text
    except requests.RequestException as e:
        result["error"] = str(e)
        return result

    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title_tag = soup.find("title")
        result["title"] = title_tag.get_text(strip=True) if title_tag else ""

        body = soup.find("body") or soup
        main = body.find("main") or body.find("article") or body.find(attrs={"role": "main"}) or body

        text = main.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"

        result["text"] = text
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# JSONL helpers (append-only, no full-file rewrites)
# ---------------------------------------------------------------------------

def load_existing_ids(path: Path) -> set[int]:
    """Stream .jsonl line-by-line and return set of IDs already written."""
    ids = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ids.add(record["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def append_record(path: Path, record: dict, existing_ids: set[int]):
    """Append a single JSON record, with a final duplicate guard."""
    rid = record["id"]
    if rid in existing_ids:
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    existing_ids.add(rid)
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_query(
    query_id: int,
    query: str,
    api_key: str,
    count: int,
    max_chars: int,
) -> dict:
    """Search + scrape for a single query. Returns a dataset record."""
    print(f"\n[{query_id}] Searching: '{query}' (count={count})")

    try:
        search_data = brave_search(query=query, api_key=api_key, count=count)
    except requests.RequestException as e:
        print(f"  Search failed: {e}")
        return {
            "id": query_id,
            "query": query,
            "search_results": [],
            "scraped_pages": [],
        }

    web_results = search_data.get("web", {}).get("results", [])
    if not web_results:
        print("  No search results.")
        return {
            "id": query_id,
            "query": query,
            "search_results": [],
            "scraped_pages": [],
        }

    search_results = [
        {
            "rank": i,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
        }
        for i, r in enumerate(web_results, 1)
    ]
    print(f"  Got {len(search_results)} results")

    urls = [r["url"] for r in search_results if r["url"]]
    print(f"  Scraping {len(urls)} pages...")

    scraped_pages = []
    for i, url in enumerate(urls, 1):
        print(f"    [{i}/{len(urls)}] {url[:70]}...")
        page = fetch_and_extract(url, max_chars=max_chars)
        scraped_pages.append(page)
        status = "✓" if page["success"] else f"✗ {(page.get('error') or '')[:50]}"
        print(f"           {status}")

    return {
        "id": query_id,
        "query": query,
        "search_results": search_results,
        "scraped_pages": scraped_pages,
    }


def run_pipeline(
    queries: list[str],
    api_key: str,
    count: int = 10,
    output_file: str = "outputs/search_dataset.jsonl",
    max_chars: int = 50000,
    start_id: int = 0,
):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    acquire_lock(out_path)

    existing_ids = load_existing_ids(out_path)
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} records already in {out_path}")

    processed = 0
    skipped = 0

    for i, query in enumerate(queries):
        query_id = start_id + i
        if query_id in existing_ids:
            skipped += 1
            continue

        record = process_query(
            query_id=query_id,
            query=query,
            api_key=api_key,
            count=count,
            max_chars=max_chars,
        )
        written = append_record(out_path, record, existing_ids)
        if not written:
            skipped += 1
            continue
        processed += 1

        n_ok = sum(1 for p in record["scraped_pages"] if p["success"])
        n_total = len(record["scraped_pages"])
        print(f"  → Appended id={query_id} ({n_ok}/{n_total} pages scraped)")

    total_in_file = len(existing_ids)
    print(f"\n{'='*50}")
    print(f"  Done: {processed} new, {skipped} skipped (already existed)")
    print(f"  Output: {out_path}")
    print(f"  Total records in file: {total_in_file}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search + Scrape pipeline → JSONL dataset (append-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Single search query (positional)")
    parser.add_argument(
        "-q", "--queries", action="append", default=[],
        help="Search query (repeatable: -q 'query1' -q 'query2')",
    )
    parser.add_argument(
        "--queries-file",
        help="Text file with one query per line",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAVE_API_KEY"),
        help="Brave API key (or BRAVE_API_KEY env var)",
    )
    parser.add_argument("--count", type=int, default=10, help="Results per query (1-20)")
    parser.add_argument(
        "--output", default="data/search_dataset.jsonl",
        help="Output .jsonl file path",
    )
    parser.add_argument("--max-chars", type=int, default=50000, help="Max chars per page")
    parser.add_argument("--start-id", type=int, default=0, help="Starting sequential ID")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: BRAVE_API_KEY required. Set env var or use --api-key.")
        sys.exit(1)

    all_queries = []

    if args.queries_file:
        p = Path(args.queries_file)
        if not p.exists():
            print(f"Error: queries file not found: {p}")
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_queries.append(line)

    all_queries.extend(args.queries)

    if args.query:
        all_queries.append(args.query)

    if not all_queries:
        print("Error: No queries provided. Use positional arg, -q, or --queries-file.")
        sys.exit(1)

    print(f"Pipeline: {len(all_queries)} queries → {args.output}")

    run_pipeline(
        queries=all_queries,
        api_key=args.api_key,
        count=args.count,
        output_file=args.output,
        max_chars=args.max_chars,
        start_id=args.start_id,
    )


if __name__ == "__main__":
    main()
