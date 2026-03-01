"""
Brave Search API Client
Docs: https://api.search.brave.com/app/documentation/web-search/get-started
"""

import requests
import json
import argparse
import os


BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def brave_search(
    query: str,
    api_key: str,
    count: int = 10,
    offset: int = 0,
    country: str = "US",
    search_lang: str = "en",
    safe_search: str = "moderate",
    freshness: str = None,
    text_decorations: bool = True,
    spellcheck: bool = True,
) -> dict:
    """
    Perform a web search using the Brave Search API.

    Args:
        query:            Search query string
        api_key:          Brave Search API key
        count:            Number of results to return (1–20, default 10)
        offset:           Pagination offset (default 0)
        country:          Country code, e.g. "US", "GB" (default "US")
        search_lang:      Language code, e.g. "en", "fr" (default "en")
        safe_search:      "off", "moderate", or "strict" (default "moderate")
        freshness:        Optional date filter: "pd" (past day), "pw" (past week),
                          "pm" (past month), or "py" (past year)
        text_decorations: Whether to include result text decorations (default True)
        spellcheck:       Whether to enable spellcheck (default True)

    Returns:
        Parsed JSON response as a dict
    """
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": count,
        "offset": offset,
        "country": country,
        "search_lang": search_lang,
        "safesearch": safe_search,
        "text_decorations": str(text_decorations).lower(),
        "spellcheck": str(spellcheck).lower(),
    }

    if freshness:
        params["freshness"] = freshness

    response = requests.get(BRAVE_API_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def print_results(data: dict) -> None:
    """Pretty-print search results to the console."""
    web_results = data.get("web", {}).get("results", [])

    if not web_results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(web_results)} result(s)")
    print(f"{'='*60}\n")

    for i, result in enumerate(web_results, start=1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        description = result.get("description", "No description")

        print(f"[{i}] {title}")
        print(f"    URL: {url}")
        print(f"    {description}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Brave Search API CLI")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAVE_API_KEY"),
        help="Brave Search API key (or set BRAVE_API_KEY env var)",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of results (1-20)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset")
    parser.add_argument("--country", default="US", help="Country code (e.g. US, GB)")
    parser.add_argument("--lang", default="en", help="Search language code")
    parser.add_argument(
        "--safe-search",
        choices=["off", "moderate", "strict"],
        default="moderate",
        help="Safe search level",
    )
    parser.add_argument(
        "--freshness",
        choices=["pd", "pw", "pm", "py"],
        help="Result freshness: pd=past day, pw=past week, pm=past month, py=past year",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key required. Set BRAVE_API_KEY env var or use --api-key.")
        return

    try:
        results = brave_search(
            query=args.query,
            api_key=args.api_key,
            count=args.count,
            offset=args.offset,
            country=args.country,
            search_lang=args.lang,
            safe_search=args.safe_search,
            freshness=args.freshness,
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_results(results)

    except requests.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    main()
