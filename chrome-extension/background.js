chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error("Failed to set panel behavior:", error));

const SEARCH_TIMEOUT_MS = 12000;
const SCRAPE_PER_PAGE_MS = 8000;
const MAX_PAGES_TO_SCRAPE = 10;

function scrapeDDGResults() {
  const results = [];
  for (const div of document.querySelectorAll(".result")) {
    if (results.length >= 10) break;
    const linkEl = div.querySelector(".result__a");
    const snippetEl = div.querySelector(".result__snippet");
    if (!linkEl) continue;
    const title = linkEl.textContent.trim();
    let href = linkEl.getAttribute("href") || "";
    if (href.includes("uddg=")) {
      try {
        const parsed = new URL(href, "https://duckduckgo.com");
        href = decodeURIComponent(parsed.searchParams.get("uddg") || href);
      } catch (e) {}
    }
    const snippet = snippetEl ? snippetEl.textContent.trim() : "";
    if (title && href.startsWith("http")) {
      results.push({ title, url: href, snippet });
    }
  }
  return results;
}

function extractPageContent() {
  try {
    const cloned = document.body.cloneNode(true);
    ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]
      .forEach(tag => cloned.querySelectorAll(tag).forEach(el => el.remove()));

    const mainSelectors = ["main", "article", '[role="main"]', "#content", ".content"];
    let text = "";
    for (const sel of mainSelectors) {
      const el = cloned.querySelector(sel);
      if (el && el.innerText.trim().length > 200) {
        text = el.innerText.trim();
        break;
      }
    }
    if (!text) text = cloned.innerText.trim();
    if (text.length > 2000) text = text.substring(0, 2000) + "\n[truncated]";
    return { success: true, text: text };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

function getActiveTabId() {
  return chrome.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
    return tabs.length > 0 ? tabs[0].id : null;
  });
}

function navigateAndExtract(tabId, url, timeoutMs) {
  return new Promise((resolve) => {
    let listener = null;
    let timer = null;
    let fired = false;

    function done(text) {
      if (fired) return;
      fired = true;
      if (timer) clearTimeout(timer);
      if (listener) {
        chrome.tabs.onUpdated.removeListener(listener);
        listener = null;
      }
      resolve(text);
    }

    timer = setTimeout(() => done(""), timeoutMs);

    listener = (tid, info) => {
      if (tid !== tabId || info.status !== "complete") return;
      chrome.scripting
        .executeScript({ target: { tabId: tabId }, func: extractPageContent })
        .then((res) => {
          const r = res && res[0] ? res[0].result : null;
          done(r && r.success ? r.text : "");
        })
        .catch(() => done(""));
    };
    chrome.tabs.onUpdated.addListener(listener);

    chrome.tabs.update(tabId, { url: url }).catch(() => done(""));
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "do_search") {
    handleDoSearch(message);
    return true;
  }
  if (message.action === "scrape_pages") {
    handleScrapePages(message);
    return true;
  }
  return false;
});

function handleDoSearch(message) {
  const { query, requestId } = message;
  const ddgUrl =
    "https://html.duckduckgo.com/html/?q=" + encodeURIComponent(query);

  let settled = false;
  let timeoutId = null;

  function respond(payload) {
    if (settled) return;
    settled = true;
    if (timeoutId) clearTimeout(timeoutId);
    chrome.runtime.sendMessage(
      Object.assign({ action: "search_results", requestId: requestId }, payload)
    );
  }

  timeoutId = setTimeout(() => {
    respond({ error: "Search timed out after " + SEARCH_TIMEOUT_MS + "ms" });
  }, SEARCH_TIMEOUT_MS);

  getActiveTabId().then((tabId) => {
    if (!tabId) {
      respond({ error: "No active tab found" });
      return;
    }

    let listener = (tid, info) => {
      if (tid !== tabId || info.status !== "complete") return;
      chrome.tabs.onUpdated.removeListener(listener);

      chrome.scripting
        .executeScript({ target: { tabId: tabId }, func: scrapeDDGResults })
        .then((injectionResults) => {
          const results =
            injectionResults && injectionResults[0]
              ? injectionResults[0].result || []
              : [];
          respond({ results: results });
        })
        .catch((err) => {
          respond({ error: "Script injection failed: " + err.message });
        });
    };

    chrome.tabs.onUpdated.addListener(listener);
    chrome.tabs.update(tabId, { url: ddgUrl }).catch((err) => {
      respond({ error: "Failed to navigate: " + err.message });
    });
  });
}

async function handleScrapePages(message) {
  const { results, requestId } = message;
  const toScrape = (results || []).slice(0, MAX_PAGES_TO_SCRAPE);

  const tabId = await getActiveTabId();
  if (!tabId) {
    chrome.runtime.sendMessage({
      action: "scrape_results",
      requestId: requestId,
      results: results,
    });
    return;
  }

  const enriched = results.slice();
  for (let i = 0; i < toScrape.length; i++) {
    const text = await navigateAndExtract(tabId, toScrape[i].url, SCRAPE_PER_PAGE_MS);
    if (text) {
      enriched[i] = Object.assign({}, enriched[i], { pageContent: text });
    }
  }

  chrome.runtime.sendMessage({
    action: "scrape_results",
    requestId: requestId,
    results: enriched,
  });
}
