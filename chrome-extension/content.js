// Content script - injected into all pages
// Handles: page text extraction for potential future use

(() => {
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extract_page") {
      sendResponse(extractFullPage());
    } else if (request.action === "extract_content") {
      sendResponse(extractContent(request.selector));
    } else if (request.action === "get_page_info") {
      sendResponse({
        success: true,
        url: window.location.href,
        title: document.title,
      });
    }
    return true;
  });

  function extractFullPage() {
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
      if (text.length > 4000) text = text.substring(0, 4000) + "\n\n[... truncated]";

      return { success: true, text, url: window.location.href, title: document.title };
    } catch (e) {
      return { success: false, error: e.message };
    }
  }

  function extractContent(selector) {
    try {
      if (!selector) return extractFullPage();
      const elements = document.querySelectorAll(selector);
      if (elements.length === 0) return { success: false, error: `No elements: ${selector}` };
      const texts = Array.from(elements).map(el => el.innerText.trim()).filter(Boolean);
      return { success: true, text: texts.join("\n\n"), count: elements.length, url: window.location.href };
    } catch (e) {
      return { success: false, error: e.message };
    }
  }
})();
