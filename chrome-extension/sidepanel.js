// ────────────────────────────────────────────────
//  DEEP RESEARCH AGENT -- 2-Model Pipeline
// ────────────────────────────────────────────────

const CONFIG = {
  queryAgentUrl: "http://localhost:8001/v1/chat/completions",
  queryAgentModel: "search-query-agent",
  reasonerUrl: "http://localhost:8001/v1/chat/completions",
  reasonerModel: "search-reasoner",
  maxResults: 10,
};

const TODAY_STR = new Date().toLocaleDateString("en-US", {
  weekday: "long", year: "numeric", month: "long", day: "numeric",
});

const REASONER_SYSTEM_PROMPT_MCQA = [
  "You are a Search Result Reasoning Agent. Your job is to analyze web search results and their scraped content to answer a user's question accurately.",
  "",
  "## Process",
  "",
  "You MUST think step-by-step inside <think>...</think> tags before giving your final answer. Your reasoning should:",
  "",
  "1. **Evaluate each search result**: For each of the provided search results, assess:",
  "   - Is the source credible and authoritative for this topic?",
  "   - Does the title/snippet suggest it contains relevant information?",
  "   - Does the scraped page content actually contain the answer?",
  "   - Rate each result: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT",
  "",
  "2. **Identify answer-bearing results**: Which specific results contain information that directly answers the question? Quote the relevant passages.",
  "",
  "3. **Cross-reference**: Do multiple sources agree? Are there contradictions? Which source is most trustworthy?",
  "",
  "4. **Synthesize**: Combine information from the best sources into a coherent answer.",
  "",
  "## Output Format",
  "",
  "After your <think>...</think> reasoning, provide your answer in this exact JSON format:",
  "",
  "```json",
  "{",
  '  "result_rankings": [',
  '    {"rank": 1, "result_index": 0, "relevance": "HIGHLY_RELEVANT", "reason": "..."},',
  '    {"rank": 2, "result_index": 1, "relevance": "SOMEWHAT_RELEVANT", "reason": "..."}',
  "  ],",
  '  "best_result_index": 0,',
  '  "answer": "<LETTER>",',
  '  "confidence": 0.95,',
  '  "supporting_evidence": ["<quote from source 1>", "<quote from source 2>"]',
  "}",
  "```",
  "",
  "## Rules",
  "- Always evaluate ALL results before deciding",
  "- Prefer primary sources over secondary",
  "- If no result contains a good answer, say so honestly",
  "- Be specific -- cite which result(s) informed your answer",
  "- If the question is multiple-choice, state the letter answer clearly",
].join("\n");

const REASONER_SYSTEM_PROMPT_FREEFORM = [
  "You are a Search Result Reasoning Agent. Your job is to analyze web search results and their scraped content to answer a user's question accurately.",
  "",
  "## Process",
  "",
  "You MUST think step-by-step inside <think>...</think> tags before giving your final answer. Your reasoning should:",
  "",
  "1. **Evaluate each search result**: For each of the provided search results, assess:",
  "   - Is the source credible and authoritative for this topic?",
  "   - Does the title/snippet suggest it contains relevant information?",
  "   - Does the scraped page content actually contain the answer?",
  "   - Rate each result: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT",
  "",
  "2. **Identify answer-bearing results**: Which specific results contain information that directly answers the question? Quote the relevant passages.",
  "",
  "3. **Cross-reference**: Do multiple sources agree? Are there contradictions? Which source is most trustworthy?",
  "",
  "4. **Synthesize**: Combine information from the best sources into a coherent answer.",
  "",
  "## Output Format",
  "",
  "After your <think>...</think> reasoning, provide your answer in this exact JSON format:",
  "",
  "```json",
  "{",
  '  "result_rankings": [',
  '    {"rank": 1, "result_index": 0, "relevance": "HIGHLY_RELEVANT", "reason": "..."},',
  '    {"rank": 2, "result_index": 1, "relevance": "SOMEWHAT_RELEVANT", "reason": "..."}',
  "  ],",
  '  "best_result_index": 0,',
  '  "answer": "Your synthesized answer here.",',
  '  "confidence": 0.9,',
  '  "supporting_evidence": ["<quote from source 1>", "<quote from source 2>"]',
  "}",
  "```",
  "",
  "## Rules",
  "- Always evaluate ALL results before deciding",
  "- Prefer primary sources over secondary",
  "- If no result contains a good answer, say so honestly",
  "- Be specific -- cite which result(s) informed your answer",
  "- Write the answer as a clear, informative paragraph (2-5 sentences)",
].join("\n");

function isMCQA(text) {
  return /\b[A-D]\s*[:)]\s*.+/m.test(text) && (text.match(/\b[A-D]\s*[:)]/gm) || []).length >= 2;
}

// -- State --
let isProcessing = false;

// -- DOM refs --
const chatContainer = document.getElementById("chat-container");
const emptyState = document.getElementById("empty-state");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");

// -- Input handling --
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener("click", handleSend);

document.querySelectorAll(".example-query").forEach((el) => {
  el.addEventListener("click", () => {
    userInput.value = el.dataset.query;
    handleSend();
  });
});

// -- UI Helpers --
function hideEmptyState() {
  if (emptyState) emptyState.style.display = "none";
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  });
}

function addMessage(role, content) {
  hideEmptyState();
  const msgDiv = document.createElement("div");
  msgDiv.className = "message " + role;

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role;
  msgDiv.appendChild(roleLabel);

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = content;
  msgDiv.appendChild(contentDiv);

  chatContainer.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv;
}

function addPipelineStep(text, status) {
  status = status || "active";
  hideEmptyState();
  const step = document.createElement("div");
  step.className = "pipeline-step " + status;
  if (status === "active") {
    step.innerHTML = '<div class="spinner"></div> ' + escapeHtml(text);
  } else if (status === "done") {
    step.innerHTML = '<span class="check">&#x2713;</span> ' + escapeHtml(text);
  } else {
    step.innerHTML = '<span class="check">&#x2717;</span> ' + escapeHtml(text);
  }
  chatContainer.appendChild(step);
  scrollToBottom();
  return step;
}

function markStepDone(stepEl, text) {
  stepEl.className = "pipeline-step done";
  stepEl.innerHTML = '<span class="check">&#x2713;</span> ' + escapeHtml(text || stepEl.textContent);
}

function markStepError(stepEl, text) {
  stepEl.className = "pipeline-step error";
  stepEl.innerHTML = '<span class="check">&#x2717;</span> ' + escapeHtml(text || stepEl.textContent);
}

function addSearchQueryDisplay(query) {
  const div = document.createElement("div");
  div.className = "search-query-display";
  div.innerHTML = '<div class="label">Search Query</div>' + escapeHtml(query);
  chatContainer.appendChild(div);
  scrollToBottom();
}

function addAnswerCard(answer, confidence, evidence, searchResults) {
  const card = document.createElement("div");
  card.className = "answer-card";

  let evidenceHtml = "";
  if (evidence && evidence.length > 0) {
    const items = evidence
      .map(function (e, i) {
        return '<div class="evidence-item"><span class="cite-num">[' + (i + 1) + "]</span> " + escapeHtml(e) + "</div>";
      })
      .join("");
    evidenceHtml =
      '<div class="answer-evidence"><div class="section-label">Supporting Evidence</div>' +
      items +
      "</div>";
  }

  var sourcesHtml = "";
  if (searchResults && searchResults.length > 0) {
    var sourceItems = searchResults
      .slice(0, 10)
      .map(function (r, i) {
        var domain = "";
        try { domain = new URL(r.url).hostname.replace("www.", ""); } catch (e2) { domain = r.url; }
        return (
          '<a class="source-item" href="' + escapeHtml(r.url) + '" target="_blank" rel="noopener">' +
          '<span class="source-num">[' + (i + 1) + "]</span>" +
          '<span class="source-info">' +
          '<span class="source-title">' + escapeHtml(r.title || domain) + "</span>" +
          '<span class="source-domain">' + escapeHtml(domain) + "</span>" +
          "</span>" +
          '<span class="source-arrow">&#x2197;</span>' +
          "</a>"
        );
      })
      .join("");
    sourcesHtml =
      '<div class="sources-section">' +
      '<div class="section-label">Sources (' + Math.min(searchResults.length, 10) + ")</div>" +
      '<div class="sources-list">' + sourceItems + "</div>" +
      "</div>";
  }

  const pct = Math.round((confidence || 0) * 100);
  card.innerHTML =
    '<div class="answer-header">' +
    '<div style="display:flex;align-items:baseline;gap:8px;">' +
    '<span class="dr-icon">&#x1F50D;</span>' +
    '<span style="font-size:11px;color:var(--text-secondary);font-family:\'JetBrains Mono\',monospace;">ANSWER</span>' +
    '<span class="answer-letter">' + escapeHtml(answer) + "</span>" +
    "</div>" +
    (confidence
      ? '<div class="confidence-pill">' +
        '<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="confidence-value">' + pct + "%</span>" +
        "</div>"
      : "") +
    "</div>" +
    evidenceHtml +
    sourcesHtml;
  chatContainer.appendChild(card);
  scrollToBottom();
}

function addReasoningToggle(reasoning) {
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.gap = "6px";

  const toggle = document.createElement("div");
  toggle.className = "reasoning-toggle";
  toggle.innerHTML =
    '<span class="arrow">&#x25B6;</span> Show reasoning (chain-of-thought)';

  const content = document.createElement("div");
  content.className = "reasoning-content";
  content.textContent = reasoning;

  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    content.classList.toggle("visible");
    scrollToBottom();
  });

  wrapper.appendChild(toggle);
  wrapper.appendChild(content);
  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function addFreeformAnswerCard(answer, confidence, sources, keyFacts, evidence, searchResults) {
  const card = document.createElement("div");
  card.className = "freeform-answer-card";

  const pct = Math.round((confidence || 0) * 100);

  var headerHtml =
    '<div class="answer-header">' +
    '<div class="answer-header-left">' +
    '<span class="dr-icon">&#x1F50D;</span>' +
    '<span class="label">Deepscout</span>' +
    "</div>" +
    (confidence
      ? '<div class="confidence-pill">' +
        '<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="confidence-value">' + pct + "%</span>" +
        "</div>"
      : "") +
    "</div>";

  var bodyHtml = '<div class="answer-body">' + escapeHtml(answer) + "</div>";

  var evidenceHtml = "";
  if (evidence && evidence.length > 0) {
    var items = evidence
      .map(function (e, i) {
        return '<div class="evidence-item"><span class="cite-num">[' + (i + 1) + "]</span> " + escapeHtml(e) + "</div>";
      })
      .join("");
    evidenceHtml =
      '<div class="answer-evidence">' +
      '<div class="section-label">Supporting Evidence</div>' +
      items +
      "</div>";
  }

  var factsHtml = "";
  if (keyFacts && keyFacts.length > 0) {
    var items2 = keyFacts
      .map(function (f) {
        return '<div class="key-fact-item">&#x2022; ' + escapeHtml(f) + "</div>";
      })
      .join("");
    factsHtml =
      '<div class="key-facts">' +
      '<div class="section-label">Key Facts</div>' +
      items2 +
      "</div>";
  }

  var sourcesHtml = "";
  if (searchResults && searchResults.length > 0) {
    var sourceItems = searchResults
      .slice(0, 10)
      .map(function (r, i) {
        var domain = "";
        try { domain = new URL(r.url).hostname.replace("www.", ""); } catch (e) { domain = r.url; }
        return (
          '<a class="source-item" href="' + escapeHtml(r.url) + '" target="_blank" rel="noopener">' +
          '<span class="source-num">[' + (i + 1) + "]</span>" +
          '<span class="source-info">' +
          '<span class="source-title">' + escapeHtml(r.title || domain) + "</span>" +
          '<span class="source-domain">' + escapeHtml(domain) + "</span>" +
          "</span>" +
          '<span class="source-arrow">&#x2197;</span>' +
          "</a>"
        );
      })
      .join("");
    sourcesHtml =
      '<div class="sources-section">' +
      '<div class="section-label">Sources (' + Math.min(searchResults.length, 10) + ")</div>" +
      '<div class="sources-list">' + sourceItems + "</div>" +
      "</div>";
  }

  card.innerHTML = headerHtml + bodyHtml + evidenceHtml + factsHtml + sourcesHtml;
  chatContainer.appendChild(card);
  scrollToBottom();
}

function setStatus(text, color) {
  statusLabel.textContent = text;
  statusDot.style.background = color || "var(--success)";
  statusDot.style.boxShadow = "0 0 6px " + (color || "var(--success)");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

// -- API Calls --

async function callQueryAgent(question) {
  console.log("[QueryAgent] calling model with question:", question.substring(0, 100));
  const body = {
    model: CONFIG.queryAgentModel,
    messages: [
      {
        role: "system",
        content:
          "You are a search query generator. Given a multiple-choice question, ignore answer choices, identify the core concept, and return a single 5-12 word Google search query using domain-appropriate terminology.",
      },
      { role: "user", content: question },
    ],
    max_tokens: 80,
    temperature: 0.01,
  };
  console.log("[QueryAgent] request body:", JSON.stringify(body).substring(0, 300));
  const resp = await fetch(CONFIG.queryAgentUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  console.log("[QueryAgent] response status:", resp.status);
  if (!resp.ok) throw new Error("Query agent error: " + resp.status);
  const data = await resp.json();
  let raw = data.choices[0].message.content.trim();
  console.log("[QueryAgent] raw model output:", raw);
  let cleaned;

  try {
    const parsed = JSON.parse(raw);
    if (parsed.query) {
      cleaned = cleanQuery(parsed.query);
      console.log("[QueryAgent] parsed JSON query, cleaned:", cleaned);
      return cleaned;
    }
  } catch (e) {
    /* not JSON -- use raw text */
  }

  cleaned = cleanQuery(raw);
  console.log("[QueryAgent] cleaned query:", cleaned);
  return cleaned;
}

function cleanQuery(q) {
  q = q.split("\n")[0];
  q = q.replace(/^```[a-z]*\s*/i, "").replace(/```\s*$/g, "");
  q = q.replace(/^\{.*"query"\s*:\s*"?/i, "").replace(/"?\s*\}?\s*$/, "");
  q = q.replace(/[\*\#]+/g, "");
  q = q.replace(/^["']|["']$/g, "");
  q = q.replace(/\(.*?\)/g, "");
  q = q.replace(/\s{2,}/g, " ");
  q = q.trim();
  return q;
}

let _searchIdCounter = 0;

function searchViaTab(query) {
  return new Promise(function (resolve, reject) {
    const requestId = "search_" + Date.now() + "_" + (++_searchIdCounter);
    let settled = false;
    let timeoutId = null;

    function onMessage(msg) {
      if (msg.action !== "search_results" || msg.requestId !== requestId) return;
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      chrome.runtime.onMessage.removeListener(onMessage);

      if (msg.error) {
        reject(new Error(msg.error));
      } else {
        resolve(msg.results || []);
      }
    }

    chrome.runtime.onMessage.addListener(onMessage);

    timeoutId = setTimeout(function () {
      if (settled) return;
      settled = true;
      chrome.runtime.onMessage.removeListener(onMessage);
      reject(new Error("Search timed out (15s)"));
    }, 15000);

    chrome.runtime.sendMessage({ action: "do_search", query: query, requestId: requestId });
  });
}

function scrapePages(results) {
  return new Promise(function (resolve, reject) {
    const requestId = "scrape_" + Date.now() + "_" + (++_searchIdCounter);
    let settled = false;
    let timeoutId = null;

    function onMessage(msg) {
      if (msg.action !== "scrape_results" || msg.requestId !== requestId) return;
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      chrome.runtime.onMessage.removeListener(onMessage);
      resolve(msg.results || results);
    }

    chrome.runtime.onMessage.addListener(onMessage);

    timeoutId = setTimeout(function () {
      if (settled) return;
      settled = true;
      chrome.runtime.onMessage.removeListener(onMessage);
      resolve(results);
    }, 90000);

    chrome.runtime.sendMessage({ action: "scrape_pages", results: results, requestId: requestId });
  });
}

function formatSearchContext(results) {
  var MAX_TOTAL_CHARS = 8000;
  var MAX_PER_PAGE = 2000;
  var SEPARATOR = "\n\n---\n\n";
  var parts = [];
  var totalChars = 0;
  var shown = 0;

  for (var i = 0; i < results.length && shown < CONFIG.maxResults; i++) {
    var r = results[i];
    var pageText = r.pageContent || r.snippet || "";

    if (pageText.length < 50) continue;

    var truncated = pageText.substring(0, MAX_PER_PAGE);
    if (pageText.length > MAX_PER_PAGE) truncated += "\n[truncated]";

    var part =
      "Result " + (shown + 1) + ": " + (r.title || "Untitled") + "\n" +
      "URL: " + (r.url || "") + "\n" +
      "Snippet: " + (r.snippet || "") + "\n" +
      "Page content:\n" + truncated;

    if (totalChars + part.length + SEPARATOR.length > MAX_TOTAL_CHARS && parts.length > 0) break;

    parts.push(part);
    totalChars += part.length + SEPARATOR.length;
    shown++;
  }

  if (parts.length === 0) return "(No usable search results with scraped content)";
  return parts.join(SEPARATOR);
}

async function callReasoner(question, searchContext, mcqa) {
  console.log("[Reasoner] calling model, mcqa:", mcqa, "context length:", searchContext.length);
  const systemPrompt = mcqa
    ? REASONER_SYSTEM_PROMPT_MCQA
    : REASONER_SYSTEM_PROMPT_FREEFORM;
  const userContent =
    "## Question\n" + question + "\n\n## Search Results\n" + searchContext;
  console.log("[Reasoner] total user content length:", userContent.length);

  const resp = await fetch(CONFIG.reasonerUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: CONFIG.reasonerModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
      max_tokens: 1024,
      temperature: 0.1,
    }),
  });
  console.log("[Reasoner] response status:", resp.status);
  if (!resp.ok) {
    var errBody = "";
    try { var ej = await resp.json(); errBody = ej.error && ej.error.message ? ej.error.message : JSON.stringify(ej); } catch (e) { /* ignore */ }
    console.error("[Reasoner] error:", errBody);
    throw new Error("Reasoner error " + resp.status + ": " + errBody);
  }
  const data = await resp.json();
  console.log("[Reasoner] raw output:", data.choices[0].message.content.substring(0, 300));
  console.log("[Reasoner] tokens used:", JSON.stringify(data.usage));
  return data.choices[0].message.content;
}

function sanitizeJsonString(str) {
  return str
    .replace(/\/\/[^\n]*/g, "")
    .replace(/,\s*([\]}])/g, "$1");
}

function extractJsonField(text, field) {
  var re = new RegExp('"' + field + '"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"');
  var m = text.match(re);
  return m ? m[1].replace(/\\"/g, '"').replace(/\\n/g, "\n") : null;
}

function extractJsonNumber(text, field) {
  var re = new RegExp('"' + field + '"\\s*:\\s*([0-9.]+)');
  var m = text.match(re);
  return m ? parseFloat(m[1]) : null;
}

function parseReasonerResponse(text, mcqa) {
  let reasoning = "";
  let answer = null;
  let confidence = null;
  let evidence = [];
  let sources = [];
  let keyFacts = [];

  var thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
  if (thinkMatch) {
    reasoning = thinkMatch[1].trim();
  }

  var afterThink = text.indexOf("</think>") !== -1
    ? text.split("</think>").pop()
    : text;

  var fenceMatch = afterThink.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
  var rawMatch = afterThink.match(/(\{[\s\S]*\})/);
  var jsonStr = fenceMatch ? fenceMatch[1] : rawMatch ? rawMatch[1] : null;

  if (jsonStr) {
    var parsed = null;
    try {
      parsed = JSON.parse(jsonStr);
    } catch (e1) {
      try {
        parsed = JSON.parse(sanitizeJsonString(jsonStr));
      } catch (e2) {
        /* both attempts failed */
      }
    }

    if (parsed) {
      answer = parsed.answer || null;
      confidence = typeof parsed.confidence === "number" ? parsed.confidence : null;
      evidence = parsed.supporting_evidence || [];
      sources = parsed.sources || [];
      keyFacts = parsed.key_facts || [];

      if (mcqa && answer && /^[A-D][:\s]/.test(answer)) {
        answer = answer.charAt(0);
      }

      if (parsed.result_rankings && parsed.result_rankings.length > 0) {
        var rankingText = parsed.result_rankings.map(function (r) {
          return "Result " + (r.result_index + 1) + " [" + r.relevance + "]: " + r.reason;
        }).join("\n\n");
        reasoning = reasoning ? reasoning + "\n\n---\n\n" + rankingText : rankingText;
      }
    }
  }

  if (!answer) {
    answer = extractJsonField(afterThink, "answer");
    if (!answer) answer = extractJsonField(text, "answer");
  }
  if (confidence === null) {
    var c = extractJsonNumber(afterThink, "confidence");
    if (c !== null) confidence = c;
  }

  if (answer && mcqa && /^[A-D][:\s]/.test(answer)) {
    answer = answer.charAt(0);
  }

  if (!reasoning && !answer) {
    var stripped = afterThink
      .replace(/```[\s\S]*$/g, "")
      .replace(/\{[\s\S]*$/g, "")
      .trim();
    if (stripped) answer = stripped;
  }

  return {
    reasoning: reasoning,
    answer: answer,
    confidence: confidence,
    evidence: evidence,
    sources: sources,
    keyFacts: keyFacts,
    mcqa: mcqa,
    raw: text,
  };
}

// -- Main Pipeline --

async function handleSend() {
  const text = userInput.value.trim();
  if (!text || isProcessing) return;

  isProcessing = true;
  sendBtn.disabled = true;
  userInput.value = "";
  userInput.style.height = "auto";

  addMessage("user", text);
  console.log("[Pipeline] starting with input:", text.substring(0, 100));

  try {
    // Step 1: Generate search query
    setStatus("generating query...", "var(--accent)");
    const step1 = addPipelineStep("Generating search query...");

    let searchQuery;
    try {
      searchQuery = await callQueryAgent(text);
      markStepDone(step1, "Search query generated");
      addSearchQueryDisplay(searchQuery);
    } catch (e) {
      markStepError(step1, "Query generation failed: " + e.message);
      addMessage(
        "assistant",
        "Failed to generate search query. Is the search-query-agent running on localhost:8001?\n\nError: " +
          e.message
      );
      return;
    }

    // Step 2: Search via browser tab
    setStatus("searching...", "var(--blue)");
    const step2 = addPipelineStep("Searching the web...");

    let searchResults;
    try {
      searchResults = await searchViaTab(searchQuery);
      if (searchResults.length === 0) {
        markStepError(step2, "No results found");
        addMessage(
          "assistant",
          "DuckDuckGo returned no results for this query. Try rephrasing your question."
        );
        return;
      }
      markStepDone(step2, "Found " + searchResults.length + " results");
    } catch (e) {
      markStepError(step2, "Search failed: " + e.message);
      addMessage(
        "assistant",
        "DuckDuckGo search failed.\n\nError: " + e.message
      );
      return;
    }

    // Step 2b: Scrape page content from top results
    setStatus("reading pages...", "var(--blue)");
    const step2b = addPipelineStep("Scraping page content...");

    try {
      searchResults = await scrapePages(searchResults);
      var scraped = searchResults.filter(function (r) { return r.pageContent; }).length;
      markStepDone(step2b, "Scraped " + scraped + " of " + Math.min(searchResults.length, CONFIG.maxResults) + " pages");
    } catch (e) {
      markStepDone(step2b, "Page scraping skipped (using snippets)");
    }

    // Detect question type
    const mcqa = isMCQA(text);

    // Step 3: Reason over results
    setStatus("reasoning...", "var(--purple)");
    const step3 = addPipelineStep(
      mcqa
        ? "Reasoning over search results..."
        : "Synthesizing answer from results..."
    );

    let reasonerResponse;
    try {
      const searchContext = formatSearchContext(searchResults);
      reasonerResponse = await callReasoner(text, searchContext, mcqa);
      markStepDone(step3, "Reasoning complete");
    } catch (e) {
      markStepError(step3, "Reasoning failed: " + e.message);
      addMessage(
        "assistant",
        "Failed to call search-reasoner. Is it running on localhost:8001?\n\nError: " +
          e.message
      );
      return;
    }

    // Step 4: Parse and display
    const parsed = parseReasonerResponse(reasonerResponse, mcqa);

    if (mcqa && parsed.answer) {
      addAnswerCard(parsed.answer, parsed.confidence, parsed.evidence, searchResults);
    } else if (parsed.answer) {
      addFreeformAnswerCard(
        parsed.answer,
        parsed.confidence,
        parsed.sources.length > 0 ? parsed.sources : null,
        parsed.keyFacts.length > 0 ? parsed.keyFacts : null,
        parsed.evidence,
        searchResults
      );
    } else {
      var fallback = reasonerResponse
        .replace(/<think>[\s\S]*?(<\/think>|$)/g, "")
        .replace(/```[\s\S]*/g, "")
        .replace(/\{[\s\S]*/g, "")
        .trim();
      addMessage("assistant", fallback || "Could not parse a response. Please try again.");
    }

    if (parsed.reasoning) {
      addReasoningToggle(parsed.reasoning);
    }

    setStatus("ready", "var(--success)");
  } catch (e) {
    addMessage("assistant", "Pipeline error: " + e.message);
    setStatus("error", "var(--error)");
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// -- Init --
userInput.focus();
