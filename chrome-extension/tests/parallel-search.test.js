const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { ENDPOINT, USER_AGENT, resolveProvider, ParallelSearchClient } = require("../parallel-search.js");

const evidence = "Primary source evidence about CRISPR gene editing and its use of guide RNA. ";
function fixture({ search = [{ title: "Source", url: "https://example.org/paper", excerpts: [evidence] }], fail, sse = false } = {}) {
  const calls = [];
  const fetchImpl = async (url, options) => {
    assert.equal(url, ENDPOINT);
    assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error");
    assert.equal(options.headers["User-Agent"], USER_AGENT);
    assert.equal(options.headers.Authorization, undefined);
    assert.equal(options.headers["x-api-key"], undefined);
    if (options.method === "DELETE") return new Response(null, { status: 204 });
    const request = JSON.parse(options.body);
    calls.push(request);
    if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
    if (request.method !== "initialize") assert.equal(options.headers["Mcp-Session-Id"], "test-session");
    let result;
    if (request.method === "initialize") result = { protocolVersion: "2025-03-26", capabilities: { tools: {} } };
    else if (request.method === "tools/list") result = { tools: [{ name: "web_search" }, { name: "web_fetch" }] };
    else if (fail) return fail(request, options);
    else {
      const data = request.params.name === "web_search" ? { results: search } : {
        results: [{ url: "https://example.org/paper", excerpts: [evidence] }], errors: [],
      };
      result = { content: [{ type: "text", text: JSON.stringify(data) }] };
    }
    const reply = JSON.stringify({ jsonrpc: "2.0", id: request.id, result });
    return new Response(sse ? 'event: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\r\n\r\ndata: ' + reply + "\r\n\r\n" : reply, {
      headers: { "content-type": sse ? "text/event-stream" : "application/json", "mcp-session-id": "test-session" },
    });
  };
  return { calls, fetchImpl };
}

test("fresh default, saved choices, explicit overrides, disabled and invalid settings", () => {
  assert.equal(resolveProvider(null), "ddg");
  assert.equal(resolveProvider("parallel"), "parallel");
  assert.equal(resolveProvider("parallel", "ddg"), "ddg");
  assert.equal(resolveProvider("disabled"), "disabled");
  assert.equal(resolveProvider("ddg", "parallel"), "parallel");
  assert.throws(() => resolveProvider("unknown"), /Unknown search provider/);
});

test("MCP lifecycle, text results and safe source URLs with anonymous project headers", async () => {
  const f = fixture({ search: [
    { title: "Primary", url: "https://example.org/paper?x=1&y=2", excerpts: [evidence] },
    { url: "javascript:alert(1)", excerpts: [evidence] },
    { url: "https://user:pass@example.org", excerpts: [evidence] },
    { url: "file:///secret", excerpts: [evidence] },
  ] });
  const client = new ParallelSearchClient({ ...f, sessionId: "stable-conversation" });
  await client.connect();
  const results = await client.search("CRISPR guide RNA");
  assert.deepEqual(results, [{ title: "Primary", url: "https://example.org/paper?x=1&y=2", snippet: evidence }]);
  assert.equal((await client.enrich(results, "CRISPR")), results);
  const call = f.calls.find((c) => c.method === "tools/call");
  assert.deepEqual(call.params.arguments, { objective: "CRISPR guide RNA", search_queries: ["CRISPR guide RNA"], session_id: "stable-conversation" });
  await client.close();
});

test("SSE discovery and search; fetch only short excerpts and preserve URLs and session", async () => {
  const f = fixture({ search: [{ title: "Source", url: "https://example.org/paper", excerpts: [] }], sse: true });
  const client = new ParallelSearchClient({ ...f, sessionId: "stable-conversation" });
  await client.connect();
  const enriched = await client.enrich(await client.search("CRISPR"), "CRISPR");
  assert.equal(enriched[0].pageContent, evidence);
  const fetch = f.calls.find((c) => c.params?.name === "web_fetch");
  assert.deepEqual(fetch.params.arguments.urls, ["https://example.org/paper"]);
  assert.equal(fetch.params.arguments.session_id, "stable-conversation");
});

test("structured tool responses are consumed", async () => {
  const f = fixture({ fail: async (r) => new Response(JSON.stringify({ jsonrpc: "2.0", id: r.id,
    result: { structuredContent: { results: [{ url: "https://example.org", excerpts: [evidence] }] } } })) });
  const client = new ParallelSearchClient(f);
  await client.connect();
  assert.equal((await client.search("query"))[0].snippet, evidence);
});

for (const [label, fail, expected] of [
  ["HTTP quota", async () => new Response("quota", { status: 429, headers: { "retry-after": "60" } }), /HTTP 429.*60/],
  ["permission denial", async () => { throw new TypeError("Failed to fetch"); }, /Failed to fetch/],
  ["tool errors", async (r) => new Response(JSON.stringify({ jsonrpc: "2.0", id: r.id, result: { isError: true, content: [{ type: "text", text: "quota exhausted" }] } })), /quota exhausted/],
  ["RPC errors", async (r) => new Response(JSON.stringify({ jsonrpc: "2.0", id: r.id, error: { message: "bad arguments" } })), /bad arguments/],
  ["malformed JSON", async () => new Response("not JSON"), /JSON/],
  ["wrong response ID", async () => new Response(JSON.stringify({ jsonrpc: "2.0", id: -1, result: {} })), /Invalid Parallel/],
]) {
  test(label + " is surfaced without retry or provider fallback", async () => {
    const f = fixture({ fail }); const client = new ParallelSearchClient(f);
    await client.connect();
    await assert.rejects(client.search("query"), expected);
    assert.equal(f.calls.filter((c) => c.method === "tools/call").length, 1);
  });
}

test("timeouts and cancellation abort the actual pending caller", async () => {
  for (const cancel of [false, true]) {
    let aborted = false;
    const f = fixture({ fail: (_, options) => new Promise((resolve, reject) => {
      options.signal.addEventListener("abort", () => { aborted = true; reject(options.signal.reason); }, { once: true });
    }) });
    const client = new ParallelSearchClient({ ...f, timeoutMs: 30 });
    await client.connect();
    const controller = new AbortController();
    const pending = client.search("query", controller.signal);
    if (cancel) controller.abort(new Error("Research cancelled"));
    await assert.rejects(pending, cancel ? /Research cancelled/ : /timed out/);
    assert.equal(aborted, true);
  }
});

test("cancel before discovery makes no network request", async () => {
  const f = fixture(); const client = new ParallelSearchClient(f);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(client.connect(controller.signal), { name: "AbortError" });
  assert.equal(f.calls.length, 0);
});

test("native fetch is bound to its global receiver", async () => {
  const client = new ParallelSearchClient({ fetchImpl: function () {
    assert.equal(this, globalThis);
    throw new Error("receiver verified");
  } });
  await assert.rejects(client.connect(), /receiver verified/);
});

test("missing tools and unsupported protocols stop before search", async () => {
  for (const protocol of ["2025-03-26", "unsupported"]) {
    const calls = [];
    const client = new ParallelSearchClient({ fetchImpl: async (url, options) => {
      const request = JSON.parse(options.body); calls.push(request.method);
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result:
        request.method === "initialize" ? { protocolVersion: protocol } : { tools: [] } }));
    } });
    await assert.rejects(client.connect(), protocol === "unsupported" ? /Unsupported/ : /unavailable/);
    assert.equal(calls.includes("tools/call"), false);
  }
});

test("fetch errors stop missing-evidence enrichment", async () => {
  const f = fixture({ fail: async (r) => new Response(JSON.stringify({ jsonrpc: "2.0", id: r.id,
    result: { structuredContent: { results: [], errors: [{ url: "https://example.org", error_type: "fetch_failed" }] } } })) });
  const client = new ParallelSearchClient(f); await client.connect();
  await assert.rejects(client.enrich([{ url: "https://example.org", snippet: "" }], "query"), /could not fetch/);
});

test("browser header rule is restricted to extension-initiated MCP requests", async () => {
  let listener; let rules;
  const context = { console, chrome: {
    sidePanel: { setPanelBehavior: () => Promise.resolve() },
    runtime: { id: "extension-id", onMessage: { addListener(fn) { listener = fn; } } },
    declarativeNetRequest: { updateSessionRules: async (value) => { rules = value; } },
  } };
  vm.runInNewContext(fs.readFileSync(require.resolve("../background.js"), "utf8"), context);
  const response = await new Promise((resolve) => listener({ action: "prepare_parallel" }, {}, resolve));
  assert.equal(response.ready, true);
  const rule = rules.addRules[0];
  assert.equal(rule.condition.initiatorDomains[0], "extension-id");
  const regex = new RegExp(rule.condition.regexFilter);
  assert.equal(regex.test(ENDPOINT), true);
  assert.equal(regex.test(ENDPOINT + "-oauth"), false);
  assert.equal(regex.test("https://example.org/mcp"), false);
  assert.equal(rule.action.requestHeaders[0].value, USER_AGENT);
});
