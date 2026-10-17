// Dedicated Streamable HTTP MCP client for the optional Parallel search source.
// No credentials are read or sent. Chrome applies the same User-Agent via DNR.
const DeepscoutSearch = (() => {
  const ENDPOINT = "https://search.parallel.ai/mcp";
  const USER_AGENT = "Deepscout/1.1.0 (https://github.com/NayanKanaparthi/deepscout-research-agent)";
  const PROTOCOL = "2025-03-26";

  function resolveProvider(saved, override) {
    const provider = override ?? saved ?? "ddg";
    if (!["ddg", "parallel", "disabled"].includes(provider)) {
      throw new Error("Unknown search provider. Choose a search source.");
    }
    return provider;
  }

  function safeUrl(value) {
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password;
    } catch (_) { return false; }
  }

  function normalizeResults(data) {
    if (!Array.isArray(data.results)) throw new Error("Invalid Parallel results");
    return data.results.filter((r) => r && safeUrl(r.url)).slice(0, 10).map((r) => ({
      title: typeof r.title === "string" ? r.title : r.url,
      url: r.url,
      snippet: Array.isArray(r.excerpts) ? r.excerpts.filter((x) => typeof x === "string").join("\n").slice(0, 6000) : "",
    }));
  }

  async function readResponse(response, id) {
    if (!response.headers.get("content-type")?.includes("text/event-stream")) {
      return response.json();
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let size = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        size += value?.length || 0;
        if (size > 2_000_000) throw new Error("Parallel response is too large");
        pending = pending.replace(/\r\n/g, "\n");
        let boundary;
        while ((boundary = pending.indexOf("\n\n")) >= 0) {
          const event = pending.slice(0, boundary);
          pending = pending.slice(boundary + 2);
          const data = event.split("\n").filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).replace(/^ /, "")).join("\n");
          if (data) {
            const message = JSON.parse(data);
            if (message.id === id) return message;
          }
        }
        if (done) throw new Error("Parallel stream ended without a response");
      }
    } finally { await reader.cancel(); }
  }

  class ParallelSearchClient {
    constructor({ fetchImpl = globalThis.fetch, timeoutMs = 20000, sessionId } = {}) {
      this.fetch = fetchImpl.bind(globalThis);
      this.timeoutMs = timeoutMs;
      this.conversationId = sessionId || crypto.randomUUID();
      this.nextId = 0;
      this.protocol = PROTOCOL;
      this.session = null;
    }

    async request(method, params, signal, notification = false) {
      signal?.throwIfAborted();
      const controller = new AbortController();
      const cancel = () => controller.abort(signal.reason);
      signal?.addEventListener("abort", cancel, { once: true });
      const timer = setTimeout(() => controller.abort(new Error("Parallel request timed out")), this.timeoutMs);
      const id = ++this.nextId;
      const headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": USER_AGENT,
        "MCP-Protocol-Version": this.protocol,
      };
      if (this.session) headers["Mcp-Session-Id"] = this.session;
      try {
        const response = await this.fetch(ENDPOINT, {
          method: "POST", headers, credentials: "omit", redirect: "error",
          signal: controller.signal,
          body: JSON.stringify({ jsonrpc: "2.0", ...(notification ? {} : { id }), method, ...(params === undefined ? {} : { params }) }),
        });
        if (!response.ok) {
          const retry = response.headers.get("retry-after");
          throw new Error("Parallel HTTP " + response.status + (retry ? " (Retry-After: " + retry + ")" : ""));
        }
        if (notification) return;
        const message = await readResponse(response, id);
        if (message.jsonrpc !== "2.0" || message.id !== id) throw new Error("Invalid Parallel MCP response");
        if (message.error) throw new Error("Parallel MCP: " + message.error.message);
        if (!message.result) throw new Error("Missing Parallel MCP result");
        if (method === "initialize") this.session = response.headers.get("mcp-session-id");
        return message.result;
      } catch (error) {
        if (controller.signal.aborted) throw controller.signal.reason;
        throw error;
      } finally {
        clearTimeout(timer);
        signal?.removeEventListener("abort", cancel);
      }
    }

    async connect(signal) {
      const init = await this.request("initialize", {
        protocolVersion: PROTOCOL, capabilities: {},
        clientInfo: { name: "deepscout-research-agent", version: "1.1.0" },
      }, signal);
      if (init.protocolVersion !== PROTOCOL) throw new Error("Unsupported Parallel MCP protocol");
      await this.request("notifications/initialized", undefined, signal, true);
      const { tools } = await this.request("tools/list", undefined, signal);
      if (!["web_search", "web_fetch"].every((name) => tools?.some((t) => t.name === name))) {
        throw new Error("Parallel search/fetch tools are unavailable");
      }
    }

    async call(name, args, signal) {
      const result = await this.request("tools/call", {
        name, arguments: { ...args, session_id: this.conversationId },
      }, signal);
      const text = (result.content || []).filter((c) => c.type === "text").map((c) => c.text).join("\n");
      if (result.isError) throw new Error("Parallel: " + text.slice(0, 500));
      return result.structuredContent || JSON.parse(text);
    }

    async search(query, signal) {
      return normalizeResults(await this.call("web_search", { objective: query, search_queries: [query] }, signal));
    }

    async enrich(results, query, signal) {
      // Search excerpts normally suffice for the reasoner. Fetch only missing evidence.
      const missing = results.filter((r) => r.snippet.trim().length < 50);
      if (!missing.length) return results;
      const data = await this.call("web_fetch", {
        urls: missing.map((r) => r.url), objective: query.slice(0, 200),
      }, signal);
      if (data.errors?.length) throw new Error("Parallel could not fetch requested page excerpts");
      const pages = normalizeResults(data);
      return results.map((r) => {
        const page = pages.find((p) => p.url === r.url);
        return page ? { ...r, pageContent: page.snippet } : r;
      });
    }

    async close() {
      if (!this.session) return;
      try {
        await this.fetch(ENDPOINT, {
          method: "DELETE", credentials: "omit", redirect: "error",
          headers: { "Mcp-Session-Id": this.session, "MCP-Protocol-Version": this.protocol, "User-Agent": USER_AGENT },
          signal: AbortSignal.timeout(2000),
        });
      } catch (_) { /* Session cleanup must not hide a search error or cancellation. */ }
    }
  }

  return { ENDPOINT, USER_AGENT, resolveProvider, safeUrl, ParallelSearchClient };
})();

if (typeof module !== "undefined") module.exports = DeepscoutSearch;
