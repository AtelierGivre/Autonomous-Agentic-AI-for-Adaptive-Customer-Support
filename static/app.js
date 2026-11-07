/* Demo chat UI. No framework, no build step, no third-party scripts.
 *
 * Security notes:
 *  - All model/customer text is inserted with textContent, never innerHTML.
 *  - The API key is NOT used here. The demo relies on the development default
 *    customer ("demo"); a real frontend must call its own backend, which keeps
 *    the key server-side and asserts X-Customer-Id after authenticating the user.
 */
(() => {
  "use strict";

  const log = document.getElementById("log");
  const steps = document.getElementById("steps");
  const debugPre = document.getElementById("debug");
  const sessionLabel = document.getElementById("session");
  const traceLabel = document.getElementById("trace");
  const composer = document.getElementById("composer");
  const input = document.getElementById("message");
  const sendButton = document.getElementById("send");
  const clearButton = document.getElementById("clear");
  const healthDot = document.querySelector("#health .dot");
  const healthText = document.getElementById("health-text");

  const state = {
    sessionId: loadSessionId(),
    busy: false,
    lastTraceId: null,
  };

  function loadSessionId() {
    const existing = sessionStorage.getItem("support_session_id");
    if (existing && /^[A-Za-z0-9_-]{1,64}$/.test(existing)) return existing;
    const created = "web-" + Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem("support_session_id", created);
    return created;
  }

  function newTraceId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    sendButton.disabled = busy;
    sendButton.textContent = busy ? "Working…" : "Send";
  }

  function scrollToEnd() {
    log.scrollTop = log.scrollHeight;
  }

  function clearSteps() {
    steps.replaceChildren();
    const hint = document.createElement("p");
    hint.className = "muted";
    hint.textContent = "Send a message to watch the plan unfold.";
    steps.append(hint);
  }

  function addStep(text, done = false) {
    const hint = steps.querySelector(".muted");
    if (hint) hint.remove();
    const row = document.createElement("div");
    row.className = "step" + (done ? " done" : "");
    const strong = document.createElement("strong");
    const arrow = text.indexOf("->");
    if (arrow > 0) {
      strong.textContent = text.slice(0, arrow).trim();
      row.append(strong, document.createTextNode(" " + text.slice(arrow + 2).trim()));
    } else {
      strong.textContent = text;
      row.append(strong);
    }
    steps.append(row);
    const previous = row.previousElementSibling;
    if (previous && previous.classList.contains("step")) previous.classList.add("done");
    steps.scrollTop = steps.scrollHeight;
  }

  function markStepsDone() {
    steps.querySelectorAll(".step").forEach((el) => el.classList.add("done"));
  }

  function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.textContent = text;
    log.append(el);
    scrollToEnd();
  }

  function addBotShell() {
    const el = document.createElement("div");
    el.className = "msg bot";

    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = "working…";

    const body = document.createElement("div");
    body.className = "body";

    const sources = document.createElement("div");
    sources.className = "sources";

    const actions = document.createElement("div");
    actions.className = "actions";

    el.append(badge, body, sources, actions);
    log.append(el);
    scrollToEnd();
    return { el, badge, body, sources, actions };
  }

  function outcomeLabel(result) {
    const labels = {
      answered: "answered",
      no_answer: "no confident answer",
      proposed: "confirmation needed",
      awaiting_confirmation: "confirmation needed",
      escalated: "ticket opened",
      declined: "cancelled",
      blocked: "blocked by guardrail",
      out_of_scope: "out of scope",
      budget: "request limit",
      timeout: "timed out",
      degraded: "temporarily unavailable",
      error: "error",
    };
    return labels[result.outcome] || result.outcome || "reply";
  }

  function setBadge(badge, result) {
    badge.textContent = outcomeLabel(result) + (result.verified ? " · verified" : " · unverified");
    badge.classList.toggle("good", Boolean(result.verified) && result.outcome !== "blocked");
  }

  function addActionButton(actions, label, handler, ghost = false) {
    const button = document.createElement("button");
    button.type = "button";
    if (ghost) button.className = "ghost";
    button.textContent = label;
    button.addEventListener("click", handler);
    actions.append(button);
    return button;
  }

  function showSources(sourcesEl, sources) {
    sourcesEl.replaceChildren();
    if (!Array.isArray(sources) || sources.length === 0) return;
    const label = document.createElement("div");
    label.textContent = "Sources";
    sourcesEl.append(label);
    const list = document.createElement("ul");
    sources.forEach((source) => {
      const item = document.createElement("li");
      item.textContent = `${source.source} (${Number(source.score).toFixed(2)})`;
      list.append(item);
    });
    sourcesEl.append(list);
  }

  function finalizeBot(shell, result) {
    shell.body.textContent = result.answer || "(empty reply)";
    shell.body.classList.remove("error");
    setBadge(shell.badge, result);
    showSources(shell.sources, result.sources);

    if (result.trace_id) {
      state.lastTraceId = result.trace_id;
      traceLabel.textContent = "trace " + result.trace_id.slice(0, 12);
    }

    if (result.debug) {
      debugPre.textContent = JSON.stringify(result.debug, null, 2);
    }

    shell.actions.replaceChildren();

    if (result.needs_confirmation) {
      addActionButton(shell.actions, "Yes, do it", () => {
        shell.actions.replaceChildren();
        send("yes");
      });
      addActionButton(shell.actions, "No, cancel", () => {
        shell.actions.replaceChildren();
        send("no");
      }, true);
    }

    if (result.trace_id) {
      const up = addActionButton(shell.actions, "👍 Helpful", () => rate(result.trace_id, 1, shell.actions), true);
      const down = addActionButton(shell.actions, "👎 Not helpful", () => rate(result.trace_id, -1, shell.actions), true);
      up.dataset.rating = "up";
      down.dataset.rating = "down";
    }

    scrollToEnd();
  }

  function showBotError(shell, message) {
    shell.badge.textContent = "error";
    shell.body.textContent = message;
    shell.body.classList.add("error");
    shell.sources.replaceChildren();
    shell.actions.replaceChildren();
    scrollToEnd();
  }

  async function rate(traceId, rating, actionsEl) {
    actionsEl.querySelectorAll("button").forEach((b) => {
      if (b.dataset.rating) b.disabled = true;
    });
    const note = document.createElement("span");
    note.className = "muted";
    note.textContent = "saving…";
    actionsEl.append(note);
    try {
      const response = await fetch("/v1/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, rating }),
      });
      if (!response.ok) throw new Error("HTTP " + response.status);
      note.textContent = "thanks — recorded";
    } catch (error) {
      note.textContent = "could not save feedback";
      console.error("feedback failed", error);
    }
  }

  function parseSseBlock(block) {
    let event = "message";
    const dataLines = [];
    block.split("\n").forEach((line) => {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    });
    if (dataLines.length === 0) return null;
    const raw = dataLines.join("\n");
    try {
      return { event, data: JSON.parse(raw) };
    } catch (error) {
      console.error("bad SSE payload", raw, error);
      return null;
    }
  }

  async function send(text) {
    const message = (text !== undefined ? text : input.value).trim();
    if (!message || state.busy) return;

    addUserMessage(message);
    if (text === undefined) input.value = "";
    autoGrow();
    setBusy(true);
    clearSteps();
    const shell = addBotShell();

    try {
      const response = await fetch("/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, message }),
      });

      if (!response.ok || !response.body) {
        const detail = await safeError(response);
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answered = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const parsed = parseSseBlock(block);
          if (parsed) {
            if (parsed.event === "step") {
              addStep(parsed.data.text || parsed.data.stage || "working");
            } else if (parsed.event === "answer") {
              answered = true;
              markStepsDone();
              finalizeBot(shell, parsed.data);
            } else if (parsed.event === "error") {
              showBotError(shell, "The server hit an internal error. Please try again.");
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }

      if (!answered) {
        showBotError(shell, "The stream ended without an answer. Please try again.");
      }
    } catch (error) {
      console.error("chat failed", error);
      showBotError(shell, "Could not reach the support service. " + (error.message || ""));
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  async function safeError(response) {
    try {
      const payload = await response.json();
      if (payload && payload.detail) return `HTTP ${response.status}: ${payload.detail}`;
      if (payload && payload.error) return `HTTP ${response.status}: ${payload.error}`;
    } catch (_) {
      /* not JSON */
    }
    return "HTTP " + response.status;
  }

  async function checkHealth() {
    try {
      const live = await fetch("/health/live", { cache: "no-store" });
      if (!live.ok) throw new Error("not live");
      const ready = await fetch("/health/ready", { cache: "no-store" });
      const report = ready.ok ? await ready.json() : null;
      healthDot.classList.remove("bad");
      healthDot.classList.add("ok");
      const provider = report && report.provider ? ` · ${report.provider}` : "";
      const backend = report && report.index && report.index.backend ? ` · ${report.index.backend}` : "";
      healthText.textContent = (ready.ok ? "ready" : "degraded") + provider + backend;
    } catch (error) {
      healthDot.classList.remove("ok");
      healthDot.classList.add("bad");
      healthText.textContent = "unreachable";
    }
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    send();
  });

  input.addEventListener("input", autoGrow);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  document.querySelectorAll("#suggestions button").forEach((button) => {
    button.addEventListener("click", () => send(button.dataset.q || button.textContent));
  });

  clearButton.addEventListener("click", () => {
    log.replaceChildren();
    clearSteps();
    debugPre.textContent = "—";
    traceLabel.textContent = "trace —";
    state.lastTraceId = null;
    input.focus();
  });

  sessionLabel.textContent = "session " + state.sessionId;
  clearSteps();
  checkHealth();
  setInterval(checkHealth, 30000);
})();
