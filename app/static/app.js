/*
 * Chat client for the RAG Document Assistant.
 *
 * Two flows: upload a document to POST /ingest, and ask a question of POST /query,
 * consuming its server-sent event stream. EventSource is not usable here — it only
 * issues GET requests — so the stream is read off the fetch response body directly.
 *
 * Every string that originates outside this file (document filenames, retrieved
 * snippets, model output, error details) is written with textContent or createTextNode.
 * Nothing untrusted is ever assigned to innerHTML.
 */
(() => {
  "use strict";

  const config = JSON.parse(document.getElementById("ui-config").textContent);

  const fileInput = document.getElementById("file-input");
  const uploadButton = document.getElementById("upload-button");
  const uploadStatus = document.getElementById("upload-status");
  const modelSelect = document.getElementById("model-select");
  const topKInput = document.getElementById("topk-input");
  const scopeToggle = document.getElementById("scope-toggle");
  const scopeLabel = document.getElementById("scope-label");
  const chat = document.getElementById("chat");
  const chatEmpty = document.getElementById("chat-empty");
  const srStatus = document.getElementById("sr-status");
  const questionInput = document.getElementById("question-input");
  const sendButton = document.getElementById("send-button");

  /** Id of the most recently ingested document, for the optional scope toggle. */
  let lastDocumentId = null;
  /** Guards against a second in-flight question while one is streaming. */
  let busy = false;
  /** Aborts the in-flight query, from either the stop button or the client timeout. */
  let inFlight = null;

  /**
   * Client-side ceiling on one question, a little above the server's own budget
   * (request_timeout_s, 120 s by default). Without it a wedged connection — a sleeping
   * laptop, a proxy holding the socket — would never resolve the reader and would leave
   * the composer disabled with no way back short of a reload.
   */
  const REQUEST_TIMEOUT_MS = 150000;

  // --- small DOM helpers -----------------------------------------------------

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setStatus(message, tone) {
    uploadStatus.textContent = message;
    uploadStatus.className =
      "mt-2 text-xs " +
      (tone === "error" ? "text-red-400" : tone === "ok" ? "text-emerald-400" : "text-zinc-500");
  }

  /** True when the user is already reading the tail, so streaming should follow it. */
  function isNearBottom() {
    return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
  }

  function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
  }

  /** Announce a one-off state change to assistive technology. */
  function announce(message) {
    srStatus.textContent = message;
  }

  /** Read {detail} off an error response without letting a non-JSON body throw. */
  async function errorDetail(response, fallback) {
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") return body.detail;
      // FastAPI's own validation errors put a list of error objects in `detail`; show
      // the first message rather than falling through to the opaque default.
      if (body && Array.isArray(body.detail) && typeof body.detail[0]?.msg === "string") {
        return body.detail[0].msg;
      }
    } catch {
      /* A proxy or an unhandled error may return HTML; fall through to the default. */
    }
    return fallback + " (" + response.status + ")";
  }

  // --- message bubbles -------------------------------------------------------

  function addUserMessage(text) {
    if (chatEmpty) chatEmpty.remove();
    const row = element("div", "flex justify-end");
    const bubble = element(
      "div",
      "max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white",
      text,
    );
    row.appendChild(bubble);
    chat.appendChild(row);
    scrollToBottom();
  }

  /**
   * Create the assistant bubble and return handles for filling it in as events arrive.
   */
  function addAssistantMessage() {
    if (chatEmpty) chatEmpty.remove();
    const row = element("div", "flex justify-start");
    const bubble = element(
      "div",
      "w-full max-w-[95%] rounded-xl rounded-bl-sm border border-zinc-800 bg-zinc-900/60 px-4 py-3",
    );

    const sources = element("div", "mb-2 hidden flex-wrap gap-1.5");
    // Expanded snippets go here rather than under their badge, so opening one does not
    // break the badge row onto separate lines.
    const snippets = element("div", "mb-2 space-y-1");
    const body = element("div", "whitespace-pre-wrap text-sm leading-relaxed text-zinc-100");
    const footer = element("div", "mt-2 hidden text-xs text-zinc-500");

    // The thinking indicator lives inside the body and is removed on the first token.
    const thinking = element("span", "inline-flex items-center gap-1 align-middle");
    thinking.appendChild(element("span", "sr-only", "Thinking"));
    for (let index = 0; index < 3; index += 1) {
      thinking.appendChild(element("span", "dot h-1.5 w-1.5 rounded-full bg-zinc-500"));
    }
    body.appendChild(thinking);

    bubble.append(sources, snippets, body, footer);
    row.appendChild(bubble);
    chat.appendChild(row);
    scrollToBottom();

    return { bubble, sources, snippets, body, footer, thinking, tokenSeen: false };
  }

  /** A citation label: "[2] report.pdf · p.4 · 0.71", with absent fields omitted. */
  function sourceLabel(source) {
    let label = "[" + source.source_num + "] " + source.filename;
    if (typeof source.page === "number") label += " · p." + source.page;
    if (typeof source.score === "number") label += " · " + source.score.toFixed(2);
    return label;
  }

  /** Render the citation badges; each expands its snippet on click. */
  function renderSources(message, sources) {
    if (!Array.isArray(sources) || sources.length === 0) return;
    message.sources.classList.remove("hidden");
    message.sources.classList.add("flex");

    sources.forEach((source, index) => {
      const badge = element(
        "button",
        "max-w-full truncate rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1 text-left text-xs text-zinc-400 transition hover:border-zinc-700 hover:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-indigo-500",
        sourceLabel(source),
      );
      badge.type = "button";
      badge.setAttribute("aria-expanded", "false");

      const snippet = element(
        "p",
        "hidden whitespace-pre-wrap rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-xs leading-relaxed text-zinc-400",
        source.snippet || "",
      );
      snippet.id = "snippet-" + Date.now() + "-" + index;
      badge.setAttribute("aria-controls", snippet.id);

      badge.addEventListener("click", () => {
        // classList.toggle returns true when the class is now present, i.e. hidden.
        const hidden = snippet.classList.toggle("hidden");
        badge.setAttribute("aria-expanded", String(!hidden));
      });

      message.sources.appendChild(badge);
      message.snippets.appendChild(snippet);
    });
    if (isNearBottom()) scrollToBottom();
  }

  function appendToken(message, text) {
    const follow = isNearBottom();
    if (!message.tokenSeen) {
      message.thinking.remove();
      message.body.classList.add("cursor");
      message.tokenSeen = true;
    }
    message.body.appendChild(document.createTextNode(text));
    if (follow) scrollToBottom();
  }

  function finishMessage(message, note, announcement) {
    message.thinking.remove();
    message.body.classList.remove("cursor");
    if (note) {
      message.footer.textContent = note;
      message.footer.classList.remove("hidden");
    }
    // Announced once, rather than per token: see the role="log" note in the template.
    announce((announcement || "Answer complete.") + " " + message.body.textContent);
    if (isNearBottom()) scrollToBottom();
  }

  function showMessageError(message, detail) {
    message.thinking.remove();
    message.body.classList.remove("cursor");
    const error = element("p", "mt-2 text-xs text-red-400", detail);
    message.bubble.appendChild(error);
    announce(detail);
    if (isNearBottom()) scrollToBottom();
  }

  // --- upload ----------------------------------------------------------------

  async function upload() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Choose a PDF or DOCX file first.", "error");
      return;
    }

    uploadButton.disabled = true;
    setStatus("Uploading and indexing " + file.name + "…", "muted");

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch("/ingest", { method: "POST", body: form });
      if (!response.ok) {
        setStatus(await errorDetail(response, "Upload failed"), "error");
        return;
      }
      const body = await response.json();
      lastDocumentId = body.document.id;
      scopeToggle.disabled = false;
      scopeLabel.className = "flex cursor-pointer items-center gap-2 py-2 text-sm text-zinc-300";
      // Parents are the passages an answer is written from, children the vectors that
      // match; showing both is what makes the small-to-big split visible from the UI.
      const parents = body.parent_count === 1 ? "1 passage" : body.parent_count + " passages";
      setStatus(
        body.document.filename + " indexed — " + parents + ", " + body.child_count + " vectors.",
        "ok",
      );
    } catch {
      // Distinct from an HTTP error: the request never reached the server.
      setStatus("Could not reach the server. Is it still running?", "error");
    } finally {
      uploadButton.disabled = false;
    }
  }

  // --- ask -------------------------------------------------------------------

  function buildRequestBody(question) {
    // QueryRequest forbids extra fields, so only send keys the schema declares, and
    // omit the optional ones entirely rather than sending nulls.
    const body = { query: question, stream: true };
    // An empty value (a dropdown with no options) fails the schema's name pattern with a
    // 422 the user cannot act on; letting the server pick its default is better.
    if (modelSelect.value) body.model = modelSelect.value;

    const topK = Number.parseInt(topKInput.value, 10);
    if (Number.isInteger(topK) && topK >= 1 && topK <= config.maxTopK) {
      body.top_k = topK;
    } else {
      topKInput.value = String(config.defaultTopK);
    }

    if (scopeToggle.checked && lastDocumentId) body.document_id = lastDocumentId;
    return body;
  }

  /** Apply one decoded SSE payload to the assistant message. */
  function handleEvent(message, event, state) {
    if (event.type === "sources") {
      renderSources(message, event.sources);
    } else if (event.type === "token") {
      appendToken(message, event.text);
    } else if (event.type === "error") {
      state.failed = true;
      showMessageError(message, event.detail || "Generation failed before completion.");
    } else if (event.type === "done") {
      state.done = true;
      const elapsed = Math.round(performance.now() - state.startedAt);
      finishMessage(message, event.model + " · " + elapsed + " ms");
    }
  }

  /** Apply every complete `data:` frame in `text` to the message. */
  function handleFrame(frame, message, state) {
    for (const line of frame.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      let event;
      try {
        event = JSON.parse(payload);
      } catch {
        // A malformed frame is not worth aborting a good answer over. Only the parse is
        // guarded: a failure inside handleEvent is a real bug and should surface.
        continue;
      }
      handleEvent(message, event, state);
    }
  }

  /**
   * Read the SSE body, buffering across network chunks. A frame boundary is a blank
   * line, and a single read can land mid-frame or mid-multibyte-character, so the
   * decoder is kept in streaming mode and the tail is carried over.
   */
  async function consumeStream(response, message, state) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          handleFrame(buffer.slice(0, boundary), message, state);
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");
        }
      }
      // Flush the decoder and apply anything left unterminated, so a frame lost to an
      // abrupt close is distinguishable from one this loop simply never processed.
      buffer += decoder.decode();
      if (buffer.trim()) handleFrame(buffer, message, state);
    } finally {
      // Releases the connection when the loop exits early (an abort, or a throw).
      reader.cancel().catch(() => {});
    }
  }

  function setBusy(value) {
    busy = value;
    questionInput.disabled = value;
    // The button becomes a stop control while streaming rather than going dead: an
    // answer that runs long, or a connection that wedges, must stay cancellable.
    sendButton.setAttribute("aria-label", value ? "Stop generating" : "Send question");
    sendButton.classList.toggle("bg-red-600", value);
    sendButton.classList.toggle("hover:bg-red-500", value);
    sendButton.classList.toggle("bg-indigo-600", !value);
    sendButton.classList.toggle("hover:bg-indigo-500", !value);
    const icon = value ? "square" : "arrow-up";
    sendButton.replaceChildren(element("i", "h-4 w-4"));
    sendButton.firstChild.setAttribute("data-lucide", icon);
    if (window.lucide) window.lucide.createIcons({ nameAttr: "data-lucide" });
  }

  async function ask() {
    if (busy) return;
    const question = questionInput.value.trim();
    if (!question) return;

    addUserMessage(question);
    questionInput.value = "";
    questionInput.style.height = "auto";
    setBusy(true);

    const message = addAssistantMessage();
    const state = { startedAt: performance.now(), done: false, failed: false };
    const controller = new AbortController();
    inFlight = controller;
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(buildRequestBody(question)),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        showMessageError(message, await errorDetail(response, "The request failed"));
        return;
      }

      await consumeStream(response, message, state);

      if (!state.done && !state.failed) {
        // The connection closed before the done event — the answer may be truncated.
        showMessageError(message, "The connection closed before the answer finished.");
      }
    } catch (error) {
      if (error && error.name === "AbortError") {
        finishMessage(message, "Stopped.", "Answer stopped.");
      } else {
        showMessageError(message, "Could not reach the server. Check that it is running.");
      }
    } finally {
      window.clearTimeout(timeout);
      inFlight = null;
      setBusy(false);
      questionInput.focus();
    }
  }

  /** Send, or stop the answer already streaming. */
  function sendOrStop() {
    if (busy) {
      if (inFlight) inFlight.abort();
      return;
    }
    ask();
  }

  // --- wiring ----------------------------------------------------------------

  uploadButton.addEventListener("click", upload);
  sendButton.addEventListener("click", sendOrStop);

  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  });

  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
  });

  if (window.lucide) window.lucide.createIcons();
})();
