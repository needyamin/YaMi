"""Self-contained web playground served at ``GET /`` by the dev server.

One static HTML page, zero external assets: it talks to the same JSON API
(``POST /generate`` with ``stream: true``) so the playground and the programmatic
contract can never drift apart.

Conversation model: the page keeps the last few exchanges client-side and
serializes them into the prompt (``### Instruction / ### Response`` blocks),
giving the model multi-turn memory. The inference engine truncates over-long
prompts from the left, so the tiny context window degrades gracefully.
"""

PLAYGROUND_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fontaine AI — Playground</title>
<style>
  :root {
    --ink: #2d2a26; --soft: #6b655d; --paper: #f6f3ee; --card: #ffffff;
    --brand: #4a6fa5; --brand-dark: #375480; --edge: #e7e0d6;
    --user-bg: #eef3fa; --bot-bg: #ffffff; --err-bg: #fdeceb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    display: flex; justify-content: center; min-height: 100vh;
  }
  .app { width: 100%; max-width: 760px; display: flex; flex-direction: column; padding: 18px 16px; }
  header { display: flex; align-items: center; gap: 10px; padding: 4px 2px 14px; }
  header h1 { font-size: 1.15rem; font-weight: 600; margin: 0; color: var(--brand-dark); flex: 1; }
  header .sub { font-size: .8rem; color: var(--soft); }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #c33; flex: none; }
  .dot.ok { background: #3d9a5f; }
  button {
    font: inherit; color: #fff; background: var(--brand); border: 0;
    border-radius: 12px; cursor: pointer;
  }
  button:hover { background: var(--brand-dark); }
  button:disabled { opacity: .55; cursor: wait; }
  #newchat {
    padding: 8px 16px; font-size: .85rem; font-weight: 600; margin-left: 12px;
    background: #fff; color: var(--brand-dark); border: 1px solid var(--edge);
  }
  #newchat:hover { border-color: var(--brand); background: var(--user-bg); }
  .controls {
    display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
    background: var(--card); border: 1px solid var(--edge); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 12px; font-size: .85rem; color: var(--soft);
  }
  .controls label { display: flex; align-items: center; gap: 7px; }
  .controls input[type=range] { width: 110px; accent-color: var(--brand); }
  .controls input[type=number] { width: 70px; padding: 3px 6px; border: 1px solid var(--edge); border-radius: 6px; }
  .controls b { color: var(--ink); font-weight: 600; min-width: 30px; text-align: right; }
  #log { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding: 4px 2px; }
  .msg { border: 1px solid var(--edge); border-radius: 12px; padding: 12px 15px; white-space: pre-wrap;
         word-break: break-word; font-size: .95rem; line-height: 1.55; }
  .msg.you { background: var(--user-bg); align-self: flex-end; max-width: 85%; }
  .msg.bot { background: var(--bot-bg); align-self: flex-start; width: 100%; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
  .msg.err { background: var(--err-bg); align-self: flex-start; width: 100%; color: #8a2f28; }
  .msg .who { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: var(--soft);
              display: block; margin-bottom: 5px; }
  .caret { display: inline-block; width: 8px; height: 15px; background: var(--brand);
           vertical-align: -2px; animation: blink 1s steps(1) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  form { display: flex; gap: 10px; padding-top: 12px; }
  textarea {
    flex: 1; resize: none; height: 74px; padding: 11px 13px; font: inherit; color: var(--ink);
    border: 1px solid var(--edge); border-radius: 12px; background: var(--card); outline: none;
  }
  textarea:focus { border-color: var(--brand); }
  #go { align-self: flex-end; padding: 12px 22px; font-weight: 600; }
  footer { padding-top: 10px; font-size: .75rem; color: var(--soft); text-align: center; }
</style>
</head>
<body>
<div class="app">
  <header>
    <span class="dot" id="dot"></span>
    <div>
      <h1>Fontaine AI — Playground</h1>
      <div class="sub">conversation <span id="turns">0</span> · memory: last 3 exchanges</div>
    </div>
    <button id="newchat" type="button" title="Clear the conversation and start fresh">＋ New conversation</button>
  </header>

  <div class="controls">
    <label>Temperature
      <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.8"> <b id="tempv">0.80</b></label>
    <label>Max new tokens
      <input type="number" id="maxtok" value="160" min="16" max="512" step="16"></label>
  </div>

  <div id="log"></div>

  <form id="ask">
    <textarea id="q"
      placeholder="e.g. Write a Python function that reverses a string.
(Enter to send · Shift+Enter for a new line)"></textarea>
    <button id="go" type="submit">Ask</button>
  </form>
  <footer>Answers come from a tiny 3–5M parameter model — style demo, not production quality.
Tiny model, tiny memory: only the last 3 exchanges fit its context window.</footer>
</div>

<script>
const log = document.getElementById("log");
const q = document.getElementById("q");
const go = document.getElementById("go");
const temp = document.getElementById("temp");
const tempv = document.getElementById("tempv");
const maxtok = document.getElementById("maxtok");
const turnsEl = document.getElementById("turns");

const HISTORY_TURNS = 3;
let history = [];   // [{q: string, a: string}] — the conversation memory

temp.addEventListener("input", () => tempv.textContent = Number(temp.value).toFixed(2));

fetch("/health").then(r => r.ok && (document.getElementById("dot").classList.add("ok")))
  .catch(() => {});

function add(cls, who, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  const w = document.createElement("span"); w.className = "who"; w.textContent = who;
  const body = document.createElement("span"); body.textContent = text;
  el.append(w, body); log.append(el);
  log.scrollTop = log.scrollHeight;
  return body;
}

function updateTurns() { turnsEl.textContent = history.length; }

function resetConversation() {
  history = [];
  updateTurns();
  log.innerHTML = "";
  add("bot", "model", "New conversation started. Ask me something —\\n" +
      "e.g. \\"Write a Python function that adds two numbers.\\"");
  q.focus();
}

function buildPrompt(question) {
  let prompt = "";
  for (const turn of history.slice(-HISTORY_TURNS)) {
    prompt += "### Instruction:\\n" + turn.q.trim() + "\\n\\n### Response:\\n" + turn.a.trim() + "\\n\\n";
  }
  prompt += "### Instruction:\\n" + question.trim() + "\\n\\n### Response:\\n";
  return prompt;
}

async function send(question) {
  add("you", "you", question);
  const body = add("bot", "model", "");
  const caret = document.createElement("span"); caret.className = "caret";
  body.after(caret);
  go.disabled = true;

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: buildPrompt(question), stream: true,
                             temperature: Number(temp.value),
                             max_new_tokens: Number(maxtok.value) }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.error || ("HTTP " + res.status));
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\\n\\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!block.startsWith("data: ")) continue;
        const payload = block.slice(6).trim();
        if (payload === "[DONE]") continue;
        const obj = JSON.parse(payload);
        if (obj.delta) { body.textContent += obj.delta; log.scrollTop = log.scrollHeight; }
      }
    }
    history.push({ q: question, a: body.textContent });
    updateTurns();
  } catch (err) {
    add("err", "error", String(err.message || err));
  } finally {
    caret.remove(); go.disabled = false; q.focus();
  }
}

document.getElementById("ask").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = q.value.trim();
  if (!text) return;
  q.value = ""; send(text);
});

document.getElementById("newchat").addEventListener("click", resetConversation);

resetConversation();
</script>
</body>
</html>
"""
