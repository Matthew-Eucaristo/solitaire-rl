/* Klondike UI — talks only to /api; every rule decision lives server-side
   in the same engine the tests and evals use. */

const $ = (s) => document.querySelector(s);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
};
const post = (path, body) =>
  api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });

const SUIT_GLYPH = ["♣", "♦", "♥", "♠"];
const SUIT_RED = [false, true, true, false];
const K = { DRAW: 0, W2T: 1, W2F: 2, T2F: 3, T2T: 4 };

let S = null;              // last serialized state
let watching = false;
let watchTimer = null;
let prevRects = new Map(); // card id -> rect, for FLIP animation
let dealStart = Date.now();
let timeTimer = null;

/* ---------------- sfx (synthesized, no assets) ---------------- */

const sfx = (() => {
  let ctx = null;
  let muted = localStorage.getItem("klondike.muted") === "1";
  const ac = () => (ctx = ctx || new (window.AudioContext || window.webkitAudioContext)());
  const tone = (freq, dur, type = "triangle", gain = 0.12, delay = 0) => {
    if (muted) return;
    try {
      const c = ac(), o = c.createOscillator(), g = c.createGain();
      o.type = type; o.frequency.value = freq;
      g.gain.setValueAtTime(gain, c.currentTime + delay);
      g.gain.exponentialRampToValueAtTime(0.001, c.currentTime + delay + dur);
      o.connect(g).connect(c.destination);
      o.start(c.currentTime + delay); o.stop(c.currentTime + delay + dur);
    } catch {}
  };
  const swish = (dur = 0.09, gain = 0.05) => {
    if (muted) return;
    try {
      const c = ac(), n = c.sampleRate * dur, buf = c.createBuffer(1, n, c.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
      const src = c.createBufferSource(), g = c.createGain(), f = c.createBiquadFilter();
      f.type = "bandpass"; f.frequency.value = 1800;
      g.gain.value = gain;
      src.buffer = buf; src.connect(f).connect(g).connect(c.destination);
      src.start();
    } catch {}
  };
  return {
    get muted() { return muted; },
    toggle() { muted = !muted; localStorage.setItem("klondike.muted", muted ? "1" : "0"); return muted; },
    unlock() { try { ac().resume(); } catch {} },
    flip: () => swish(0.09, 0.06),
    draw: () => { tone(340, 0.06, "square", 0.05); swish(0.05, 0.03); },
    found: () => { tone(880, 0.12, "sine", 0.1); tone(1320, 0.1, "sine", 0.06, 0.05); },
    place: () => tone(520, 0.07, "triangle", 0.08),
    bad: () => tone(140, 0.16, "sawtooth", 0.08),
    deal: () => { for (let i = 0; i < 7; i++) swish(0.04, 0.02), tone(600 + i * 40, 0.03, "square", 0.015, i * 0.045); },
    win: () => [523, 659, 784, 1046, 1318].forEach((f, i) => tone(f, 0.35, "triangle", 0.12, i * 0.12)),
    lose: () => { tone(220, 0.3, "sine", 0.1); tone(165, 0.4, "sine", 0.1, 0.18); },
    concede: () => tone(196, 0.25, "sine", 0.08),
  };
})();

/* ---------------- confetti on win ---------------- */

function confetti() {
  const colors = ["#ffd95e", "#4fd2ff", "#ff6b6b", "#7ee081", "#c084fc", "#fff"];
  for (let i = 0; i < 70; i++) {
    const p = document.createElement("div");
    p.className = "confetti";
    p.style.left = `${45 + Math.random() * 10}%`;
    p.style.top = "42%";
    p.style.background = colors[i % colors.length];
    p.style.setProperty("--dx", `${(Math.random() - 0.5) * 560}px`);
    p.style.setProperty("--dy", `${-80 - Math.random() * 320}px`);
    p.style.setProperty("--rot", `${(Math.random() - 0.5) * 720}deg`);
    p.style.animationDelay = `${Math.random() * 0.15}s`;
    document.body.appendChild(p);
    setTimeout(() => p.remove(), 2600);
  }
}

/* ---------------- rendering ---------------- */

function cardEl(code, faceDown) {
  const el = document.createElement("div");
  el.className = "card " + (faceDown ? "down" : "up");
  if (!faceDown) {
    const rank = code[0], suit = "CDHS".indexOf(code[1]);
    el.dataset.card = code;
    if (SUIT_RED[suit]) el.classList.add("red");
    const glyph = SUIT_GLYPH[suit];
    el.innerHTML =
      `<div class="corner tl">${rank}<br>${glyph}</div>` +
      `<div class="corner br">${rank}<br>${glyph}</div>` +
      `<div class="pip">${glyph}</div>`;
  }
  return el;
}

function snapshotRects() {
  prevRects = new Map();
  document.querySelectorAll(".card.up[data-card]").forEach((el) => {
    prevRects.set(el.dataset.card + "|" + el.dataset.pos, el.getBoundingClientRect());
  });
}

function flipAnimate() {
  document.querySelectorAll(".card.up[data-card]").forEach((el) => {
    const old = prevRects.get(el.dataset.card + "|" + el.dataset.pos);
    if (!old) return;
    const r = el.getBoundingClientRect();
    const dx = old.left - r.left, dy = old.top - r.top;
    if (dx || dy) {
      el.style.transition = "none";
      el.style.transform = `translate(${dx}px, ${dy}px)`;
      requestAnimationFrame(() => {
        el.style.transition = "";
        el.style.transform = "";
      });
    }
  });
}

function render(newDeal) {
  snapshotRects();
  const st = S;

  /* stock */
  const stock = $("#stock");
  stock.className = "pile slot";
  stock.innerHTML = "";
  if (st.stock_count > 0) {
    const back = cardEl(null, true);
    back.style.position = "relative";
    stock.appendChild(back);
    stock.title = `${st.stock_count} in stock — click to draw ${st.draw_size}`;
  } else {
    const recycles = st.legal.some((m) => m.kind === K.DRAW);
    stock.innerHTML = recycles ? `<div class="redeal-badge">↺</div>` : "";
    stock.title = recycles ? "redeal waste into stock" : "empty";
  }
  stock.classList.toggle("clickable", st.legal.some((m) => m.kind === K.DRAW));
  stock.onclick = () => doMove({ kind: K.DRAW });
  delete stock.dataset.col;

  /* waste — fan the last few for draw-3 */
  const waste = $("#waste");
  waste.innerHTML = "";
  waste.dataset.zone = "waste";
  const show = Math.min(st.waste.length, st.draw_size === 3 ? 3 : 1);
  for (let i = st.waste.length - show, k = 0; i < st.waste.length; i++, k++) {
    const el = cardEl(st.waste[i]);
    el.style.position = "absolute";
    el.style.left = `${k * 18}px`;
    el.style.top = "0px";
    el.dataset.zone = "waste";
    el.dataset.idx = i;
    if (i < st.waste.length - 1) el.style.pointerEvents = "none";
    waste.appendChild(el);
  }
  waste.style.width = `calc(var(--cw) + ${(show - 1) * 18}px)`;

  /* foundations */
  const fdn = $("#foundations");
  fdn.innerHTML = "";
  for (let s = 0; s < 4; s++) {
    const pile = document.createElement("div");
    pile.className = "pile slot suit";
    pile.setAttribute("data-suit", SUIT_GLYPH[s]);
    pile.dataset.col = `f${s}`;
    const top = st.foundations[s][st.foundations[s].length - 1];
    if (top) pile.appendChild(cardEl(top));
    fdn.appendChild(pile);
  }

  /* tableau */
  const tab = $("#tableau");
  tab.innerHTML = "";
  st.tableau.forEach((col, c) => {
    const pile = document.createElement("div");
    pile.className = "pile";
    pile.dataset.col = c;
    let y = 0;
    for (let i = 0; i < col.down; i++) {
      const el = cardEl(null, true);
      el.style.top = `${y}px`;
      y += cssNum("--dn-off");
      pile.appendChild(el);
    }
    col.up.forEach((code, j) => {
      const el = cardEl(code);
      el.style.top = `${y}px`;
      y += cssNum("--up-off");
      el.dataset.zone = "tableau";
      el.dataset.col = c;
      el.dataset.idx = j;
      el.dataset.pos = `t${c}.${j}`;
      pile.appendChild(el);
    });
    if (!col.down && !col.up.length) pile.classList.add("slot");
    if (newDeal) pile.querySelectorAll(".card").forEach((el, i) => {
      el.classList.add("deal-anim");
      el.style.animationDelay = `${(c * 40 + i * 30)}ms`;
    });
    tab.appendChild(pile);
  });

  /* status */
  $("#stSeed").textContent = st.seed;
  $("#stMoves").textContent = st.moves;
  $("#stFound").textContent = `${st.foundations_count}/52`;
  $("#stRedeals").textContent = st.redeals;
  $("#undo").disabled = !st.can_undo;
  $("#auto").disabled = st.outcome !== null || st.facedown > 0;
  $("#auto").title = st.facedown > 0
    ? `auto-finish unlocks once all ${st.facedown} face-down cards are revealed`
    : "let the heuristic finish the remaining safe moves";
  flipAnimate();
  showOutcome(st.outcome);
}

function cssNum(v) {
  return parseInt(getComputedStyle(document.documentElement).getPropertyValue(v));
}

/* ---------------- outcomes / toast ---------------- */

function showOutcome(outcome) {
  const b = $("#banner");
  if (!outcome) { b.classList.add("hidden"); return; }
  stopAgent();
  const msgs = {
    win: ["You win! 🏆", "All 52 cards on the foundations."],
    concede: ["Conceded", "Some deals can't be won — knowing when to fold is a skill too."],
    no_moves: ["No moves left", "The board is locked. Deal again?"],
    no_progress_cycle: ["Cycle detected", "You revisited an earlier position — the game declared a loss."],
    no_progress_idle: ["No progress", "Too long without revealing or founding a card — the deal timed out."],
    truncated: ["Step limit", "500 moves reached — the episode was truncated."],
    illegal_action: ["Illegal move", "The engine rejected an action."],
  };
  const [t, sub] = msgs[outcome] || [outcome, ""];
  b.className = outcome === "win" ? "win" : "lose";
  b.innerHTML = `<div class="card-msg">${t}<div class="sub">${sub}<br>deal ${S.seed} · ${S.moves} moves · ${elapsed()}</div><button class="primary" onclick="newDeal()">New deal</button></div>`;
  b.classList.remove("hidden");
  if (outcome === "win") { sfx.win(); confetti(); }
  else if (outcome === "concede") sfx.concede();
  else sfx.lose();
}

function elapsed() {
  const s = Math.floor((Date.now() - dealStart) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

let toastT = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastT);
  toastT = setTimeout(() => t.classList.add("hidden"), 2200);
}

/* ---------------- moves ---------------- */

async function apply(resp, keepWatch) {
  const wasWatching = watching;
  S = resp;
  render();
  if (resp.applied) {
    const a = resp.applied;
    const name = a.concede ? "concede"
      : a.kind_name === "DRAW" ? "draw / redeal"
      : (a.cards || []).join(" ");
    setAgentStatus(`${wasWatching ? "agent" : "you"}: ${name}`);
  }
  if (watching && !keepWatch) scheduleAgent();
}

async function doMove(desc) {
  try {
    const resp = await post("/api/move", desc);
    if (desc.kind === K.DRAW) sfx.draw();
    else if (resp.applied && (resp.applied.kind === K.T2F || resp.applied.kind === K.W2F)) sfx.found();
    else if (resp.applied && resp.applied.kind === K.T2T && resp.applied.cards && resp.applied.cards.length) sfx.flip();
    else sfx.place();
    apply(resp);
  } catch (e) {
    sfx.bad();
    toast(e.message);
  }
}

async function newDeal() {
  stopAgent();
  const seed = $("#seed").value.trim();
  S = await post("/api/new", { variant: $("#variant").value, seed: seed ? +seed : null });
  setAgentStatus("");
  dealStart = Date.now();
  sfx.deal();
  render(true);
}
window.newDeal = newDeal;

/* ---------------- drag & drop ---------------- */

function movesForCard(el) {
  const zone = el.dataset.zone;
  if (!S || !S.legal) return [];
  if (zone === "waste") {
    if (+el.dataset.idx !== S.waste.length - 1) return [];
    return S.legal.filter((m) => m.kind === K.W2T || m.kind === K.W2F);
  }
  const c = +el.dataset.col, j = +el.dataset.idx;
  const upLen = S.tableau[c].up.length;
  return S.legal.filter(
    (m) =>
      (m.kind === K.T2T && m.src === c && m.run_start === j) ||
      (m.kind === K.T2F && m.src === c && j === upLen - 1)
  );
}

function dropTargets(cands) {
  const out = new Map();
  for (const m of cands) {
    const key = m.kind === K.T2F || m.kind === K.W2F ? `f${m.dst}` : `${m.dst}`;
    out.set(key, m);
  }
  return out;
}

document.addEventListener("pointerdown", (e) => {
  if (watching || !S || S.outcome) return;
  const el = e.target.closest(".card.up");
  if (!el || !el.dataset.zone) return;
  const cands = movesForCard(el);
  if (!cands.length) return;

  const targets = dropTargets(cands);
  const zone = el.dataset.zone;
  const col = +el.dataset.col, idx = +el.dataset.idx;

  /* ghost stack */
  const ghost = document.createElement("div");
  ghost.className = "ghost";
  const srcCards =
    zone === "waste" ? [el] : [...el.parentElement.querySelectorAll(".card.up")].filter((n) => +n.dataset.idx >= idx);
  srcCards.forEach((n) => ghost.appendChild(n.cloneNode(true)));
  document.body.appendChild(ghost);
  const grab = { x: e.clientX - el.getBoundingClientRect().left, y: e.clientY - el.getBoundingClientRect().top };
  const place = (ev) => { ghost.style.left = ev.clientX - grab.x + "px"; ghost.style.top = ev.clientY - grab.y + "px"; };
  place(e);
  srcCards.forEach((n) => n.classList.add("dragging"));

  /* highlight legal targets */
  document.querySelectorAll("#tableau .pile, #foundations .pile").forEach((p) => {
    p.classList.toggle("drop-ok", targets.has(p.dataset.col));
  });

  const up = (ev) => {
    document.removeEventListener("pointermove", place);
    document.removeEventListener("pointerup", up);
    ghost.remove();
    srcCards.forEach((n) => n.classList.remove("dragging"));
    document.querySelectorAll(".drop-ok").forEach((p) => p.classList.remove("drop-ok"));
    ghost.style.display = "none";
    const under = document.elementFromPoint(ev.clientX, ev.clientY);
    const pile = under && under.closest(".pile");
    if (pile && targets.has(pile.dataset.col)) {
      doMove(targets.get(pile.dataset.col));
    }
  };
  document.addEventListener("pointermove", place);
  document.addEventListener("pointerup", up);
  e.preventDefault();
});

/* double-click: foundation first, else any move for that card */
document.addEventListener("dblclick", (e) => {
  if (watching || !S || S.outcome) return;
  const el = e.target.closest(".card.up");
  if (!el || !el.dataset.zone) return;
  const cands = movesForCard(el);
  const pick = cands.find((m) => m.kind === K.T2F || m.kind === K.W2F) || cands[0];
  if (pick) doMove(pick);
});

/* ---------------- agent ---------------- */

async function loadPolicies() {
  const { policies } = await api("/api/agent/policies");
  const sel = $("#agentPolicy");
  sel.innerHTML = policies.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  sel.value = "ppo";
}

function setAgentStatus(t) { $("#stAgent").textContent = t; }

function scheduleAgent() {
  if (!watching || S.outcome) return;
  const delay = 1400 - +$("#agentSpeed").value; // slider: right = faster
  watchTimer = setTimeout(agentStep, Math.max(60, delay));
}

async function agentStep() {
  if (!watching || S.outcome) return;
  try {
    const resp = await post("/api/agent/step", { policy: $("#agentPolicy").value });
    if (resp.applied) {
      const a = resp.applied;
      if (a.kind === K.DRAW) sfx.draw();
      else if (a.kind === K.T2F || a.kind === K.W2F) sfx.found();
      else if (a.cards && a.cards.length) sfx.flip();
    }
    apply(resp, true);
    if (resp.applied && resp.applied.cards) {
      resp.applied.cards.forEach((c) => {
        const n = document.querySelector(`.card[data-card="${c}"]`);
        if (n) { n.classList.add("moving"); setTimeout(() => n.classList.remove("moving"), 700); }
      });
    }
    scheduleAgent();
  } catch (e) {
    stopAgent();
    toast(e.message);
  }
}

function startAgent() {
  watching = true;
  $("#agentToggle").textContent = "⏸ Stop";
  $("#agentToggle").classList.add("watching");
  scheduleAgent();
}
function stopAgent() {
  watching = false;
  clearTimeout(watchTimer);
  $("#agentToggle").textContent = "▶ Watch";
  $("#agentToggle").classList.remove("watching");
}

/* ---------------- wiring ---------------- */

$("#newGame").onclick = newDeal;
$("#undo").onclick = async () => { try { apply(await post("/api/undo")); } catch (e) { toast(e.message); } };
$("#concede").onclick = async () => { try { apply(await post("/api/concede")); } catch (e) { toast(e.message); } };
$("#hint").onclick = async () => {
  if (watching) return;
  try {
    const { suggested } = await post("/api/agent/hint", { policy: $("#agentPolicy").value });
    if (suggested.concede) { toast(`${$("#agentPolicy").selectedOptions[0].text}: would concede here`); return; }
    (suggested.cards || []).forEach((c) => {
      const n = document.querySelector(`.card[data-card="${c}"]`);
      if (n) { n.classList.add("hintcard"); setTimeout(() => n.classList.remove("hintcard"), 1600); }
    });
    toast(`${$("#agentPolicy").selectedOptions[0].text.split("—")[0].trim()}: ${suggested.kind_name}${suggested.dst >= 0 && (suggested.kind === 1 || suggested.kind === 4) ? " → col " + (suggested.dst + 1) : ""}`);
  } catch (e) { toast(e.message); }
};
$("#agentToggle").onclick = () => (watching ? stopAgent() : startAgent());
$("#agentSpeed").oninput = () => { if (watching) { clearTimeout(watchTimer); scheduleAgent(); } };
$("#auto").onclick = () => {
  if (watching) return;
  $("#agentPolicy").value = "heuristic";
  $("#agentSpeed").value = 1200;
  startAgent();
};
$("#mute").onclick = () => {
  $("#mute").textContent = sfx.toggle() ? "🔇" : "🔊";
  if (!sfx.muted) sfx.found();
};

/* keyboard: Space=draw, U=undo, H=hint, N=new deal, Esc=stop agent */
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input,select") || !S) return;
  if (e.key === " " && !S.outcome && !watching) { e.preventDefault(); doMove({ kind: K.DRAW }); }
  else if (e.key === "u" || e.key === "U") { $("#undo").click(); }
  else if (e.key === "h" || e.key === "H") { $("#hint").click(); }
  else if (e.key === "n" || e.key === "N") { newDeal(); }
  else if (e.key === "Escape") { stopAgent(); }
});

/* first gesture unlocks WebAudio */
document.addEventListener("pointerdown", () => sfx.unlock(), { once: true });

$("#mute").textContent = sfx.muted ? "🔇" : "🔊";

(async () => {
  await loadPolicies();
  S = await api("/api/state");
  dealStart = Date.now();
  timeTimer = setInterval(() => {
    if (!S.outcome) $("#stTime").textContent = elapsed();
  }, 1000);
  render(true);
})();
