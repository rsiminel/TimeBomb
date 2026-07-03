/* Time Bomb hosted game — setup, tabletop, and the polling/version flow
 * (specs/002-host-local-game, T016/T017/T018).
 *
 * Transport only: every rule decision is the engine's. The client renders the
 * TableView from GET /api/game, forwards intents with the version it saw, and
 * treats 409 as "the table moved on — refetch". Private info is only ever held
 * after an explicit unlock and dropped on any accepted intent or lock. */

"use strict";

const THEME_KEY = "timebomb-theme";
const MIN_SEATS = 4;
const MAX_SEATS = 8;
// UI copy for the official role deal per seat count (the server is the authority).
const OFFICIAL_DEAL = { 4: "1 or 2", 5: "2", 6: "2", 7: "2 or 3", 8: "3" };
const POLL_MS = 1000;

const CLAIM_LABELS = {
  trust: "I trust…",
  distrust: "I don't trust…",
  accuse_lie: "…lied about their wires",
  self_honest: "My declaration is honest",
};
const DIRECTED_CLAIMS = ["trust", "distrust", "accuse_lie"];

const $ = (id) => document.getElementById(id);

let view = null;        // latest TableView from GET /api/game
let priv = null;        // PrivateView after unlock, or null (locked)
let pollTimer = null;
let seenCuts = 0;       // cutLog length already rendered (drives the flip)
let seenClaims = 0;
let pendingSetup = null; // setup JSON held while the 409 conflict box is up
let selectedCutTarget = null;

/* ---------------- transport ---------------- */

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  let data = null;
  try { data = await resp.json(); } catch { /* 204s have no body */ }
  return { status: resp.status, data };
}

async function refresh() {
  const { status, data } = await api("GET", "/api/game");
  if (status === 404) {
    view = null;
    priv = null;
    showSetup();
    return;
  }
  const unchanged = view !== null && data.version === view.version;
  // A resumed finished save opens straight into the replay (contracts/api.md).
  if (view === null && data.phase === "finished") replayOpen = true;
  view = data;
  if (priv !== null && view.unlockedSeat !== priv.myIndex) priv = null; // re-locked
  // Skip the re-render on a no-op poll tick, so it can't wipe a form mid-typing.
  if (!unchanged || $("game-screen").hidden) renderGame();
  schedulePoll();
}

/* Poll while the server can change state without us: an AI seat is pending or
 * thinking, or an exhibition is playing itself out. A lone human's turn is
 * quiescent — nothing to poll for. */
function schedulePoll() {
  clearTimeout(pollTimer);
  if (!view || view.phase === "finished") return;
  const aiPending = view.pending &&
    view.pending.seats.some((s) => view.occupants[s] !== "human");
  if (aiPending || view.thinking.length > 0) {
    pollTimer = setTimeout(refresh, POLL_MS);
  }
}

/* ---------------- setup screen (T016) ---------------- */

function showSetup() {
  $("setup-screen").hidden = false;
  $("game-screen").hidden = true;
  $("conflict-box").hidden = true;
  $("setup-error").hidden = true;
}

function initSetup() {
  const count = $("seat-count");
  for (let n = MIN_SEATS; n <= MAX_SEATS; n++) {
    count.add(new Option(`${n} players`, n));
  }
  count.value = "5";
  count.addEventListener("change", renderSeatRows);
  renderSeatRows();

  $("setup-form").addEventListener("submit", onSetupSubmit);
  $("return-to-game").addEventListener("click", () => {
    pendingSetup = null;
    refresh();
  });
  $("abandon-and-start").addEventListener("click", async () => {
    await api("DELETE", "/api/game");
    if (pendingSetup) await createGame(pendingSetup);
  });
}

function renderSeatRows() {
  const n = Number($("seat-count").value);
  const rows = $("seat-rows");
  // Keep whatever the user already typed in surviving rows.
  while (rows.children.length > n) rows.lastChild.remove();
  while (rows.children.length < n) {
    const i = rows.children.length;
    const row = document.createElement("div");
    row.className = "seat-row";
    const name = document.createElement("input");
    name.type = "text";
    name.value = `Player ${i + 1}`;
    name.required = true;
    name.maxLength = 20;
    const kind = document.createElement("select");
    kind.add(new Option("Human", "human"));
    kind.add(new Option("Solver bot", "solver_bot"));
    kind.value = i === 0 ? "human" : "solver_bot";
    row.append(name, kind);
    rows.append(row);
  }

  const deal = $("role-deal");
  deal.replaceChildren(new Option(`Official (${OFFICIAL_DEAL[n]})`, ""));
  for (let b = 1; b <= n - 1; b++) {
    deal.add(new Option(`Exactly ${b}`, b));
  }
}

function readSetup() {
  const seats = [...$("seat-rows").children].map((row) => ({
    name: row.children[0].value.trim(),
    occupant: row.children[1].value,
  }));
  const setup = { seats, panel_allowed: $("panel-optin").checked };
  if ($("role-deal").value !== "") setup.role_deal = Number($("role-deal").value);
  if ($("seed-input").value !== "") setup.seed = Number($("seed-input").value);
  return setup;
}

async function onSetupSubmit(event) {
  event.preventDefault();
  await createGame(readSetup());
}

async function createGame(setup) {
  const { status, data } = await api("POST", "/api/game", setup);
  if (status === 201) {
    pendingSetup = null;
    seenCuts = 0;
    seenClaims = 0;
    replayOpen = false;
    replayCursor = null;
    panelVersion = 0;
    $("last-cut").hidden = true;
    await refresh();
    return;
  }
  if (status === 409) {
    pendingSetup = setup;
    $("conflict-box").hidden = false;
    return;
  }
  const err = $("setup-error");
  err.textContent = (data && data.error) || `Setup failed (${status})`;
  err.hidden = false;
}

/* ---------------- table screen (T017) ---------------- */

function renderGame() {
  $("setup-screen").hidden = true;
  $("game-screen").hidden = false;

  renderStatus();
  renderSeats();
  renderTicker();
  renderBanner();
  renderTurnPanel();
  renderPanelDrawer();
  if (view.phase === "finished") renderReplay();
}

function renderStatus() {
  const roundNo = view.roundIndex + 1;
  const phaseText = {
    awaiting_declarations: "declarations",
    awaiting_cut: `${view.playerNames[view.currentCutter]} holds the cutters`,
    finished: "game over",
  }[view.phase] || view.phase;
  $("status-bar").textContent =
    `Round ${roundNo} · ${view.handSize} cards each · ` +
    `${view.activeWires} wire${view.activeWires === 1 ? "" : "s"} to find · ${phaseText}`;
}

function roleBadge(role) {
  const span = document.createElement("span");
  span.className = `role-badge ${role === 1 ? "bad" : "good"}`;
  span.textContent = role === 1 ? "BAD" : "GOOD";
  return span;
}

function renderSeats() {
  const felt = $("table-wrap");
  felt.querySelectorAll(".seat").forEach((el) => el.remove());
  const n = view.numPlayers;
  const roles = view.reveal ? view.reveal.roles : null;
  const newestCut = view.cutLog.length > seenCuts
    ? view.cutLog[view.cutLog.length - 1] : null;

  for (let i = 0; i < n; i++) {
    const seat = document.createElement("div");
    seat.className = "seat";
    // Around the circle, seat 0 at the bottom (where the device holder sits).
    const angle = (2 * Math.PI * i) / n + Math.PI / 2;
    seat.style.left = `${50 + 42 * Math.cos(angle)}%`;
    seat.style.top = `${50 + 42 * Math.sin(angle)}%`;
    if (view.pending && view.pending.seats.includes(i)) seat.classList.add("pending");

    const name = document.createElement("div");
    name.className = "seat-name";
    name.textContent =
      (view.phase !== "finished" && i === view.currentCutter ? "✂ " : "") +
      view.playerNames[i] +
      (view.occupants[i] === "human" ? "" : " 🤖");
    seat.append(name);

    const sub = document.createElement("div");
    sub.className = "seat-sub";
    if (roles) sub.append(roleBadge(roles[i]));
    if (view.thinking.includes(i)) {
      const think = document.createElement("span");
      think.className = "thinking";
      think.textContent = " thinking";
      sub.append(think);
    } else if (view.declarations[i] !== null && view.declarations[i] !== undefined) {
      sub.append(roles ? ` · says ${view.declarations[i]}` : `says ${view.declarations[i]}`);
    }
    seat.append(sub);

    // This round's hand: found wires (green), duds (grey), then face-down backs.
    const cards = document.createElement("div");
    cards.className = "cards";
    const wires = view.found[i];
    const duds = view.revealed[i] - view.found[i];
    const facedown = view.handSize - view.revealed[i];
    for (let c = 0; c < wires; c++) cards.append(miniCard("cut-wire"));
    for (let c = 0; c < duds; c++) cards.append(miniCard("cut-dud"));
    for (let c = 0; c < facedown; c++) cards.append(miniCard(""));
    // Animate the just-revealed card on this seat. A cut bomb is counted in
    // `revealed` but not `found`, so it's the last "dud" slot — recolor it.
    if (newestCut && newestCut.target === i && newestCut.round === view.roundIndex) {
      const idx = newestCut.result === "wire" ? wires - 1 : wires + duds - 1;
      const el = cards.children[idx];
      if (el) {
        if (newestCut.result === "bomb") el.className = "card-mini cut-bomb";
        el.classList.add("flip");
      }
    }
    seat.append(cards);
    felt.append(seat);
  }
  seenCuts = view.cutLog.length;
}

function miniCard(cls) {
  const el = document.createElement("div");
  el.className = `card-mini ${cls}`;
  return el;
}

/* One-line ticker under the table: the latest cut and/or claim. */
function renderTicker() {
  const lines = [];
  if (view.cutLog.length > 0) {
    const cut = view.cutLog[view.cutLog.length - 1];
    const what = { wire: "a safe wire 🟢", dud: "nothing", bomb: "💥 THE BOMB" }[cut.result];
    lines.push(`${view.playerNames[cut.cutter]} cut ${view.playerNames[cut.target]}: ${what}.`);
  }
  if (view.claimLog.length > 0) {
    const claim = view.claimLog[view.claimLog.length - 1];
    const who = view.playerNames[claim.speaker];
    const target = claim.target === null ? "" : view.playerNames[claim.target];
    const text = {
      trust: `I trust ${target}`,
      distrust: `I don't trust ${target}`,
      accuse_lie: `${target} lied about their wires`,
      self_honest: "my declaration is honest",
    }[claim.kind];
    lines.push(`${who} says: “${text}”.`);
  }
  seenClaims = view.claimLog.length;
  $("last-cut").textContent = lines.join(" ");
  $("last-cut").hidden = lines.length === 0;
}

function renderBanner() {
  const banner = $("banner");
  if (view.phase !== "finished") {
    banner.hidden = true;
    $("new-game-button").hidden = true;
    $("reveal-line").hidden = true;
    $("replay-toggle").hidden = true;
    $("replay-box").hidden = true;
    return;
  }
  const outcome = view.reveal.outcome;
  const goodWon = outcome.good_guys_won;
  banner.className = `banner ${goodWon ? "good" : "bad"}`;
  banner.textContent = goodWon
    ? `🎉 The good team wins — ${outcome.reason}!`
    : `💥 The bad team wins — ${outcome.reason}!`;
  banner.hidden = false;
  $("new-game-button").hidden = false;

  const badSeats = view.reveal.roles
    .map((role, i) => (role === 1 ? view.playerNames[i] : null))
    .filter((name) => name !== null);
  $("reveal-line").textContent = `The bad guy${badSeats.length === 1 ? " was" : "s were"}: ` +
    badSeats.join(", ") + ".";
  $("reveal-line").hidden = false;
  $("replay-toggle").hidden = false;
}

/* ---------------- turn panel: pass-the-device + action forms ---------------- */

function humanTurnSeat() {
  if (!view.pending) return null;
  const humans = view.pending.seats.filter((s) => view.occupants[s] === "human");
  return humans.length > 0 ? humans[0] : null;
}

function renderTurnPanel() {
  const panel = $("turn-panel");
  const waiting = $("waiting-note");
  const seat = view.phase === "finished" ? null : humanTurnSeat();

  if (seat === null) {
    panel.hidden = true;
    $("handoff-overlay").hidden = true;
    if (view.phase === "finished") {
      waiting.hidden = true;
    } else {
      waiting.textContent = "Waiting on the AI seats…";
      waiting.hidden = false;
    }
    return;
  }
  waiting.hidden = true;
  panel.hidden = false;

  const unlocked = priv !== null && priv.myIndex === seat;
  const humans = view.occupants.filter((o) => o === "human").length;
  // With several humans on the device, the handoff masks the whole screen
  // (FR-009); a lone human just gets the inline tap-to-reveal.
  const useOverlay = humans > 1 && !unlocked;
  $("handoff-overlay").hidden = !useOverlay;
  $("pass-prompt").hidden = unlocked || useOverlay;
  $("action-box").hidden = !unlocked;

  if (!unlocked) {
    const name = view.playerNames[seat];
    const verb = view.pending.kind === "declare" ? "declare" : "cut a wire";
    if (useOverlay) {
      $("handoff-text").textContent =
        `Pass the device to ${name} — it's their turn to ${verb}.`;
      $("handoff-reveal").textContent = `I'm ${name} — show my hand`;
      $("handoff-reveal").onclick = () => unlockSeat(seat);
    } else {
      $("pass-text").textContent = `It's your turn to ${verb}, ${name}.`;
      $("show-hand").textContent = `I'm ${name} — show my hand`;
      $("show-hand").onclick = () => unlockSeat(seat);
    }
    return;
  }

  $("private-card").innerHTML = "";
  const roleLine = document.createElement("div");
  roleLine.append("You are ");
  roleLine.append(roleBadge(priv.myRole));
  roleLine.append(priv.myRole === 1
    ? " — you win if the bomb goes off or time runs out."
    : " — find every safe wire before time runs out.");
  const handLine = document.createElement("div");
  handLine.textContent =
    `Your hand this round holds ${priv.myWires} safe wire${priv.myWires === 1 ? "" : "s"}` +
    (priv.iHoldBomb ? " — and THE BOMB." : ".");
  $("private-card").append(roleLine, handLine);

  const declaring = view.pending.kind === "declare";
  $("declare-form").hidden = !declaring;
  $("cut-form").hidden = declaring;
  if (declaring) {
    $("declare-value").max = view.handSize;
    $("declare-error").hidden = true;
    fillClaimMenu("declare", seat);
  } else {
    renderCutTargets(seat);
    $("cut-error").hidden = true;
    fillClaimMenu("cut", seat);
  }
}

async function unlockSeat(seat) {
  const { status, data } = await api("POST", "/api/game/unlock",
                                     { seat, version: view.version });
  if (status === 200) {
    priv = data;
    $("declare-value").value = "";
    selectedCutTarget = null;
    renderGame();
  } else {
    await refresh(); // 409 stale tab or 403 — the fresh view explains itself
  }
}

async function lockSeat() {
  await api("POST", "/api/game/lock");
  priv = null;
  renderGame();
}

function renderCutTargets(seat) {
  const box = $("cut-targets");
  box.replaceChildren();
  selectedCutTarget = null;
  $("cut-submit").disabled = true;
  for (const target of view.legalTargets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = view.playerNames[target];
    btn.addEventListener("click", () => {
      selectedCutTarget = target;
      box.querySelectorAll("button").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      $("cut-submit").disabled = false;
    });
    box.append(btn);
  }
}

function fillClaimMenu(prefix, seat) {
  const kindSel = $(`${prefix}-claim-kind`);
  const targetRow = $(`${prefix}-claim-target-row`);
  const targetSel = $(`${prefix}-claim-target`);
  // Idempotent per seat-and-table, so a mid-typing re-render can't reset the
  // selection (but a new game with different names does rebuild).
  const stamp = `${seat}:${view.playerNames.join(",")}`;
  if (kindSel.dataset.stamp === stamp) return;
  kindSel.dataset.stamp = stamp;
  kindSel.replaceChildren(new Option("Nothing", ""));
  for (const kind of Object.keys(CLAIM_LABELS)) {
    kindSel.add(new Option(CLAIM_LABELS[kind], kind));
  }
  targetSel.replaceChildren();
  for (let i = 0; i < view.numPlayers; i++) {
    if (i !== seat) targetSel.add(new Option(view.playerNames[i], i));
  }
  kindSel.onchange = () => {
    targetRow.hidden = !DIRECTED_CLAIMS.includes(kindSel.value);
  };
  targetRow.hidden = true;
}

function readClaim(prefix) {
  const kind = $(`${prefix}-claim-kind`).value;
  if (kind === "") return null;
  if (!DIRECTED_CLAIMS.includes(kind)) return { kind };
  return { kind, target: Number($(`${prefix}-claim-target`).value) };
}

/* ---------------- post-game replay (T031, FR-025) ---------------- */

let replayOpen = false;
let replayCursor = null;   // index of the last event shown; null = full history

function replayEvents() {
  // Skip bookkeeping-only entries; every remaining event reads as a story beat.
  return view.reveal.events.filter((e) => e.type !== "round_end");
}

function describeEvent(e) {
  const name = (i) => view.playerNames[i];
  switch (e.type) {
    case "game_start":
      return { text: `Game start — ${e.num_players} players.` };
    case "round_start": {
      const hands = e.wires
        .map((w, i) => `${name(i)}: ${w}${e.bombs[i] ? " +💣" : ""}`)
        .join(", ");
      return { text: `Round ${e.round + 1} — ${e.hand_size} cards each. ` +
                     `Dealt wires: ${hands}.` };
    }
    case "declaration":
      return { text: `${name(e.player)} declared ${e.declared}`, decl: e };
    case "claim": {
      const target = e.target === null ? "" : name(e.target);
      const text = {
        trust: `I trust ${target}`,
        distrust: `I don't trust ${target}`,
        accuse_lie: `${target} lied about their wires`,
        self_honest: "my declaration is honest",
      }[e.kind];
      return { text: `${name(e.speaker)} says: “${text}”.` };
    }
    case "cut": {
      const what = { wire: "a safe wire 🟢", dud: "nothing",
                     bomb: "💥 THE BOMB" }[e.result];
      return { text: `${name(e.cutter)} cut ${name(e.target)}: ${what}.` };
    }
    case "cut_skipped":
      return { text: `${name(e.cutter)} had no legal target — the round ends early.` };
    case "game_end":
      return { text: e.good_guys_won
        ? `🎉 The good team wins — ${e.reason}.`
        : `💥 The bad team wins — ${e.reason}.` };
    default:
      return { text: e.type };
  }
}

function renderReplay() {
  $("replay-box").hidden = !replayOpen;
  if (!replayOpen) return;
  const events = replayEvents();
  if (replayCursor === null || replayCursor >= events.length) {
    replayCursor = events.length - 1;
  }
  $("replay-pos").textContent = `${replayCursor + 1} / ${events.length}`;
  $("replay-prev").disabled = replayCursor <= 0;
  $("replay-next").disabled = replayCursor >= events.length - 1;

  const log = $("replay-log");
  log.replaceChildren();
  for (let i = 0; i <= replayCursor; i++) {
    const item = document.createElement("li");
    const { text, decl } = describeEvent(events[i]);
    item.append(text);
    if (decl) {
      item.append(" — ");
      const badge = document.createElement("span");
      if (decl.lie) {
        badge.className = "lie-badge";
        badge.textContent = `LIED (had ${decl.true_wires})`;
      } else {
        badge.className = "truth-badge";
        badge.textContent = "truthful";
      }
      item.append(badge);
    }
    if (i === replayCursor) item.className = "current";
    log.append(item);
  }
  log.lastChild?.scrollIntoView({ block: "nearest" });
}

function initReplay() {
  $("replay-toggle").addEventListener("click", () => {
    replayOpen = !replayOpen;
    replayCursor = null;     // (re)open on the full history
    renderReplay();
  });
  $("replay-prev").addEventListener("click", () => {
    replayCursor = Math.max(0, replayCursor - 1);
    renderReplay();
  });
  $("replay-next").addEventListener("click", () => {
    replayCursor += 1;
    renderReplay();
  });
}

/* ---------------- opt-in stats drawer (T028, FR-019/020/021) ---------------- */

let panelOpen = false;
let panelBusy = false;
let panelVersion = 0;   // game version the drawer currently shows

function renderPanelDrawer() {
  $("panel-toggle").hidden = !view.panelAllowed;
  $("panel-drawer").hidden = !(view.panelAllowed && panelOpen);
  if (view.panelAllowed && panelOpen) maybeFetchPanel();
}

/* Refetch only when the table moved (the drawer rides the poll cycle) and never
 * concurrently — a 7-8 player panel can take the solver a couple of seconds. */
async function maybeFetchPanel() {
  if (panelBusy || panelVersion === view.version) return;
  panelBusy = true;
  const fetchedAt = view.version;
  const { status, data } = await api("GET", "/api/game/panel");
  panelBusy = false;
  if (status !== 200) return;
  panelVersion = fetchedAt;
  renderPanelBody(data.belief);
  if (panelVersion !== view.version) maybeFetchPanel(); // moved again meanwhile
}

function formatPct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

function renderPanelBody(belief) {
  const table = $("panel-table");
  const chip = $("panel-chip");
  const note = $("panel-note");
  if (!belief) {
    table.hidden = true;
    chip.hidden = true;
    note.textContent = "Nothing to show yet — the panel wakes up after the first " +
      "round's declarations.";
    note.hidden = false;
    return;
  }

  const counts = Object.keys(belief.pNumBad);
  chip.hidden = counts.length <= 1;
  if (counts.length > 1) {
    chip.textContent = "Bad guys: " + counts
      .map((b) => `${b} (${formatPct(belief.pNumBad[b])})`)
      .join(" · ");
  }

  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  view.playerNames.forEach((name, i) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = name;
    row.insertCell().textContent = formatPct(belief.pBad[i]);
    if (!belief.panel) {
      const cell = row.insertCell();
      cell.colSpan = 3;
      return;
    }
    const panelRow = belief.panel[i];
    if (panelRow.noCards) {
      const cell = row.insertCell();
      cell.colSpan = 3;
      cell.className = "no-cards";
      cell.textContent = "no cards left";
      return;
    }
    row.insertCell().textContent = formatPct(panelRow.pBomb);
    row.insertCell().textContent = formatPct(panelRow.pSafe);
    const info = row.insertCell();
    info.textContent = (belief.approx ? "≈" : "") + panelRow.horizon.toFixed(3);
    info.title = `1-ply: ${panelRow.onePly.toFixed(3)} — expected remaining role ` +
      `entropy after cutting here; lower teaches the table more`;
  });
  table.hidden = false;

  if (!belief.panel) {
    note.textContent = "Between rounds — per-cut stats return with the next " +
      "declarations.";
    note.hidden = false;
  } else if (belief.approx) {
    note.textContent =
      `Info stat is approximate (lookahead capped at depth ${belief.maxDepth}).`;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function initPanelDrawer() {
  $("panel-toggle").addEventListener("click", () => {
    panelOpen = !panelOpen;
    renderPanelDrawer();
  });
  $("panel-close").addEventListener("click", () => {
    panelOpen = false;
    renderPanelDrawer();
  });
}

/* ---------------- intents (T018 version flow) ---------------- */

async function submitIntent(seat, kind, value, claim, errorEl) {
  const { status, data } = await api("POST", "/api/game/intent",
                                     { seat, kind, value, claim, version: view.version });
  if (status === 200) {
    priv = null; // accepted intents re-lock privacy (contracts/api.md)
    await refresh();
    return;
  }
  if (status === 409) {
    await refresh(); // stale tab: the table moved on; re-render current truth
    return;
  }
  errorEl.textContent = (data && data.error) || `Rejected (${status})`;
  errorEl.hidden = false;
}

function initGameScreen() {
  $("declare-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const value = Number($("declare-value").value);
    submitIntent(priv.myIndex, "declare", value, readClaim("declare"), $("declare-error"));
  });
  $("cut-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (selectedCutTarget === null) return;
    submitIntent(priv.myIndex, "cut", selectedCutTarget, readClaim("cut"), $("cut-error"));
  });
  document.querySelectorAll(".hide-hand").forEach((btn) =>
    btn.addEventListener("click", lockSeat));

  $("new-game-button").addEventListener("click", async () => {
    await api("DELETE", "/api/game");
    showSetup();
  });
  $("abandon-button").addEventListener("click", async () => {
    if (!confirm("Abandon this game? Progress since the last save is lost.")) return;
    await api("DELETE", "/api/game");
    showSetup();
  });
  $("save-button").addEventListener("click", async () => {
    const name = prompt("Name this save:");
    if (name === null || name.trim() === "") return;
    const { status, data } = await api("POST", "/api/saves", { name: name.trim() });
    const note = $("save-note");
    note.textContent = status === 201
      ? `Saved as “${name.trim()}” — resumable from the home page.`
      : (data && data.error) || `Save failed (${status})`;
    note.hidden = false;
  });

  // A backgrounded tab stops polling; catch up the moment it returns.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && view) refresh();
  });
}

/* ---------------- theme (same pattern as the v1 assistant) ---------------- */

function applyTheme(theme) {
  if (theme === "dark" || theme === "light") {
    document.documentElement.dataset.theme = theme;
  } else {
    delete document.documentElement.dataset.theme; // follow the OS preference
  }
  const dark = theme === "dark" ||
    (!theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("theme-toggle").textContent = dark ? "Light" : "Dark";
}

function initTheme() {
  applyTheme(localStorage.getItem(THEME_KEY));
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme &&
        matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });
}

/* ---------------- boot ---------------- */

initTheme();
initSetup();
initGameScreen();
initPanelDrawer();
initReplay();
// Page load re-locks all private views (FR-009: fresh eyes on the screen), then
// rejoins the running game if there is one (FR-024) or shows setup on 404.
api("POST", "/api/game/lock").finally(refresh);
