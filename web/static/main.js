/* Time Bomb web assistant — client (specs/001-web-cut-panel).
 *
 * The whole game lives in one localStorage record {setup, events[]}; every entry
 * appends an event and re-sends the record to the stateless /api/panel endpoint.
 * All numbers rendered here come from the response untouched (Constitution I), in
 * fixed seat order with no sorting or highlighting (Constitution II).
 */
"use strict";

const STORAGE_KEY = "timebomb-game";
// UI copy for the official role deal per player count (mirrors the solver's
// NUM_BAD_PRIOR composition; the server is the authority).
const OFFICIAL_DEAL = { 4: "1 or 2", 5: "2", 6: "2", 7: "2 or 3", 8: "3" };

const $ = (id) => document.getElementById(id);

let record = loadRecord();
let pendingCutSeat = null;

/* ---------------- storage ---------------- */

function loadRecord() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

function saveRecord() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
}

function clearRecord() {
  localStorage.removeItem(STORAGE_KEY);
  record = null;
}

/* ---------------- setup screen ---------------- */

function initSetupScreen() {
  const count = $("player-count");
  count.innerHTML = "";
  for (let n = 4; n <= 8; n++) {
    count.add(new Option(`${n} players`, n));
  }
  count.value = "5";
  count.addEventListener("change", rebuildSetupInputs);
  rebuildSetupInputs();

  $("setup-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const n = Number(count.value);
    const names = [];
    for (let i = 0; i < n; i++) names.push($(`name-${i}`).value.trim());
    if (names.some((name) => name === "")) {
      return showSetupError("Every player needs a name.");
    }
    if (new Set(names).size !== names.length) {
      return showSetupError("Player names must be unique.");
    }
    const deal = $("role-deal").value;
    record = {
      setup: {
        players: names,
        bomb: $("bomb-toggle").checked,
        numBadOverride: deal === "official" ? null : Number(deal),
      },
      events: [],
    };
    saveRecord();
    showGame();
  });
}

function rebuildSetupInputs() {
  const n = Number($("player-count").value);
  const box = $("name-inputs");
  const previous = [...box.querySelectorAll("input")].map((el) => el.value);
  box.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const label = document.createElement("label");
    label.textContent = `Player ${i + 1} `;
    const input = document.createElement("input");
    input.type = "text";
    input.id = `name-${i}`;
    input.value = previous[i] ?? "";
    input.placeholder = `Name`;
    label.appendChild(input);
    box.appendChild(label);
  }
  const deal = $("role-deal");
  deal.innerHTML = "";
  deal.add(new Option(`Official deal (${OFFICIAL_DEAL[n]} bad guys)`, "official"));
  for (let b = 1; b <= n - 1; b++) {
    deal.add(new Option(`Exactly ${b}`, b));
  }
}

function showSetupError(message) {
  const err = $("setup-error");
  err.textContent = message;
  err.hidden = false;
}

/* ---------------- game screen ---------------- */

function showSetup() {
  $("setup-screen").hidden = false;
  $("game-screen").hidden = true;
  $("setup-error").hidden = true;
}

function showGame() {
  $("setup-screen").hidden = true;
  $("game-screen").hidden = false;
  refresh();
}

async function refresh() {
  let response;
  try {
    response = await fetch("/api/panel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    });
  } catch {
    return showWarnings([{ message: "Could not reach the assistant — is the server running?" }]);
  }
  const body = await response.json();
  if (!response.ok) {
    return showWarnings([{ message: `Rejected entry: ${body.error}` }]);
  }
  render(body);
}

function render(body) {
  const { state, belief, warnings } = body;
  const players = record.setup.players;
  renderStatus(state);
  renderBanner(state);
  showWarnings(warnings);
  renderBelief(state, belief, players);
  renderEntryForms(state, players);
}

function renderStatus(state) {
  $("status-bar").textContent =
    `Round ${state.round} · hands of ${state.handSize} · ` +
    `${state.activeWires} wire${state.activeWires === 1 ? "" : "s"} left · ` +
    `cut ${state.cutsMade}/${state.cutsThisRound}`;
}

function renderBanner(state) {
  const banner = $("banner");
  if (!state.gameOver) {
    banner.hidden = true;
    return;
  }
  const text = {
    bomb: "The bomb was detonated — bad guys win!",
    time: "Out of time — bad guys win!",
    wires: "All wires cut — good guys win!",
  }[state.gameOver.reason];
  banner.textContent = text;
  banner.className = state.gameOver.winner === "good" ? "banner good" : "banner bad";
  banner.hidden = false;
}

function showWarnings(warnings) {
  const box = $("warnings");
  box.innerHTML = "";
  for (const warning of warnings ?? []) {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = warning.message;
    box.appendChild(p);
  }
}

function renderBelief(state, belief, players) {
  const table = $("panel-table");
  const chip = $("num-bad-chip");
  const note = $("panel-note");
  if (!belief) {
    table.hidden = true;
    chip.hidden = true;
    note.hidden = true;
    return;
  }

  const counts = Object.keys(belief.pNumBad);
  if (counts.length > 1) {
    chip.textContent = "Bad guys: " + counts
      .map((b) => `${b} (${formatPct(belief.pNumBad[b])})`)
      .join(" · ");
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }

  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  players.forEach((name, i) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = name;
    const pBadCell = row.insertCell();
    pBadCell.textContent = formatPct(belief.pBad[i]);
    if (!belief.panel) {
      row.insertCell().colSpan = 3;
      return;
    }
    const panelRow = belief.panel[i];
    if (panelRow.noCards) {
      const cell = row.insertCell();
      cell.colSpan = 3;
      cell.className = "no-cards";
      cell.textContent = "no face-down cards left";
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

  if (belief.panel && belief.approx) {
    note.textContent = `Info stat is approximate (lookahead capped at depth ${belief.maxDepth}).`;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function renderEntryForms(state, players) {
  const declForm = $("decl-form");
  const cutForm = $("cut-form");
  declForm.hidden = state.awaiting !== "declarations";
  cutForm.hidden = state.awaiting !== "cut";
  if (state.awaiting === "declarations") {
    buildDeclForm(state, players);
  } else if (state.awaiting === "cut") {
    buildCutForm(state, players);
  }
}

function buildDeclForm(state, players) {
  $("decl-title").textContent =
    `Round ${state.round} declarations (0–${state.handSize} wires each)`;
  const box = $("decl-inputs");
  box.innerHTML = "";
  players.forEach((name, i) => {
    const label = document.createElement("label");
    label.textContent = `${name} `;
    const input = document.createElement("input");
    input.type = "number";
    input.id = `decl-${i}`;
    input.min = "0";
    input.max = String(state.handSize);
    input.step = "1";
    input.inputMode = "numeric";
    input.required = true;
    label.appendChild(input);
    box.appendChild(label);
  });
  $("decl-error").hidden = true;
}

function buildCutForm(state, players) {
  $("cut-title").textContent = "Whose wire was cut?";
  const box = $("cut-players");
  box.innerHTML = "";
  players.forEach((name, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.disabled = state.revealed[i] >= state.handSize;
    button.addEventListener("click", () => {
      pendingCutSeat = i;
      $("cut-chosen").textContent = `${name}'s card showed…`;
      $("cut-results").hidden = false;
    });
    box.appendChild(button);
  });
  $("bomb-result").hidden = !record.setup.bomb;
  $("cut-results").hidden = true;
  pendingCutSeat = null;
}

/* ---------------- entries ---------------- */

function appendEvent(event) {
  record.events.push(event);
  saveRecord();
  refresh();
}

function initGameHandlers() {
  $("decl-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const n = record.setup.players.length;
    const values = [];
    for (let i = 0; i < n; i++) {
      const value = Number($(`decl-${i}`).value);
      values.push(value);
    }
    appendEvent({ type: "declarations", values });
  });

  $("cut-results").addEventListener("click", (ev) => {
    const result = ev.target.dataset?.result;
    if (result && pendingCutSeat !== null) {
      appendEvent({ type: "cut", player: pendingCutSeat, result });
    }
  });
  $("cut-cancel").addEventListener("click", () => {
    $("cut-results").hidden = true;
    pendingCutSeat = null;
  });

  $("new-game-button").addEventListener("click", () => {
    if (!confirm("Abandon this game and start a new one?")) return;
    clearRecord();
    showSetup();
  });
}

/* ---------------- helpers ---------------- */

function formatPct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

/* ---------------- boot ---------------- */

initSetupScreen();
initGameHandlers();
if (record) {
  showGame(); // reload survival: replay the stored record (SC-005)
} else {
  showSetup();
}
