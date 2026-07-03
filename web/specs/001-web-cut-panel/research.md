# Research: Time Bomb Web Cut Panel

Decisions resolving every open technical question in plan.md's Technical Context.

## R1 — Server framework: Flask

- **Decision**: Flask, serving both the static page and `POST /api/panel`, run directly
  from the repo's `.venv`.
- **Rationale**: the solver is Python, so the server must be Python; Flask is the
  smallest mainstream option, was already the shape of the old `web/app.py`, and one
  endpoint needs nothing more. `flask-cors` is dropped — same-origin serving makes it
  unnecessary.
- **Alternatives considered**: FastAPI (async + pydantic buy nothing for one local
  endpoint); `http.server` (no request routing/JSON ergonomics, false economy);
  Pyodide running the solver in-browser (violates the one-solver principle in spirit —
  a second runtime for the same math — and balloons page weight).

## R2 — State model: client-held event log, stateless server

- **Decision**: the browser keeps `{setup, events[]}` in `localStorage` and sends the
  whole record on every entry; the server replays it and returns derived state +
  belief. No server-side sessions.
- **Rationale**: full-history undo (= truncate), reload survival (= replay), and
  restart-proofness (server holds nothing) all come from one mechanism; replay cost is
  low (R4). Testing gets a pure function `record → response`.
- **Alternatives considered**: server-side session state (loses undo/reload for free,
  adds lifecycle bugs); incremental client-side belief caching (splits the source of
  truth, invites drift).

## R3 — Orchestration is a mirror of `General.Play`

- **Decision**: the replay reproduces the interactive `Play` loop's call pattern
  exactly — per cut: `ProbDeclaration` → `ProbCut` on the round's cumulative
  `revealed/found`; per completed round: accumulate `RoundLogU` per candidate bad
  count; belief readout: `JointBadBelief` with the current round folded in
  provisionally; panel: `CutPanel` at `candidate_bs[0]` (matching `PrintPanel` usage);
  feasibility warnings via `Consistency` exactly as `Play` does (once per round; the
  impossible-cut signal is `ProbCut` returning its prior).
- **Rationale**: `Play` is the solver author's reference orchestration; copying its
  call pattern (not its code) means the web layer embodies zero modelling decisions of
  its own — Constitution I at the orchestration level, not just the formula level.
- **Alternatives considered**: bespoke orchestration (every divergence from `Play` is
  an opportunity to be subtly wrong); importing and screen-scraping `Play` itself
  (interactive `input()`/`print` loop, not callable).

## R4 — Replay cost

- **Decision**: recompute from scratch per request; no caching.
- **Rationale**: per replay, completed rounds cost one `RoundLogU` per candidate bad
  count (≤2 candidates × ≤3 rounds) and the current round costs one
  `ProbDeclaration` + one `ProbCut` + one `CutPanel` — a constant handful of solver
  calls regardless of how many cuts were entered. The panel's round-horizon stat
  dominates everything else (R7); caching would optimise the cheap part.

## R5 — Frontend: vanilla JS, no build step

- **Decision**: one `index.html`, one `main.js` (ES modules), one `styles.css`;
  phone-first layout; no framework, no bundler, no npm.
- **Rationale**: one page with two screens (setup, game) and one fetch call does not
  justify a toolchain; the repo has no Node infrastructure and the constitution asks
  for dependency-light.
- **Alternatives considered**: React/Vite (build pipeline for one table); htmx
  (server-rendered rounds fight the client-held event log).

## R6 — Web test suite lives in `web/tests/`, oracle is the live solver

- **Decision**: pytest + Flask test client under `web/tests/` with its own
  `conftest.py` (adds the `timebomb/` source root to `sys.path`). Parity tests script
  whole games and assert the API's numbers equal direct `General.py` calls exactly.
  Replay/bookkeeping tests (rounds, budgets, game end, undo, warnings, 422s) test the
  web layer's own logic against hand-worked expectations.
- **Rationale**: Constitution III forbids touching the repo-root `tests/`; the
  constitution's testing constraint says web tests verify *presentation against live
  solver output*, not the math — so unlike backend tests, the implementation (the
  solver) **is** the oracle here, by design.
- **Alternatives considered**: browser e2e (Playwright) — deferred; the JS layer is
  thin (render + fetch + localStorage) and quickstart.md covers it manually. Can be
  added later without design changes.

## R7 — Round-horizon info stat: depth cap with an "approximate" label

- **Decision**: the server calls `CutPanel(..., max_depth=d)` with a per-player-count
  cap chosen so the response beats the 2 s budget (SC-002), starting from the
  backend's own display default `max_depth=3` and reducing for large `N`; the response
  reports `approx: true` whenever the cap bit, and the UI labels the stat. Measured
  `CutPanel` wall time at a worst-case round start (hand 5, all wires live, this
  machine):

  | depth | N=5 | N=8 |
  |-------|--------|---------|
  | 1 | 0.05 s | 1.38 s |
  | 2 | 0.26 s | 10.9 s |
  | 3 | 2.5 s | 156 s |
  | 4 | 25.9 s | 2237 s |

  Each ply costs ~10×. Cap table (confirmed by full-replay measurement during
  implementation): `{4: 3, 5: 2, 6: 2, 7: 1, 8: 1}` — N=4: 0.32 s, N=5: 0.30 s,
  N=6: 0.84 s, N=7: 0.65 s. The contract and UI are cap-agnostic.

  **N=8 floor**: the full replay at cap 1 measures ~2.1 s, and dropping to depth 0
  does not help — the dominant cost is the per-player `ProbCut` sweep that stat 3
  (`NextHBad`) and stat 4's first ply share, which no depth cap removes. Optimising
  that is solver work (off-limits from `web/`, Constitution III), so SC-002 was
  amended instead: 2 s through N=7, 3 s at N=8.
- **Rationale**: the exact lookahead is `O((2N)^stop)` (see `H_Min`) — the table shows
  it is unusable live beyond tiny depths at N=8; the backend already depth-caps for
  display (`PrintPanel` default 3), and the clarification session chose "depth-capped,
  always shown, labeled approximate". The cap, not the budget, gives way.
- **Alternatives considered**: async fill-in and on-demand computation (both rejected
  in the clarification session); beam/analytic approximation (a *backend* roadmap item
  — deferred modelling refinement in TODO.md, not web work).

## R8 — Agent context file update

- **Decision**: skip — this Spec Kit version's `.specify/scripts/bash/` has no
  update-agent-context script, and the repo's `CLAUDE.md` already carries the project
  context the step would inject.
