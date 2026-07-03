# Tasks: Hosted Local Time Bomb Game

**Input**: Design documents from `web/specs/002-host-local-game/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included — the spec's success criteria (SC-003…SC-009) demand automated
verification, and the repo convention is test-backed work. Test tasks come first
within each story; make them fail before implementing.

**Organization**: Grouped by user story (US1–US7 from spec.md) after a foundational
engine-promotion phase that every story depends on.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different files, no dependency on an incomplete task)
- **[Story]**: US1…US7, matching spec.md priorities P1…P7

## Path Conventions

Three source roots at the repo root (`timebomb/` — frozen, `sim/`, and the new
`tbgame/`), plus the web app in `web/`. Paths below are repo-relative.

---

## Phase 1: Setup

**Purpose**: skeleton and hygiene for the new package and web storage

- [ ] T001 Create the `tbgame/` source-root skeleton (`tbgame/`, `tbgame/agents/`, `tbgame/tests/`) and add `tbgame` + `tbgame/agents` to the source-root list in `conftest.py` (repo root — the one shared-infrastructure edit, noted in plan.md)
- [ ] T002 [P] Add `web/saves/` and `web/instance/` to `.gitignore` and create the (empty) directories

---

## Phase 2: Foundational — engine promotion (BLOCKS all user stories)

**Purpose**: the shared rules engine exists, stepwise, with sim/ still green
(constitution v2.0.0 carve-out). ⚠️ No user story work until this phase is done and
the untouched root suite passes.

- [ ] T003 Move the state objects from `sim/state.py` to `tbgame/state.py` (GroundTruth, PublicState, PrivateView, AgentView, EventLog, `legal_targets`) extended per data-model.md (claim_log, phase, pending; a new engine `SCHEMA_VERSION` constant — the arena's stays untouched at its current value); the renderer and prompt assembly stay behind in `sim/state.py`
- [ ] T004 [P] Write engine contract tests in `tbgame/tests/test_table_game.py` — state transitions of data-model.md, `IllegalIntent` on every bad intent shape (wrong seat/kind/range/claim), deal determinism under a fixed seed, `from_events` round-trip equality, reveal only when finished, per contracts/engine.md (independent of T005's internals; run red first)
- [ ] T005 Implement `tbgame/engine.py::TableGame` — stepwise event-sourced core: `__init__(setup)` with SetupError validation (seats 4–8, names, role-deal feasibility, 0-human allowed), `from_events` with SchemaError, `pending`, `submit(intent)` incl. claim riders and automatic round advance / win judgment, `public_state()`, `private_view(seat)`, `reveal()`, instance `random.Random(seed)` for role sample + `General.DistributeWires` deals + cut draws; rule behavior byte-compatible with today's `sim/engine.py`
- [ ] T006 Implement `tbgame/driver.py::Engine` — the blocking arena API (`play_game(agents, seed)`) over `TableGame`: threaded simultaneous declarations, sequential free-text `discuss` → `statement` events, malformed-output fallbacks, `panel_for`/`assistant` adapter, `wire`/`dud`/`bomb` vocabulary, per contracts/engine.md
- [ ] T007 Move `sim/agents/base.py::Agent` to `tbgame/agents/base.py` and add the no-op `claim(view)` hook
- [ ] T008 Convert `sim/engine.py`, `sim/state.py`, `sim/agents/base.py` into re-export shims (renderer/prompt code stays in `sim/state.py`); then run the **unmodified** root suite `.venv/bin/python -m pytest -q` and `tbgame/tests` — all green (if `tests/test_arena_engine.py` proves to depend on global-RNG sequences, keep global seeding in the driver only, per research R2)

**Checkpoint**: shared engine live; sim arena unaffected; user stories can start

---

## Phase 3: User Story 1 — Solo game against AI opponents (P1) 🎯 MVP

**Goal**: complete hosted game vs solver bots at `/play`: setup → deal → declare →
cut → rounds → winner, AI within latency budget, exhibition mode included.

**Independent Test**: quickstart Scenario B (and G for exhibition); SC-001/002/005/006.

### Tests

- [ ] T009 [P] [US1] Self-play gate in `tbgame/tests/test_self_play.py`: 100 seeded all-bot games run to completion, zero illegal intents, winner matches engine judgment, same seed ⇒ same deal (SC-005; mark `slow` per repo convention)
- [ ] T010 [P] [US1] Beats-random gate in `tbgame/tests/test_solver_bot.py`: good-bot cut choices find safe wires above the uniform-random baseline using the self-calibrating threshold methodology of `tests/baseline_stats.py`, reimplemented here — the root suite cannot be imported-and-edited (SC-006)
- [ ] T011 [P] [US1] API contract tests in `web/tests/test_game_api.py`: POST/GET/DELETE `/api/game`, `/api/game/intent` happy path + 422 illegal intent + 409 version conflict + 409 second game, per contracts/api.md

### Implementation

- [ ] T012 [US1] Solver panel adapter in `tbgame/agents/solver.py` (or lifted from the arena's existing assistant adapter if one lives in `sim/run.py`/prototypes): `PublicState` → four-stat panel dict via `General.py` calls only
- [ ] T013 [US1] `SolverBot` in `tbgame/agents/solver.py`: truthful/lying declaration policy by role, claim policy from posteriors, cut policy sampling over panel stats (research R5); legal-move guarantee; own unseeded RNG
- [ ] T014 [US1] `web/game_service.py`: the single ActiveGame (create/abandon/get), monotonically increasing `version` with stale-echo rejection, background AI worker thread submitting bot decisions with per-seat `thinking` flags (data-model.md)
- [ ] T015 [US1] Game routes in `web/app.py`: `POST/GET/DELETE /api/game`, `POST /api/game/intent`, `GET /play` — transport only, engine reasons forwarded verbatim (contracts/api.md)
- [ ] T016 [US1] Setup screen in `web/static/play/` (index.html + play.js + play.css): seat count/mix (0 humans allowed), names, official role-deal with validated manual override, optional seed
- [ ] T017 [US1] Table UI in `web/static/play/`: seats around a styled-DOM table, card backs with revealed/found states, declaration input (bounded), legal-target cut picking, flip transition on results, claim-menu rider on declare/cut, thinking indicator, winner banner with role reveal
- [ ] T018 [US1] Polling + version flow in `web/static/play/play.js`: refetch on action, ~1 s poll while `thinking`, stale-tab 409 handling; exhibition games render the open-information table

**Checkpoint**: MVP — a full solo/exhibition game is playable end to end

---

## Phase 4: User Story 2 — Hotseat with friends (P2)

**Goal**: multiple humans on one device with server-enforced privacy.

**Independent Test**: quickstart Scenario C; SC-003.

### Tests

- [ ] T019 [P] [US2] Privacy test in `web/tests/test_privacy.py`: script a 3-human game through the API and assert no response body ever contains another seat's role/wires/bomb fields; unlock for seat A then requesting seat B's view is refused (SC-003, FR-008)

### Implementation

- [ ] T020 [US2] Unlock/lock in `web/game_service.py` + `web/app.py`: `POST /api/game/unlock` (version-checked, human seats only, single unlocked seat) returning PrivateView, `POST /api/game/lock`, implicit re-lock on any accepted intent and on game load (contracts/api.md)
- [ ] T021 [US2] Pass-the-device flow in `web/static/play/`: full-viewport named handoff overlay, tap-to-reveal private view (role + hand), explicit hide, auto re-lock on refresh (server is authoritative)

**Checkpoint**: US1 + US2 — human-only and mixed tables playable

---

## Phase 5: User Story 3 — Home page, v1 moved as-is (P3)

**Goal**: `/` is a two-door home; v1 assistant lives unchanged at `/assistant`.

**Independent Test**: quickstart Scenario A; SC-004.

### Implementation

- [ ] T022 [US3] Move v1 static files to `web/static/assistant/` (index.html, main.js, styles.css — contents untouched) and serve `GET /assistant` from them in `web/app.py`; `POST /api/panel` unchanged
- [ ] T023 [P] [US3] New home page `web/static/index.html` (+ shared styles): two doors (Assistant, Play), continue/resume affordance placeholder; `GET /` serves it
- [ ] T024 [US3] Update v1 tests in `web/tests/` for the URL move only (static paths), then run the full v1 suite — behavior identical (SC-004)

**Checkpoint**: the front door exists; v1 users unaffected

---

## Phase 6: User Story 4 — Opt-in stats side panel (P4)

**Goal**: per-game opt-in, collapsible drawer, numbers identical to v1.

**Independent Test**: quickstart Scenario D; SC-008.

### Tests

- [ ] T025 [P] [US4] Parity test in `web/tests/test_panel_parity.py`: play a scripted game, feed the same public events to v1 `POST /api/panel` and to `GET /api/game/panel`, assert numeric identity at every step; assert 403 when not opted in (SC-008, FR-020)

### Implementation

- [ ] T026 [US4] `web/panel_bridge.py`: hosted-game public events → 001 GameRecord (claims omitted) → `replay.replay_record` (research R4; zero probability code)
- [ ] T027 [US4] `GET /api/game/panel` in `web/app.py` (403 unless the game opted in) and the `panel_allowed` setup field end to end (`GameSetup` → create → TableView)
- [ ] T028 [US4] Collapsible side drawer in `web/static/play/`: toggle visible only when allowed, collapsed by default, four-stat table per seat, updates with the poll cycle, no recommendation styling (FR-021)

**Checkpoint**: the signature assistant-at-the-table, gated per game

---

## Phase 7: User Story 5 — Post-game reveal & replay (P5)

**Goal**: full reveal at game end; step through history with truth annotations.

**Independent Test**: quickstart Scenario F; SC-009.

### Tests

- [ ] T029 [P] [US5] Reveal/annotation test in `tbgame/tests/test_reveal.py`: `reveal()` raises before finish (non-exhibition), exposes roles/hands after, and annotates every declaration truthful/lie against recorded `true_wires` (SC-009)

### Implementation

- [ ] T030 [US5] Include the reveal payload (truth-annotated event list) in TableView when `phase == finished` (and always for exhibition) via `web/game_service.py`
- [ ] T031 [US5] Replay UI in `web/static/play/`: end-screen reveal (roles + hands), enter replay, step backward/forward through declarations/claims/cuts with lie badges

**Checkpoint**: the post-mortem works

---

## Phase 8: User Story 6 — Save and resume (P6)

**Goal**: manual named saves surviving app restarts; refresh needs no save.

**Independent Test**: quickstart Scenario E; SC-007.

### Tests

- [ ] T032 [P] [US6] Save round-trip test in `web/tests/test_saves.py`: save mid-round-2, rebuild service (simulated restart), resume, assert identical public state/pending/version reset and intact hidden state; schema_version mismatch rejected cleanly; duplicate name 409 (SC-007, FR-023)

### Implementation

- [ ] T033 [US6] Save store in `web/game_service.py` (or `web/save_store.py` if it outgrows the service): write/list/load/delete `web/saves/<name>.json` per data-model.md SaveFile; resume via `TableGame.from_events`
- [ ] T034 [US6] Endpoints in `web/app.py`: `GET/POST /api/saves`, `POST /api/saves/<name>/resume`, `DELETE /api/saves/<name>` (contracts/api.md)
- [ ] T035 [US6] UI: save dialog in `web/static/play/`, resume list + delete (with confirm) on the home page `web/static/index.html`; finished saves open straight into replay

**Checkpoint**: interruptible game nights

---

## Phase 9: User Story 7 — LLM opponents, optional (P7)

**Goal**: LLM seats behind the same interface, key-gated, graceful failure.

**Independent Test**: quickstart Scenario H; FR-017/018 scenarios.

### Tests

- [ ] T036 [P] [US7] Stub-LLM tests in `web/tests/test_llm_seats.py`: a deliberately failing fake LLM agent pauses the game (`paused_llm`), `POST /api/game/llm-recover` retry and substitute paths both work; LLM seats rejected at setup when no key is configured

### Implementation

- [ ] T037 [US7] Settings in `web/app.py` + `web/game_service.py`: `GET/PUT /api/settings/llm` (env wins, file fallback `web/instance/settings.json`, key never echoed) per contracts/api.md
- [ ] T038 [US7] LLM seat wiring in `web/game_service.py`: instantiate the arena's `sim/agents/llm.py` agent behind the shared Agent interface with a claim-menu-constrained ask (web-side configuration only — arena prompts untouched, research R7); thinking flag on the AI worker; failure → `paused_llm`
- [ ] T039 [US7] UI: LLM seat option in setup (disabled + hint without a key), per-seat thinking indicator styling, retry/substitute dialog in `web/static/play/`

**Checkpoint**: all seven stories done

---

## Phase 10: Polish & Cross-Cutting

- [ ] T040 [P] Docs: update repo `README.md` and `CLAUDE.md` web-layer notes for the v2 layout (`tbgame/` source root, `/assistant` + `/play` routes, new test suites and commands)
- [ ] T041 [P] Responsive + dark-mode pass over home and play surfaces (seat ellipse degrades to a list on phone widths, FR-014; same dark-mode mechanism as v1)
- [ ] T042 Run every quickstart.md scenario (A–H) against a fresh venv start; fix what falls out
- [ ] T043 Full sweep: `.venv/bin/python -m pytest -q` (frozen root suite) + `.venv/bin/python -m pytest tbgame/tests web/tests -q` — all green; commit in clean chunks per constitution

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (1)** → **Foundational (2)**: T003 blocks T004–T008; T008 gates everything after
- **User stories (3–9)**: all depend on Phase 2 only — except US4's T026 (needs US1's
  event flow to exist, and the parity test T025 exercises US3's untouched `/api/panel`,
  which exists already in v1), US5's T030–T031 and US6's T033–T035 (need US1's service
  and UI shell). US2, US3 are fully independent of US1's internals (US3 touches only
  static files/routes).
- **Polish (10)**: last

### Story order (single developer)

US1 (MVP) → US2 → US3 → US4 → US5 → US6 → US7 — exactly the spec's priority order;
stop-and-validate at every checkpoint using the matching quickstart scenario.

### Parallel Opportunities

- Phase 2: T004 (tests) alongside T005–T007 implementation; T006/T007 in parallel after T005
- Each story's test tasks ([P]) before/alongside its implementation
- US3 (static move + home page) can run in parallel with US1/US2 at any point after Phase 2
- Polish T040/T041 in parallel

## Implementation Strategy

MVP first: Phases 1–3 deliver a playable solo game (quickstart B/G) — demo it before
building the rest. Then one story per increment, validating with its quickstart
scenario and committing per checkpoint. The frozen root suite is run at every
checkpoint (it is the regression harness for the engine promotion, not just polish).
