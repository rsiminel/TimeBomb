# Tasks: Time Bomb Web Cut Panel

**Input**: Design documents from `/specs/001-web-cut-panel/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api.md, quickstart.md

**Tests**: included — the spec's SC-003/SC-004 and the constitution's Constraints
explicitly require the web suite (solver parity, warning discipline), so test tasks are
not optional here.

**Organization**: grouped by user story; each story phase is an independently testable
increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different files, no dependency on an incomplete task)
- **[US1/US2/US3]**: maps to the spec's user stories

## Phase 1: Setup

**Purpose**: clear out the old implementation, put the new skeleton in place

- [X] T001 Delete the old implementation (`web/app.py`, `web/index.html`,
      `web/main.js`, `web/styles.css`), create the `web/static/` and `web/tests/`
      directories per plan.md, and install Flask into the project venv
      (`.venv/bin/pip install flask`)
- [X] T002 [P] Create `web/tests/conftest.py`: insert the `timebomb/` source root on
      `sys.path` (mirroring the repo-root `conftest.py`) and provide a Flask
      test-client fixture for the app

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the event-log record model, its validation, and the derived-state replay —
every story and every endpoint response depends on these

**⚠️ CRITICAL**: no user story work until this phase is complete

- [X] T003 Implement GameRecord parsing and validation in `web/replay.py` per
      data-model.md: setup constraints (4–8 unique non-empty names, bomb flag,
      `numBadOverride` legal for N — any `1 ≤ B ≤ N−1`, the range the solver
      supports, since the override exists for unofficial tables) and
      event-shape/ordering constraints; invalid
      records raise a structured error carrying `eventIndex` (−1 for setup) for the
      422 body of contracts/api.md
- [X] T004 Implement derived-state replay (rules bookkeeping only, zero probability
      code) in `web/replay.py`: round number, hand size 5→2, round-start
      `total_active`, per-round `revealed`/`found`, cut budget, `awaiting`, and
      game-over detection (bomb / wires / time) per data-model.md's transition table
- [X] T005 [P] Write validation + bookkeeping tests in `web/tests/test_replay.py`:
      hand-worked scripted logs covering every 422 case in contracts/api.md, round
      rollover, all three game-over reasons, and replay purity (truncated log ≡
      shorter game)
- [X] T006 Implement the Flask app in `web/app.py`: serve `web/static/` at `/`, wire
      `POST /api/panel` to the replay (400 malformed / 422 invalid / 200 with
      `state` block), no game state held server-side

**Checkpoint**: `POST /api/panel` returns correct `state` for any record
(`belief: null` for now) — user stories can begin

---

## Phase 3: User Story 1 - Live-game assistance (Priority: P1) 🎯 MVP

**Goal**: setup → declarations → cuts → the four-stat panel plus P(bad)/P(num_bad)
after every entry, quantities only

**Independent Test**: quickstart.md scenarios 1–2 — set up a 5-player game, enter
declarations and cuts, see the panel update after each entry with no recommendation
affordance

- [X] T007 [US1] Implement belief orchestration in `web/replay.py`, mirroring
      `General.Play`'s call pattern exactly (research R3): `NUM_BAD_PRIOR`/override →
      per-cut `ProbDeclaration` → `ProbCut` on cumulative `revealed`/`found` →
      provisional `RoundLogU` fold-in → `JointBadBelief` → `Separate` for P(bomb) →
      `CutPanel(..., max_depth)`; populate the `belief` block of contracts/api.md
      verbatim from solver outputs (NaN rows → `noCards`)
- [X] T008 [US1] Add the per-player-count depth-cap table from research R7 to
      `web/replay.py` (`{4:3, 5:2, 6:2, 7:1, 8:1}`), set `approx`/`maxDepth` in the
      response, and confirm the N=4 and N=6 entries with the R7 micro-benchmark
      (adjust the table, not the budget, if over 2 s)
- [X] T009 [P] [US1] Write solver-parity tests in `web/tests/test_solver_parity.py`
      (SC-003): scripted single-round games at N=4 (uncertain deal), N=5 (fixed
      deal), a `numBadOverride` game, and a `bomb: false` game (`num_bom=0` path,
      zero `pBomb` column) — every `belief` number exactly equals the direct
      `General.py` computation
- [X] T010 [P] [US1] Write endpoint tests for panel responses in
      `web/tests/test_api.py`: `belief: null` before declarations, panel present
      after, fixed seat order preserved, `approx` flag set when capped
- [X] T011 [P] [US1] Build the setup screen in `web/static/index.html` +
      `web/static/main.js`: player count/names, bomb toggle, official-deal default
      with fixed-count override, client-side validation, GameRecord creation in
      `localStorage`
- [X] T012 [US1] Build the game screen in `web/static/index.html` +
      `web/static/main.js`: declaration entry, cut entry (player + safe/nothing/bomb;
      `noCards` players rendered unselectable), fetch to `/api/panel` after every
      entry, panel table rendered in seat order with no sorting/highlighting
      (Constitution II), the spec's "information value" column showing `horizon`
      (the depth-capped stat) with its "approximate" label wired to `approx` and
      `onePly` as an optional secondary detail, record replayed from `localStorage`
      on page load (SC-005)
- [X] T013 [P] [US1] Write `web/static/styles.css`: phone-first layout, panel readable
      on a small screen (spec Assumptions)

**Checkpoint**: MVP — a full round is playable end-to-end against the real solver

---

## Phase 4: User Story 2 - Multi-round play (Priority: P2)

**Goal**: rounds advance with shrinking hands, evidence accumulates across rounds, the
game ends with a winner banner

**Independent Test**: quickstart.md scenarios 3–4 — two scripted rounds show carried
evidence; a bomb cut ends the game

- [X] T014 [US2] Implement cross-round evidence accumulation in `web/replay.py`: sum
      `RoundLogU` per candidate bad count over completed rounds (exactly as
      `General.Play` does at round end) so the current round's belief conditions on
      all prior rounds
- [X] T015 [P] [US2] Extend `web/tests/test_solver_parity.py` with a scripted
      two-round game: round-2 `belief` equals the direct solver computation with
      round-1 evidence accumulated, and differs from a fresh game fed only round-2
      entries
- [X] T016 [US2] Add round-advance and game-over UI to `web/static/main.js` +
      `web/static/index.html`: rounds auto-advance when the cut budget is spent (per
      data-model.md transitions), new-round declaration prompt at the smaller hand size,
      winner banner (side + reason, no advice), entry controls disabled after game
      over, new-game action that discards the stored record
- [X] T017 [P] [US2] Extend `web/tests/test_api.py`: round rollover response fields,
      all three `gameOver` variants, and no-entries-accepted-after-game-over (422)

**Checkpoint**: a complete 4-round game is playable without leaving the page

---

## Phase 5: User Story 3 - Table-mistake tolerance (Priority: P3)

**Goal**: bad input never wedges the assistant — inline rejection, once-per-round
impossibility warnings, full-history undo

**Independent Test**: quickstart.md scenario 5 — out-of-range declaration rejected
inline, impossible declarations warn once, undo steps back to game start

- [ ] T018 [US3] Implement warning detection in `web/replay.py` mirroring
      `General.Play`: declaration feasibility via `Consistency` across candidate bad
      counts, impossible-cut signal (`ProbCut` returning its prior), each emitted at
      most once per round in the `warnings` array (200, never 422 — contract
      invariant 3)
- [ ] T019 [US3] Add mistake handling to `web/static/main.js`: inline `[0, hand
      size]` declaration validation before any request, warning banner rendering,
      undo button that truncates the stored log by one event and refetches, and 422
      recovery that surfaces `eventIndex` with an undo offer
- [ ] T020 [P] [US3] Write warning-discipline and undo tests in
      `web/tests/test_api.py` + `web/tests/test_replay.py`: jointly impossible
      declarations → 200 with one warning (not repeated next entry), impossible cut →
      warning with belief left at prior, undo-to-start equivalence (SC-004)

**Checkpoint**: all three stories independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T021 [P] Add a latency guard test in `web/tests/test_api.py`: worst-case
      round-start request at N=8 answers within the SC-002 budget on this machine
      (marked slow; generous CI margin)
- [ ] T022 Run every quickstart.md validation scenario end-to-end in a real browser
      (desktop + phone-sized viewport) and fix what falls out; record results in
      `specs/001-web-cut-panel/quickstart.md` checkboxes or notes
- [ ] T023 [P] Update repo docs for the new web layer: `README.md` (how to run the
      assistant), `TODO.md` C1 checkbox, and the `CLAUDE.md` "known traps" bullet
      that still calls `web/app.py` bug-ridden

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none
- **Foundational (Phase 2)**: T003/T004 after T001; T005 after T003+T004; T006 after
  T003+T004 (T002 anytime after T001) — **blocks all stories**
- **US1 (Phase 3)**: after Phase 2. T007 → T008; T009/T010 after T007 (T009 also
  after T008 for `approx`); T011 after T001; T012 after T006+T007+T011; T013 anytime
- **US2 (Phase 4)**: after US1's T007 (extends the same orchestration); T015/T017
  after T014; T016 after T012+T014
- **US3 (Phase 5)**: T018 after T007; T019 after T012; T020 after T018
- **Polish (Phase 6)**: after all stories

### Parallel Opportunities

- T002 ∥ T003/T004 (different files)
- T005 ∥ T006 once T003/T004 land
- T009 ∥ T010 ∥ T011 ∥ T013 within US1
- T015 ∥ T016 ∥ T017 within US2 (T015/T017 after T014)
- T021 ∥ T023 in Polish

---

## Implementation Strategy

**MVP first**: Phases 1–3 (T001–T013) deliver User Story 1 — one full round against
the live solver, panel after every entry. Stop, run quickstart scenarios 1–2, validate
with a real game state, commit.

**Incremental delivery**: US2 (T014–T017) makes it a whole-game tool; US3 (T018–T020)
makes it table-proof; Polish (T021–T023) locks in latency and updates repo docs. Each
checkpoint is a committable, demonstrable increment — per the constitution, commit
verified work in clean, well-scoped chunks as each checkpoint passes.
