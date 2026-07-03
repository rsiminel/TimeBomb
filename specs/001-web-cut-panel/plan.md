# Implementation Plan: Time Bomb Web Cut Panel

**Branch**: `001-web-cut-panel` | **Date**: 2026-07-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-web-cut-panel/spec.md`

## Summary

Rework `web/` into a local-machine browser assistant: the user sets up a game, enters
declarations and cut results, and after every entry sees the four-stat cut panel plus
P(bad)/P(num_bad) — all computed by `timebomb/General.py`. The web layer is an
append-only **event log** (kept in the browser) plus a **stateless recompute endpoint**
that replays the log through the exact call pattern of `General.Play` (the reference
orchestration): `ProbDeclaration` → `ProbCut` for the current round, `RoundLogU`
accumulation for completed rounds, `JointBadBelief` for the joint bad-guy belief,
`CutPanel` for the four stats. Full-history undo and reload survival fall out of the
event-log design for free (undo = truncate log; reload = replay stored log).

## Technical Context

**Language/Version**: Python 3 (backend, matching the repo's `.venv`); vanilla ES2020
JavaScript + CSS (frontend, no build step)

**Primary Dependencies**: Flask (server + static file serving; added to `.venv`);
numpy/scipy already present for the solver. No frontend dependencies.

**Storage**: browser `localStorage` for the event log of the in-progress game; the
server holds no game state. No database.

**Testing**: pytest with Flask's test client, in `web/tests/` (the repo-root `tests/`
directory is backend-only and off-limits per Constitution III)

**Target Platform**: user's own machine (Linux/macOS), used from a desktop browser or a
phone on the same Wi-Fi; no public hosting, no auth

**Project Type**: small web app — one-module Python backend + static frontend

**Performance Goals**: panel response < 2 s at 8 players (SC-002); measured depth cap
for the round-horizon information stat (see research.md R7)

**Constraints**: zero probability formulas in the web layer (Constitution I);
quantities-only display (Constitution II); no modifications under `timebomb/` or
`tests/` (Constitution III)

**Scale/Scope**: one user, one game at a time, 4–8 players, four rounds

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Design compliance |
|-----------|-------------------|
| I. The Solver Owns the Math | The endpoint only **orchestrates** `General.py` calls, mirroring `General.Play`'s call pattern verbatim (`ProbDeclaration`, `ProbCut`, `RoundLogU`, `JointBadBelief`, `Separate`, `CutPanel`, `NUM_BAD_PRIOR`, plus `Consistency` checks). No arithmetic on probabilities anywhere in `web/` — not even a division. The old `web/app.py` math is deleted outright. |
| II. Quantities, Not Commands | The panel renders the solver's numbers in fixed seat order — no sorting by any stat, no highlighting, no "best cut" affordance. Game-end banners state the winner, never advice. |
| III. Web Work Stays in web/ | All new code and tests live under `web/`. `timebomb/` and `tests/` are untouched; the solver is imported read-only (via `sys.path` insertion of the `timebomb/` source root). Flask is added to the git-ignored `.venv` only. |

**Initial gate: PASS.** **Post-Phase-1 re-check: PASS** — the API contract exposes only
solver outputs (see contracts/api.md); the event-log replay adds bookkeeping (whose
turn, hand sizes, wire counts) but no probability computation; bookkeeping values are
fed *into* solver calls, never derived *from* their outputs.

## Project Structure

### Documentation (this feature)

```text
specs/001-web-cut-panel/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api.md           # Phase 1 output — HTTP contract
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
web/
├── app.py               # Flask app: static serving + POST /api/panel (rewritten from scratch)
├── replay.py            # Event-log validation & replay → solver calls → response dict
├── static/
│   ├── index.html       # Single page: setup screen + game screen (rewritten)
│   ├── main.js          # Event log, localStorage, fetch, rendering (rewritten)
│   └── styles.css       # Phone-first layout (rewritten)
└── tests/
    ├── conftest.py      # sys.path for the timebomb/ source root + app fixture
    ├── test_replay.py   # Log validation, round bookkeeping, undo/replay equivalence
    ├── test_api.py      # Endpoint behaviour, warnings, game end, error cases
    └── test_solver_parity.py  # SC-003: scripted games — API output == direct General.py calls
```

**Structure Decision**: keep the existing flat `web/` folder but split orchestration
(`replay.py`) from transport (`app.py`) so parity tests can drive the replay logic
without HTTP. Old `web/app.py`, `web/index.html`, `web/main.js`, `web/styles.css` are
replaced; nothing else in the repo changes.

## Complexity Tracking

No constitution violations to justify — table intentionally empty.
