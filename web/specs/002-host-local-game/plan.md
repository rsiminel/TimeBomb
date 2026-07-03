# Implementation Plan: Hosted Local Time Bomb Game

**Branch**: `002-host-local-game` | **Date**: 2026-07-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-host-local-game/spec.md`

## Summary

v2 turns the web app from a play-along assistant into a host for the game itself:
a home page fronts the untouched v1 assistant (`/assistant`) and a new hosted game
(`/play`) — hotseat humans and/or pluggable AI seats over a shared rules engine
promoted out of `sim/engine.py` into a new top-level source root (`tbgame/`). The
promotion inverts the referee from a blocking agent-driven loop into a stepwise,
event-sourced core (`TableGame`) that the web can drive one intent at a time and the
sim arena keeps using through a compatibility driver, so `tests/test_arena_engine.py`
passes unmodified. Every probability shown (the opt-in side panel) is produced by
translating the game's public events into a 001 GameRecord and reusing `web/replay.py`
→ `General.py` — zero new math anywhere. Solver bots decide from the same solver
numbers plus a policy layer; LLM seats optionally reuse the arena agents.

## Technical Context

**Language/Version**: Python 3.12 (backend, engine); ES2020 vanilla JavaScript (frontend)

**Primary Dependencies**: Flask (already used by v1); numpy via the solver; no new
Python dependencies; no JS framework, no build step (constitution constraint)

**Storage**: JSON files on the hosting machine — named saves in `web/saves/`
(git-ignored), optional API key in `web/instance/settings.json` (git-ignored).
No database.

**Testing**: pytest (`.venv`), same split as today: engine tests in `tbgame/tests/`,
web tests in `web/tests/` (both outside the frozen root `tests/` suite, which must
keep passing untouched)

**Target Platform**: local machine (Linux/macOS), browser UI usable on a phone-sized
screen; single running instance, one active game

**Project Type**: web application (Python backend + static frontend) over a shared
Python engine package

**Performance Goals**: solver-bot decisions ≤ 2 s (SC-002); panel computation within
v1's 2 s budget via the existing `DEPTH_CAP`; non-AI UI interactions feel immediate

**Constraints**: no math or rule adjudication in `web/` (constitution I); secrets
never sent to a browser in human games (FR-008); no modification to `timebomb/`,
`tests/`, or v1 assistant behavior; `sim/` adapted only via shims that keep its tests
green; offline-capable except optional LLM seats

**Scale/Scope**: 4–8 seats, one active game, ~4 rounds × ≤8 cuts; event logs of a few
hundred entries; single household of users

## Constitution Check

*GATE: constitution v2.0.0, Principles I–III. Evaluated pre-research and re-checked
post-design.*

- **I. Solver owns the math, engine owns the rules — PASS.** All probabilities reach
  the browser through `web/replay.py` → `General.py` (the panel path is literally
  v1's, fed by a translation of the game's public events). All rules live in
  `tbgame/` (promoted referee); `web/` holds transport, session, and presentation
  only. Solver bots consume solver output through the same panel adapter plus their
  private view; their policy layer makes decisions, it computes no probabilities.
- **II. Quantities, not commands (assistant surface) — PASS.** The panel is per-game
  opt-in, symmetric, recommendation-free (FR-019/021). Bots deciding cuts is the
  players-not-panel case the principle explicitly permits; no bot intention reaches
  the panel.
- **III. Web work stays in web/ + carve-out — PASS.** New code lands in `web/` and
  the new shared package `tbgame/`; `sim/engine.py`, `sim/state.py`, and
  `sim/agents/base.py` become re-export shims so `tests/test_arena_engine.py` (in the
  untouchable root suite) passes without edits. `timebomb/` and `tests/` are not
  modified. The repo-root `conftest.py` gains one source-root entry (`tbgame`) — it
  is shared infrastructure, not part of `timebomb/` or `tests/`, so no violation; it
  is called out here for transparency.

**Post-design re-check (after Phase 1)**: PASS — the contracts confirm the split: the
engine contract (`contracts/engine.md`) exposes intents/views/events with no
probability surface; the web API (`contracts/api.md`) exposes no rule decisions of
its own; the panel endpoint returns `replay.py` output verbatim.

## Project Structure

### Documentation (this feature)

```text
web/specs/002-host-local-game/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── engine.md        # tbgame/ public contract (shared engine)
│   └── api.md           # web HTTP contract
└── tasks.md             # Phase 2 output (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
tbgame/                    # NEW shared source root (bare-name imports, like timebomb/ and sim/)
├── state.py               # GroundTruth / PublicState / PrivateView / AgentView / EventLog /
│                           #   legal_targets — moved from sim/state.py (renderer stays in sim/)
├── engine.py               # TableGame: stepwise event-sourced core (start/pending/submit/view)
├── driver.py                # Engine.play_game compatibility driver for the sim arena
├── agents/
│   ├── base.py              # Agent interface (moved from sim/agents/base.py) + claim hook
│   └── solver.py            # solver-driven bot (policy over panel numbers + private view)
└── tests/                   # engine + solver-bot tests (incl. beats-random, SC-005/6)

sim/                        # adapted, all its root-suite tests must stay green
├── engine.py                # shim: re-exports the compatibility driver
├── state.py                 # shim: re-exports tbgame.state + keeps renderer helpers
└── agents/base.py           # shim: re-exports tbgame.agents.base

web/
├── app.py                   # routes: /, /assistant, /play + /api/game/* (transport only)
├── game_service.py          # NEW: the one active TableGame, seat locks, saves, AI turn worker
├── panel_bridge.py          # NEW: public events -> 001 GameRecord -> replay.replay_record
├── replay.py                # UNCHANGED (v1)
├── saves/                   # named save files (git-ignored)
├── instance/settings.json   # optional LLM key (git-ignored)
├── static/
│   ├── index.html           # NEW home page (two doors)
│   ├── assistant/           # v1 page moved as-is (index.html, main.js, styles.css)
│   └── play/                # NEW game UI (index.html, play.js, play.css)
└── tests/                   # v1 tests (kept passing) + new game/API/parity tests
```

**Structure Decision**: `tbgame/` joins `timebomb/` and `sim/` as a third source root
following the repo's bare-import convention (repo `conftest.py` gains the entry; the
web app extends `sys.path` itself as v1 already does). `web/` keeps v1's shape — small
Flask backend, static frontend, no build step.

## Complexity Tracking

No constitution violations to justify. Two accepted complexity points, noted for the
record:

| Item | Why Needed | Simpler Alternative Rejected Because |
|------|------------|--------------------------------------|
| Stepwise engine core + sim compatibility driver | The web receives human intents as HTTP requests; a blocking `play_game` loop cannot be driven that way, and save/resume needs event-sourced state | Running the blocking loop in a thread with queue-fed human agents keeps sim's shape but makes mid-game save/resume (a spec requirement) effectively impossible — a blocked thread cannot be serialized |
| Background worker for AI turns | LLM seats take seconds; HTTP handlers must return immediately and the UI polls a thinking flag (FR-018) | Deciding AI turns inside the request handler blocks the shared device's UI and times out on LLM latency |
