# Research: Hosted Local Time Bomb Game

Phase 0 output. Each entry: Decision / Rationale / Alternatives considered. Grounded
in a read of `sim/engine.py`, `sim/state.py`, `sim/agents/*`, `web/replay.py`,
`web/app.py`, the root `conftest.py`/`pytest.ini`, and constitution v2.0.0.

## R1. Shape of the engine promotion

**Decision**: Create a third top-level source root `tbgame/` holding the state
dataclasses (`GroundTruth`, `PublicState`, `PrivateView`, `AgentView`, `EventLog`,
`legal_targets`), a stepwise event-sourced core `TableGame`, a compatibility driver
reproducing today's blocking `Engine.play_game(agents, seed)`, and the agent
interface. `sim/engine.py`, `sim/state.py`, and `sim/agents/base.py` become re-export
shims (the state→text renderer and prompt copy stay in `sim/` — they are arena/LLM
concerns). `tests/test_arena_engine.py` must pass byte-for-byte unmodified.

**Rationale**: The current referee calls agents synchronously inside one loop; the
web receives human decisions as HTTP requests, so control must be inverted — the game
asks "who owes what decision" and accepts submissions. Event-sourcing the core also
gives save/resume and post-game replay for free (both spec requirements). Shims keep
the untouchable root test suite green, satisfying the constitution III carve-out.

**Alternatives considered**: (a) Run the blocking loop in a thread with human agents
that block on queues fed by HTTP — least code change, but a blocked thread cannot be
saved/resumed and every game holds a thread hostage; rejected. (b) Duplicate a
web-side engine — two rule books, forbidden by constitution I; rejected. (c) Rewrite
sim/ to use the stepwise API directly — touches the frozen root tests
(`tests/test_arena_engine.py` imports the current API); shims achieve the same
without edits there.

## R2. Randomness model

**Decision**: `TableGame` owns a `random.Random(seed)` instance used for the role
sample, every round's `General.DistributeWires` deal, and cut resolution. Bots use
their own unseeded RNG. The sim compatibility driver keeps the current global
`random.seed(seed)` call semantics only if `tests/test_arena_engine.py` turns out to
depend on exact global-RNG sequences; otherwise it passes the seed through to the
instance RNG.

**Rationale**: Clarification Q2 — the seed fixes the deal, not bot behavior. An
instance RNG is also what makes save/resume deterministic to reconstruct (the event
log records deal outcomes, so resume replays events rather than re-rolling).
`DistributeWires` accepts NumPy's global state today; the engine records the *dealt
result* in the log, so replay never re-invokes the RNG.

**Alternatives considered**: global `random.seed` (v1 sim behavior) — leaks
determinism into bots, violating Q2; full-transcript determinism — rejected by
clarification.

## R3. Web session & concurrency model

**Decision**: One in-memory `TableGame` held by a `game_service` module (the spec's
one-active-game assumption). Flask handlers submit intents and read views; AI turns
run on a single background worker thread that watches `pending` and submits bot/LLM
decisions, setting a per-seat `thinking` flag the UI polls (~1 s interval). Hotseat
privacy: the server tracks which seat's private view is currently unlocked (the
pass-the-device acknowledgement is an API call); a private-view request for any other
seat is refused, so secrets stay server-side (FR-008) even though there is one
browser. Every state response carries a monotonically increasing `version`; intent
submissions must echo it, so a stale tab's action is rejected (409), never
double-applied.

**Rationale**: LAN-shaped without building LAN: views are seat-addressed and
entitlement is enforced server-side, so a future multi-device update replaces the
"unlocked seat" mechanism with per-connection identity and nothing else. Polling
beats SSE/WebSockets for a dependency-light local app.

**Alternatives considered**: WebSockets/SSE push — more moving parts for a same-host
round trip; client-side game state — violates FR-008 and the constitution; per-seat
session cookies now — pointless on one shared device, and the seat-addressed view API
already preserves the option.

## R4. Panel with zero new math

**Decision**: A `panel_bridge` module translates the hosted game's public events
(declarations, cuts, round advances, setup) into the 001 `GameRecord` shape and calls
`web/replay.py::replay_record` — the panel JSON the v1 assistant would show for the
same public history, verbatim. Claims are omitted from the record (they carry no
probability information the solver models).

**Rationale**: SC-008 demands numeric identity with v1; the cheapest correct
implementation is to *be* v1. `replay.py` is pure and stateless, already handles the
official role-deal prior, the depth-capped info stat, and the 2 s budget
(`DEPTH_CAP`). Zero probability code is added anywhere.

**Alternatives considered**: calling `General.py` directly from the game service —
duplicates 001's orchestration (round bookkeeping, prior handling) and creates a
second place panel semantics live; rejected.

## R5. Solver-bot design

**Decision**: `tbgame/agents/solver.py` implements the shared `Agent` interface. It
obtains the public-info panel through the same assistant-adapter pattern the arena
already supports (`Engine(panel_for=..., assistant=...)`) and combines it with its
`PrivateView` using *decision rules only*:
- **Good bot** — declares its true wire count; cuts by sampling among the top targets
  by P(safe wire), tempered by the info-value stat; claims distrust of the seat with
  the highest P(bad) when that posterior is confident.
- **Bad bot** — declares a lie drawn from a plausible range around the honest value;
  prefers cuts that waste turns (low P(safe) targets while avoiding suspicion);
  claims trust/honesty to muddy the water, occasionally counter-accusing.
The exact temperature/threshold knobs are implementation details; the acceptance gate
is behavioral: SC-005 (100 self-play games, zero illegal moves) and SC-006
(beats-random at finding safe wires, using the self-calibrating threshold methodology
of `tests/baseline_stats.py`, reimplemented in `tbgame/tests/` since the root suite
cannot be modified).

**Rationale**: All numbers come from the solver (constitution I); the bot layer only
ranks, samples, and thresholds them. Deception ability is FR-016; the quality bar is
deliberately "beats random," not "beats the arena LLMs."

**Alternatives considered**: reusing `General.PlayAuto` — bakes in `CutRandom` and an
omniscient belief tracker (exactly why the arena didn't use it); a belief tracker
conditioned on the bot's own hand — needs new solver interfaces (a backend task by
constitution I, explicitly out of scope for v2).

## R6. Structured claims

**Decision**: A new engine event/intent `claim` with a fixed vocabulary:
`trust(target)`, `distrust(target)`, `accuse_lie(target)` and `self_honest` — emitted
only as part of the acting seat's turn (clarification Q3): the declare and cut intents
carry an optional claim payload. The arena's free-text `statement` event remains a
separate, sim-only concept; the two never mix in one game.

**Rationale**: Menu-driven claims make AI and human table talk symmetric (FR-011),
render trivially, and replay with truth annotations. Riding the turn intents (rather
than a free-floating claim endpoint) makes attribution inherent, per clarification.

**Alternatives considered**: free-text with moderation — excluded by spec; claims as
independent events from the shared surface — rejected by clarification Q3;
mechanically meaningful claims (votes) — not in the physical game's rules.

## R7. LLM seats

**Decision**: Reuse `sim/agents/llm.py` agents behind the same interface, configured
by the web app. Key resolution: `ANTHROPIC_API_KEY` (or the arena's existing env
convention) wins; else `web/instance/settings.json` written by a small settings
endpoint (masked in UI, never echoed back — clarification Q4). LLM decisions run on
the AI worker thread with the `thinking` flag; on failure the game pauses and offers
retry or substitution by a solver bot (FR-017). For claims, the LLM ask is
constrained to the claim menu (a small prompt addition in the web-facing
configuration, not a change to arena prompts — the emergent-play rule for arena runs
stays intact).

**Rationale**: The arena agents are tested, share the firewall, and the memory rule
("no strategy steering in arena prompts") applies to arena *strategy*, not to
constraining an output format for a different product surface.

**Alternatives considered**: fresh web-only LLM agent — duplicates prompt/transport
work; skipping LLM seats — pluggability is cheap since the interface already exists.

## R8. Saves and post-game replay

**Decision**: A save is one JSON file `web/saves/<name>.json`:
`{schema_version, created, name, setup, events}` where `events` is the full engine
event log (including the sealed deal events — the file lives on the hosting machine,
same trust domain as the server's memory). Resume = rebuild `TableGame` by reducing
the event log (no RNG re-rolls, see R2). A `schema_version` mismatch is rejected with
a clear error (FR-023). Post-game replay reads the same event log; truth annotations
compare each declaration to the `true_wires` already recorded on declaration events.
Saves are manual and named (clarification Q1); browser refresh needs no save because
the service holds authoritative state (FR-024).

**Rationale**: The event log is already the sim's single source of replay
(`sim/LOGS.md` discipline); one format serves save, resume, and replay. Keeping
`true_wires` on declaration events (as the arena log already does) makes lie
annotation a lookup, not a computation.

**Alternatives considered**: pickling live objects — fragile across versions, opaque;
autosave — rejected by clarification Q1; separate save vs replay formats — two
serializers for one structure.

## R9. Frontend structure

**Decision**: Three static surfaces, vanilla JS, no build step: `static/index.html`
(home, two doors), `static/assistant/` (v1 files moved verbatim; `/assistant` serves
them), `static/play/` (game table). The table is styled DOM: seats positioned around
an elliptical table via CSS, hands as card-back divs, CSS transitions for cut
reveals, a collapsible side drawer for the panel (reusing v1's table styles where
practical), pass-the-device screens as full-viewport overlays, dark mode via the same
mechanism v1 uses. State updates by polling the state endpoint (~1 s while AI thinks,
immediate refetch after own actions).

**Rationale**: Constitution's dependency-light constraint; v1 proved the shape. A
phone-sized layout is required by FR-014 — the seat ellipse degrades to a vertical
list under a width breakpoint.

**Alternatives considered**: framework/build step — justified only if interaction
complexity outgrows vanilla JS, which a 4–8 seat turn game does not; canvas — harder
to test, no accessibility tree.

## R10. Where new tests live and how they run

**Decision**: `tbgame/tests/` (engine unit + contract tests, self-play SC-005,
beats-random SC-006) and `web/tests/` (API, privacy SC-003, panel parity SC-008, v1
regression SC-004). The root `pytest.ini` (`testpaths = tests`) is untouched; the
quickstart documents running the new suites explicitly
(`.venv/bin/python -m pytest tbgame/tests web/tests -q`). Repo `conftest.py` gains
`tbgame` in its source-root list — the one line of shared infrastructure this feature
touches outside `web/`, `tbgame/`, `sim/`.

**Rationale**: Constitution III forbids modifying `tests/`; the frozen suite keeps
guarding the solver and (via `test_arena_engine.py`) the promoted engine's
compatibility driver.

**Alternatives considered**: adding new paths to `pytest.ini` testpaths — would
change what the documented backend command runs, surprising backend workflows.
