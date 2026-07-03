# Contract: tbgame — the shared rules engine

The public surface of the `tbgame/` source root. Two consumers: the web game service
and the sim arena (through the compatibility driver). Anything not listed here is
internal.

## tbgame.state

Moved verbatim from `sim/state.py` (the renderer stays in sim):
`GroundTruth`, `PublicState`, `PrivateView`, `AgentView`, `EventLog`,
`legal_targets(public, cutter)`, `SCHEMA_VERSION`.

## tbgame.engine.TableGame — stepwise core

```python
game = TableGame(setup)              # validates GameSetup; raises SetupError with a reason
game = TableGame.from_events(events) # resume/replay: reduce a saved event log; raises
                                     #   SchemaError on schema_version mismatch
```

| Member | Contract |
|---|---|
| `game.pending` | `Pending(kind, seats)` or `None` when finished. Declarations are owed simultaneously by all listed seats; a cut is owed by exactly one. |
| `game.submit(intent)` | Applies one intent (see data-model.md). Validates seat/kind/value/claim; raises `IllegalIntent` (nothing applied) on any violation. On acceptance appends event(s), advances phase — including automatic round advance and game end — and returns the list of newly appended events. |
| `game.public_state()` | Deep-copied `PublicState` snapshot (claim_log, phase, pending included). |
| `game.private_view(seat)` | That seat's `PrivateView`. The engine hands it to any caller — *entitlement is the host's job* (web: unlocked-seat gate; sim: the firewall of view construction). |
| `game.events` | The append-only event list (includes sealed deal data; hosts must not forward it to entitled-view consumers before `finished`). |
| `game.finished` / `game.outcome` | `outcome = {good_guys_won, reason, roles}` once finished, else `None`. |
| `game.reveal()` | Only when finished (or exhibition): full ground truth + truth-annotated event list for replay. |

Determinism: all engine randomness (role sample, `General.DistributeWires` deals, cut
resolution draw) comes from `random.Random(setup.seed)`. Outcomes are recorded in
events; `from_events` re-applies recorded outcomes and never re-rolls.

Rule authority: hand sizes 5→2, one cut turn per player per round, cutter-passes-to-
target, win/loss judgment — all decided here, matching today's `sim/engine.py`
behavior exactly (the arena driver is the regression harness for this).

## tbgame.driver.Engine — sim compatibility driver

Preserves today's blocking arena API so `sim/engine.py` can shim to it and
`tests/test_arena_engine.py` passes unmodified:

```python
Engine(num_players=6, initial_hand_size=5, player_names=None,
       panel_for=(), assistant=None, max_workers=None)
outcome, log = engine.play_game(agents, seed=None)
```

Behavior contract (unchanged from sim): simultaneous threaded declaration phase;
sequential free-text `discuss` phase (arena keeps statements — the driver maps them
to `statement` events, not claims); malformed agent output falls back instead of
raising; `assistant.panel(public_state)` adapter for `panel_for` seats; same event
vocabulary (`wire`/`dud`/`bomb`).

## tbgame.agents.base.Agent

Moved from `sim/agents/base.py`, same methods (`declare`, `choose_cut`, `discuss`,
`observe`, `describe`) plus one addition, default no-op, so existing agents are
unaffected:

```python
def claim(self, view):
    """Return a Claim dict from the fixed menu (or None). Called by hosts that use
    structured claims (the web game) at the agent's decision points."""
    return None
```

## tbgame.agents.solver.SolverBot

```python
SolverBot(panel_adapter, rng=None)   # panel_adapter: PublicState -> panel dict (the
                                     #   same adapter shape the arena's assistant uses)
```

Implements `Agent` fully (declare / claim / choose_cut). Guarantees: only legal
moves; decisions derived from panel numbers + own `PrivateView` via policy rules (no
probability computed outside the solver); capable of lying when `my_role == 1`;
decision latency well under 2 s given a within-budget panel adapter.

## Error types

`SetupError`, `IllegalIntent`, `SchemaError` — all carry a human-readable `reason`
the web layer forwards verbatim (web adds no rule language of its own).
