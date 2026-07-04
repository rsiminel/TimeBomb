# Time Bomb Arena — LLM Multi-Agent Simulation (Spec)

A sandbox for **LLM agents playing full games of Time Bomb under their own free
strategy** — declaring, bluffing, accusing, and cutting — so we can *watch what emerges*.

This is a **learning / curiosity project**: the goal is to understand how LLM agents are
built and used for simulation, and — because Time Bomb is a fun social-deduction game — to
see what kind of play, deception, and table dynamics emerge when the agents are given free
reign. It is, unapologetically, a bit of a balanced-game study of Time Bomb itself.

A purely *programmatic* multi-agent sim that obeys the model's assumptions already exists:
`timebomb/General.PlayAuto`. This sub-project does **not** rebuild that. Its whole reason
to exist is the part `PlayAuto` can't give us — agents that choose their own strategy
rather than sampling a hardcoded uniform-lie distribution.

> **Status:** spec + an M0 sketch (engine + cut-decision prototype). The backend
> (`timebomb/`) remains the source of truth for all probability math and is assumed
> correct ([../docs/model.md](../docs/model.md)). The web app is on hold and `AI.py` is
> known-broken and **off-limits** — do not reference or reuse it.

---

## 1. Principles

1. **Emergent, not scripted.** We never hardcode a lie pattern. The agents are given the
   rules, their private hand, and the public history, and decide freely. Bluffing is
   something we *observe*, not something we program — a scripted liar would just be a
   worse `PlayAuto`.

2. **The information firewall is a type invariant** (§3). The engine is omniscient; an
   agent sees only its own hand plus the public log; if the assistant is in play, it too
   sees only the public log. A private field can't leak to an agent because it isn't in
   the object the agent is handed — not because we remembered to hide it.

3. **Never debug the environment and the agent at the same time.** The deterministic game
   engine is built and verified *first*, drivable by a trivial random agent and by a human
   at a CLI, before any LLM is attached. When a game goes weird you must be able to trust
   the referee absolutely. The LLM is a drop-in against an already-working interface.

---

## 2. What we're curious about

Not pre-registered hypotheses — just the questions that make this fun to build:

- **What does emergent bluffing look like?** How do LLM bad guys lie when nothing tells
  them how? Does it resemble the model's uniform lie, or something structured?
- **Does an LLM that's *shown* the assistant panel actually use it?** Trust it, ignore it,
  over-trust it? (The panel becomes an optional input block, §7 — an experiment we can
  run, not the spine of the project.)
- **Does the assistant stay calibrated when real agents play off-model?** Emergent play
  is the natural stress test for [model.md §3.6](../docs/model.md); the engine holds
  ground truth, so we can actually plot `P(bad)` reliability against reality.
- **Who wins, and why?** Win-rates and the transcripts behind them — the balanced-game
  curiosity.

These are read **offline from the event log** (§8), so we can ask new questions of old
games without re-running anything. No A/B harness, no power analysis — if a number looks
interesting we'll chase it then.

---

## 3. The information firewall (architecture core)

Three participants, three disjoint information sets — three dataclasses:

```
GroundTruth   (engine-only)   roles[], per-round wires[]/bombs[], the deal seed
PublicState   (everyone)      declarations, the cut log, revealed[]/found[] this round,
                              H, active wires (round-start + current), N, B-prior
PrivateView   (one agent)     my_index, my_role, my_wires_this_round, i_hold_bomb
```

- **Engine / referee** — holds `GroundTruth` + `PublicState`. Deals, collects
  declarations, resolves cuts against the hidden deal, judges win/loss, advances rounds.
  Makes **no** strategic choice and tracks **no** belief. Reuses `General.DistributeWires`
  so the hidden deal obeys the same law the inference assumes
  ([model.md §3.1/§3.4.1](../docs/model.md)). It does **not** reuse `PlayAuto`'s loop
  (which bakes in `CutRandom` + an omniscient belief tracker).
- **Agent** — receives `AgentView = PublicState ⊕ PrivateView ⊕ assistant_panel?`. Decides
  its own declaration and, when it holds the cutters, its cut. Never sees another hand,
  role, or the bomb — those live only in `GroundTruth`, which it is never handed.
- **Assistant** (optional, the unit under study) — receives **only** `PublicState`; a thin
  adapter over `General.py` producing the `CutPanel` + `P(bad)`/`P(bomb)`. Reference
  implementation: the public-info path of `General.Play` (`General.py:735`).

> A unit test builds an `AgentView` from a game with a known hidden deal and asserts it
> carries no `GroundTruth` field. The firewall is checked, not trusted.

---

## 4. Module layout (`sim/`)

| Path | Responsibility |
| ---- | -------------- |
| `sim/SPEC.md` | this document |
| `sim/state.py` | the three dataclasses (§3), `AgentView`, the state→text renderer, the event log |
| `sim/engine.py` | the referee: deal → declare → cut-loop → judge → next round |
| `sim/agents/base.py` | the `Agent` interface (§5) |
| `sim/agents/programmatic.py` | `RandomAgent`, `HumanAgent` — sandbox drivers, not statistical controls |
| `sim/agents/llm.py` | `LLMAgent` — the point of the project (§6) |
| `sim/assistant.py` | public-state → panel adapter over `General.py` (optional input, §7) |
| `sim/prototypes/` | throwaway vertical slices; first one is `cut_decision.py` |
| `sim/analysis/` | scripts that turn event logs into figures (bluff taxonomy, calibration) |
| `sim/run.py` | CLI: play N games with a given agent roster, write logs |
| `tests/test_arena_engine.py` | engine correctness + the firewall assertion (runs in CI) |

---

## 5. The agent interface

One small interface, shared by every tier and by the human CLI:

```python
class Agent:
    def declare(self, view: AgentView) -> int:
        """A wire count in [0, hand_size]. Free to bluff."""
    def choose_cut(self, view: AgentView) -> int:
        """Index of the player to cut. Must be someone else with a face-down card."""
    def observe(self, event: Event) -> None:
        """Optional: called for every public event, for stateful agents."""
```

The engine validates every returned action (declaration in range; cut target legal) and
applies a safe fallback on malformed output — one bad LLM parse must never crash a batch.
`RandomAgent` and `HumanAgent` exist to exercise and verify the engine before the LLM
arrives (Principle 3).

---

## 6. The LLM agent (the actual point)

Statistical power is a non-goal here; *interesting behaviour* is the goal. Starting
defaults, chosen to keep the first version debuggable:

- **Memory: one session per player.** Each agent keeps a single persistent `claude -p`
  session, resumed every turn (`--resume`), so the model carries its own running memory.
  The first turn seeds full context (`render_agent`); later turns send only the new events
  (`render_session_delta`) and the unchanged prefix is served from the prompt cache —
  cheap input, no re-serialising the whole history each call. (This replaces the original
  stateless sketch, which re-sent everything every prompt.)
- **Action protocol: structured output.** The model returns JSON — a `reasoning` string
  plus the action — via tool-use. We **log the reasoning** (never feed it to other agents):
  that field is the whole window into emergent strategy.
- **Failure handling, decided up front.** Illegal or unparseable action → retry once →
  fall back to a legal default. At scale it *will* happen.
- **Model tiering.** A fast model (`claude-haiku-4-5`) for bulk play; `claude-opus-4-8`
  reserved for a small high-quality qualitative deep-dive. Per-run token budget cap.
- **Prompt scaffold:** four XML-tagged sections (Claude models respect tag-delimited
  structure), each fact stated once — `<rules>` (the static preamble, with a
  `wire`/`dud`/`bomb` glossary) · `<your_role_and_hand>` (role + win condition + private
  hand) · `<public_record>` (per round: declarations *claimed*, table talk *said*, cuts
  *revealed*) · `<decision>` (the legal action ask) · (optionally) the assistant readout
  (§7). Prompts must not steer strategy — only rules, state, and talk; a discuss turn may
  return `""` to stay silent (a rules-faithful affordance, not a nudge).
  All copy lives in `sim/prompts.py`; players are referred to by **name only** (a cut
  target is returned as a name and resolved back to a seat index by the agent).
- **Context purity.** A player's context contains *nothing* but our persona system prompt
  and the rendered game state: `--tools ""` (no built-in tools), `--setting-sources ""`
  (no user/project settings), `--strict-mcp-config` (no MCP servers), and a neutral cwd
  (no CLAUDE.md auto-injection). ~166 fixed overhead tokens vs ~11.5k with the default
  Claude Code harness.

**Table-talk is in scope (the discussion beat).** Each round runs declare → **discuss** →
cut: after the (blind, simultaneous) declarations, every player makes one public statement
in seating order — a claim, a read, an accusation, a defense, or a bluff — before any cut.
The cutter also speaks at each cut. Talk is free-text (accusation accuracy is measured
offline, §2); nothing forces a player to suspect anyone. The assistant still consumes only
declarations + cut results, so the firewall is unchanged — table talk is extra public log,
never fed to the assistant.

---

## 7. The assistant as an optional input

The assistant panel isn't the spine of the project — it's one of the things we can hand an
agent and watch what happens. When an agent is flagged to receive it, the engine computes
the panel from `PublicState` only (firewall intact) and appends it to that agent's view.
The agent class is written once; "uses the assistant" is just "reads `view.assistant_panel`
when present." Whether to give it to good guys, bad guys, or everyone is a knob, not a
fixed experimental arm.

---

## 8. The event log (single source of replay)

The engine emits one append-only structured log per game — every deal, declaration, cut
(target + result), the agent roster (`game_start.agents`: model, instructions, knobs of
each player), and LLM reasoning string. **Every** question in §2 is answered offline from
this log; the live loop computes no statistics. This is what makes the project a sandbox
rather than a fixed experiment — add a metric, re-read old logs.

**On-disk layout.** One run = one directory `sim/logs/<label>/`, holding a `manifest.json`
index plus a `g<NNN>_s<seed>_<good|bad>.{jsonl,md}` pair per game. The complete, versioned,
analysis-oriented contract — directory layout, manifest schema, every event type and field,
the agent-config object, and the extensibility/migration rules — is **[LOGS.md](LOGS.md)**.
It is stable and stamped with `SCHEMA_VERSION`; old logs are migrated, never deleted.

---

## 9. Milestones & open decisions

- **M0 — Engine sandbox.** `state.py`, `engine.py`, `RandomAgent`/`HumanAgent`, the event
  log, `tests/test_arena_engine.py` (win/loss, cut resolution, round flow, firewall). Exit:
  you can play a full game by hand at the CLI and the engine tests are green. *(Sketched.)*
- **M1 — Cut-decision prototype.** One hand-built mid-round state, one LLM call for the
  cut, structured output + validation + fallback, reasoning printed. The smallest thing
  that exercises state→text and text→action. *(Sketched: `sim/prototypes/cut_decision.py`.)*
- **M2 — Declaration prototype + a full LLM game.** Add the declaration decision (where
  bluffing is *generated*), then run one all-LLM game end to end, logging everything.
- **M3 — Observe.** *(In progress.)* `sim/analysis/analyze.py` reads a run directory and
  reports outcomes, **bluffing** (declared − true by hidden role, with good-with-bomb broken
  out), and **cut behaviour** by role — all from the logs, no model calls. Early signal: good
  guys with no bomb declare truthfully (0% lie), good-with-bomb and bad guys over-declare,
  and bad guys' cuts reveal far fewer wires — i.e. the model's behavioural assumptions and
  the §3.6 strategic-lie pattern show up empirically. *Still to build:* assistant
  **calibration** under emergent play (a `General.py` replay over each log's public events —
  does `P(bad)` stay honest when agents lie off-model?). Feed anything surprising back to
  `docs/decisions/` if it bears on the deferred strategic-lie model (§3.6).

The **default play configuration is the standard game: 6 players, 2 bad, 1 bomb**
(`NUM_BAD_PRIOR(6) = {2: 1.0}`, so the bad count is fixed). `N=4`/`N=7` — where the bad
count is itself uncertain (model.md §3.5.1) — are supported but secondary.

**Open decisions (record as ADRs when reached):** per-game vs. per-match memory; bulk-LLM
model + token budget; cutter-passing rule fidelity (§ engine sketch); multi-pass discussion
/ rebuttals (today: one statement per player per round). *(Done: table-talk channel — the
discussion beat, §6; session memory, §6.)*

---

## 10. Non-goals

- **Not** re-validating the backend math — that's `tests/` against the `math.comb` oracle.
  The arena assumes `General.py` is correct.
- **Not** an A/B value-of-information study with power analysis — this is exploratory.
- **Not** the RL agent — `AI.py` is broken and off-limits.
- **No hidden-state leakage, ever** — any metric needing ground truth is computed by the
  engine/analysis layer from the event log, never handed to an agent or the assistant.
