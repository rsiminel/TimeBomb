# Data Model: Hosted Local Time Bomb Game

Phase 1 output. Entities as they exist in the shared engine (`tbgame/`) and the web
service layer. The engine's event log is the single source of truth; everything else
is derived or transient.

## Engine entities (tbgame/)

### GameSetup
| Field | Type | Rules |
|---|---|---|
| seats | list[Seat] | 4–8 entries; ≥0 humans (0 = exhibition mode) |
| role_deal | `official` \| int override | override must be feasible for seat count (v1 rule: 1 ≤ b ≤ seats−1); infeasible → rejected at creation |
| seed | int \| null | fixes role sample + every round's deal + cut resolution draws (engine RNG only) |
| panel_allowed | bool | default false; immutable after creation (future: second flag `private_panel_allowed`, reserved) |

### Seat
| Field | Type | Rules |
|---|---|---|
| index | int | 0-based table position |
| name | str | non-empty, unique after trim (v1 rule) |
| occupant | `human` \| `solver_bot` \| `llm:<model-tag>` | LLM occupants only accepted when a key is configured |

### GroundTruth (per round; engine-only — never in any view)
roles[], wires[], bombs[], num_bad, seed — as today in `sim/state.py`, moved verbatim.

### PublicState (snapshotted into every view)
As today in `sim/state.py`, moved verbatim, plus:
| Field | Type | Notes |
|---|---|---|
| claim_log | list[Claim] | replaces sim's free-text `discussion_log` in web games (arena keeps `discussion_log`) |
| phase | `awaiting_declarations` \| `awaiting_cut` \| `finished` | derived from event log |
| pending | Pending | who owes what decision now |

### Pending
| Field | Type | Notes |
|---|---|---|
| kind | `declare` \| `cut` | |
| seats | list[int] | declare: all seats without a declaration this round (simultaneous); cut: exactly [current_cutter] |

### Claim
| Field | Type | Rules |
|---|---|---|
| round | int | |
| speaker | int | must equal the acting seat of the carrying intent (attribution inherent) |
| kind | `trust` \| `distrust` \| `accuse_lie` \| `self_honest` | fixed menu |
| target | int \| null | required for directed kinds; ≠ speaker; null for `self_honest` |

### Intent (what `TableGame.submit` accepts)
| Field | Type | Rules |
|---|---|---|
| seat | int | must be in `pending.seats` |
| kind | `declare` \| `cut` | must equal `pending.kind` |
| value | int | declare: 0 ≤ v ≤ hand_size; cut: a `legal_targets` entry |
| claim | Claim \| null | optional rider, validated per Claim rules |

Illegal intents are rejected whole (no partial application, no fallback — fallbacks
are a batch-arena concern that stays in the compatibility driver).

### Event log (append-only; save/replay format, schema_version bumped from sim's)
Event kinds and payloads follow `sim/LOGS.md` with these deltas:
- `game_start` gains `seats` (occupant kinds) and `panel_allowed`.
- `declaration` keeps `true_wires` (basis for replay lie-annotation).
- `claim` replaces `statement` in web games: `{round, speaker, kind, target}`.
- `cut`, `round_start`, `round_end`, `cut_skipped`, `game_end` as today.
Existing sim logs are never deleted or rewritten (repo rule); the new schema is a new
version, old logs stay readable by their own tools.

### State transitions
```
created ──deal──▶ awaiting_declarations ──all declared──▶ awaiting_cut
   ▲                                                        │ cut resolved:
   │                                                        │  bomb → finished(bad win)
   └── round_end (hands shrink, re-deal) ◀── cuts exhausted ┤  last wire → finished(good win)
                                                            │  else next cutter
              final round exhausted → finished(bad win) ◀───┘
```
Terminal state exposes full GroundTruth (reveal) — the only time it leaves the engine.

## Web service entities (web/)

### ActiveGame (in-memory, exactly 0 or 1)
| Field | Type | Notes |
|---|---|---|
| game | TableGame | authoritative state |
| version | int | monotonically increasing; bumped on every accepted event; stale-echo submissions rejected |
| unlocked_seat | int \| null | which human seat's private view is currently unlocked (pass-the-device ack); reset on refresh/lock/any accepted intent |
| thinking | list[int] | AI seats currently deciding (worker-owned) |
| paused_llm | PauseInfo \| null | failed LLM seat awaiting retry/substitute decision |

### SaveFile (`web/saves/<name>.json`)
| Field | Type | Rules |
|---|---|---|
| schema_version | int | mismatch → clear rejection, file untouched |
| name | str | filesystem-safe, unique per save |
| created | ISO datetime | |
| setup | GameSetup | |
| events | list[Event] | full log incl. sealed deal; reduces back to an identical TableGame |

### Settings (`web/instance/settings.json`)
| Field | Type | Rules |
|---|---|---|
| llm_api_key | str \| null | env var wins when set; write-only via API (never echoed back) |

## View objects (the only shapes a browser receives)

- **TableView** (shared surface): PublicState minus nothing-it-already-excludes —
  declarations, claim log, cut log, revealed/found counts, phase, pending, thinking
  flags, panel availability, version. Never any role/hand/bomb field.
- **PrivateView** (one seat, only while `unlocked_seat == seat`): role, own wire
  count, bomb possession — the engine's `PrivateView`, verbatim.
- **PanelView** (only if `panel_allowed`): `replay.py` panel output for the public
  record, verbatim (per-player p_bad / p_bomb / p_safe / info stat + flags).
- **RevealView / ReplayView** (only when `phase == finished`, or any time in
  exhibition games): roles, hands, and the event list with truth annotations
  (declaration → `declared` vs `true_wires`).
