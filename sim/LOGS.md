# Time Bomb Arena — Log Format (v1)

The on-disk contract for everything `run.py` writes. It is **stable** (we commit to it),
**versioned** (`schema_version`, bumped only on a breaking change), **extensible** (new
fields/event-types are additive — old readers ignore what they don't know), and **built
for statistics** (every fact needed for analysis is in the machine-readable log, with the
hidden ground truth attached, so any metric is derivable offline).

> **Firewall note.** These files contain ground truth (roles, the per-round wire/bomb
> deal, each declaration's true count). That is fine: logs are *post-game analysis
> artifacts*, never shown to an agent. The firewall is about in-game `AgentView`s
> (SPEC.md §3), not about what we record afterwards.

`SCHEMA_VERSION` lives in `sim/state.py` and is stamped into every game's `game_start`
event. (It survived the retirement of the run manifest unchanged — no `.jsonl` event field
changed meaning, so any reader of the event log is unaffected.)

---

## 1. Directory layout

**One run (one `run.py` invocation) = one game.** Running several games means several
invocations, so each run is self-contained — there is no run-level index to keep in sync.
A run is a directory under `sim/logs/`:

```
sim/logs/<label>/
  <label>.jsonl              # the game event log (§3) — the canonical machine record
  <label>.md                # the game transcript (§4) — the human read
  <label>.<Name>.md         # one per LLM player — that seat's raw session (§4b)
  ...
```

- **`<label>`** — `--label`, or the default `<YYYYMMDD-HHMM>_<agent>` (e.g.
  `20260620-1745_llm`). One label per run; reruns get fresh labels. It is both the
  directory name and the stem of every file in it, so files from different runs stay
  distinguishable when several are open at once.
- The **seed** and **outcome** are not in the filenames; they live in the files themselves
  (`game_start.seed`, `game_end`, and the transcript's `**Run:**` / result lines).

**Retired:** earlier runs also wrote a `manifest.json` (a run-level index over multiple
games) and a `run.log` (a tee of stdout). Both are gone now that a run is a single game:
the token/cost usage the manifest carried lives in the transcript's `**Run:**` header line,
and everything else it held is already in the `.jsonl` (`game_start`/`game_end`). Old runs
that still have those files are left as-is (§7).

Logs are git-ignored (`sim/logs/`); they are data, not source.

---

## 3. `*.jsonl` — the game event log (canonical record)

One JSON object per line, in chronological order. **Every** object has:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `i` | int | 0-based sequence index within the game. |
| `type` | str | Event type (below). Readers dispatch on this and ignore unknown types. |

Event types and their additional fields:

### `game_start` (always first)
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `schema_version` | int | Format version. |
| `num_players` | int | N. |
| `num_bad` | int | Realized bad-guy count for this game. |
| `num_bom` | int | Bombs in play (1 in the standard config). |
| `roles` | int[N] | `roles[p]==1` iff player p is a bad guy (fixed all game). |
| `player_names` | str[N] | Display names, indexed by player. |
| `seed` | int \| null | RNG seed (deal + cut resolution). |
| `talk_between_cuts` | bool | Discussion structure: a full pass before **every** cut (true) or one pass per round after declarations (false). Absent in pre-2026-07-04 logs (= false). |
| `talk_top_k` | int \| null | Speak-bid ration: only the k highest urgency bidders get a discuss call each pass (null = everyone is called). Absent in pre-2026-07-06 logs (= null). |
| `panel_for` | int[] | Players shown the `<assistant_readout>` stats panel ([] = nobody). Absent in older logs (= []). |
| `agents` | object[N] | Each player's agent config (§5), indexed by player. |

### `round_start`
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `round` | int | 0-based round index (hand size = `initial_hand_size − round`). |
| `hand_size` | int | Cards per hand this round (H). |
| `active_wires` | int | Safe wires still hidden at the **start** of the round. |
| `wires` | int[N] | **Truth:** active wires actually dealt to each hand this round. |
| `bombs` | int[N] | **Truth:** `bombs[p]==1` iff player p holds the bomb this round. |

### `declaration`
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `round` | int | Round index. |
| `player` | int | Declaring player. |
| `declared` | int | The wire count announced (in `[0, hand_size]`; may be a bluff). |
| `true_wires` | int | **Truth:** that player's real wire count this round. |
| `reasoning` | str \| null | The agent's private reasoning (LLM); null for non-LLM agents. |
| `urgency` | int \| null | The reply's piggybacked speak-bid, 0–9 (see `statement`); null if absent. Absent in pre-2026-07-06 logs. |

### `statement` (table talk)
Emitted in a **discussion pass** — with `talk_between_cuts` the table gets a pass before
*every* cut (the first reacts to the declarations, later ones to the cut just made; each
pass starts at the seat after the next cutter and ends with that cutter). Without it, one
pass per round after declarations, in seating order. With `talk_top_k` set, a pass calls
only the k players with the highest current speak-bid — every reply (declaration, statement,
cut) carries an `urgency` 0–9 field that stands until the player's next reply; skipped
players get no call at all, and the next cutter holds no reserved seat (the cut's own
`message` is their mic). A later speaker has always heard the earlier statements. Silent
agents (e.g. `RandomAgent`, or an LLM returning `""`) emit none. Round flow:
`declaration`s → (`statement`s → `cut`)×N.

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `round` | int | Round index. |
| `player` | int | The speaking player. |
| `message` | str | The **public** statement said to the whole table (claim/read/accusation/defense/bluff). |
| `cuts_before` | int | Cuts already made this round when spoken (0 = the opening pass). Renderers use it to interleave talk with cuts. Absent in older logs (= 0). |
| `urgency` | int \| null | The reply's piggybacked speak-bid, 0–9; null if the reply had none. Absent in pre-2026-07-06 logs. |
| `reasoning` | str \| null | The speaker's **private** reasoning; null otherwise — including LLM statements from 2026-07-06 on, whose replies are message-only. |

### `cut`
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `round` | int | Round index. |
| `cutter` | int | Player holding the cutters (chose the target). |
| `target` | int | Player whose card was cut (always `!= cutter`, had a face-down card). |
| `result` | str | `"wire"`, `"dud"`, or `"bomb"`. (Pre-rename logs used `"active wire"`/`"blank/inactive"`/`"BOMB"` — see §7.) |
| `reasoning` | str \| null | The cutter's **private** reasoning (LLM); null otherwise. |
| `message` | str \| null | The cutter's **public** table-talk — one short line said to everyone, shown in every later context. Null for non-speaking agents. |
| `urgency` | int \| null | The reply's piggybacked speak-bid, 0–9 (see `statement`); null if absent. Absent in pre-2026-07-06 logs. |

### `cut_skipped` (rare)
`{round, cutter}` — the cutter had no legal target; no card was cut.

### `round_end`
`{round}` — the round's N cuts completed without ending the game.

### `game_end` (always last)
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `good_guys_won` | bool | Did the good team win. |
| `reason` | str | `"all wires cut"` / `"bomb detonated"` / `"out of time"`. |
| `roles` | int[N] | Roles again, for convenience when reading from the tail. |

A game ending mid-round (bomb cut, or last wire found) has **no** `round_end` for that
round; the `cut` that ended it is the second-to-last event, before `game_end`.

---

## 4. `<label>.md` — the transcript (human read)

A rendered, chronological Markdown view of the same events: a header (players, revealed
roles, result, a `**Run:**` line with token/cost usage + any `--notes`, and the **agent
roster**, §5), then per round the hidden deal, each declaration with its reasoning and true
count, the discussion, and each numbered cut with its reasoning and result. Derived from the
`.jsonl`; never the source of truth for analysis.

## 4b. `<label>.<Name>.md` — per-agent session (human read)

One file per LLM player: that seat's **raw `claude -p` session**, verbatim — the persona
system prompt, then every committed turn's exact prompt and the reply the model returned.
It is the game from one player's perspective (what it was told, what it answered), so a move
traces back to the precise context behind it. Rendered from the agent's in-memory
`transcript`, not the `.jsonl`; non-LLM agents keep no session and write no such file.

---

## 5. Agent config object (`game_start.agents[]`)

Produced by `Agent.describe()`. Always present:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `type` | str | Class name (e.g. `LLMAgent`, `RandomAgent`). |
| `name` | str | Short agent kind. |

`LLMAgent` adds: `model`, `thinking_tokens` (0 off · >0 cap · <0 Claude Code default),
`session_mode` (true: one persistent `claude -p` session per player, resumed each turn with
only the new events sent — see the `usage` totals for the resulting `cache_read` share),
`timeout`, `retries`, `system` (the persona prompt), and `declare_instruction` /
`cut_instruction` (the exact action prompts). This is what lets a transcript be fully
reproduced/understood later. New agent types extend `describe()` with their own knobs.

---

## 6. Using the logs for statistics

The format is designed so any metric is a fold over events. `sim/analysis/analyze.py` reads
`.jsonl` files directly (globbing a directory recursively — point it at one run or at
`sim/logs/` to aggregate many), never the retired manifest. Recommended derived tables:

- **Per game** — `{seed, num_players, num_bad, good_guys_won, reason}` (all in
  `game_start`/`game_end`). For win-rate, broken down by `num_bad` or agent config.
- **Per declaration** — `{role := roles[player], held_bomb := bombs[round][player],
  declared, true_wires, delta := declared − true_wires}`. For bluffing behaviour by hidden
  role (incl. good-with-bomb).
- **Per cut** — `{cutter_role := roles[cutter], result}` (extend with target role,
  declared/true of target, …). For who finds wires vs wastes cuts.

**Rules for writers and readers**
1. Append-only, chronological; never rewrite a past event.
2. Additive evolution: new fields/event-types only. Optional fields read with a default
   (`e.get("reasoning")`). Dispatch on `type`; ignore unknown types.
3. Bump `SCHEMA_VERSION` (and this doc) only when an existing field changes meaning or a
   guaranteed field is removed — i.e. a change that could break an existing reader.

---

## 7. Migrating older logs

Logs predating a format are **kept, never deleted** — they cost tokens to produce. When
the format changes, write a converter that reads the old shape and emits v-current files
(filling absent fields with explicit nulls / a lower `schema_version`), rather than
discarding the run. Pre-`v1` flat files (`sim/logs/game_*.jsonl`, no run directory) are the
first conversion target.
