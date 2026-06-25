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

`SCHEMA_VERSION` lives in `sim/state.py` and is stamped into both the manifest and every
game's `game_start` event.

---

## 1. Directory layout

One **run** (one `run.py` invocation) = one directory under `sim/logs/`:

```
sim/logs/<label>/
  manifest.json              # run-level index (§2)
  g000_s0_bad.jsonl          # game 0 event log (§3) — the canonical machine record
  g000_s0_bad.md             # game 0 transcript (§4) — the human read
  g001_s1_good.jsonl
  g001_s1_good.md
  ...
```

- **`<label>`** — `--label`, or the default `<YYYYMMDD-HHMM>_<agent>` (e.g.
  `20260620-1745_llm`). One label per experiment; reruns get fresh labels.
- **Game file stem** — `g<NNN>_s<seed>_<verdict>` where `NNN` is the zero-padded game
  index, `<seed>` is the integer seed or `rand` (no seed), and `<verdict>` is `good` or
  `bad` (which team won). The stem appears in `manifest.games[].file`.

Logs are git-ignored (`sim/logs/`); they are data, not source.

---

## 2. `manifest.json` — the run index

The entry point for analysis: read this first, then open the games it points to.

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `schema_version` | int | Format version (see header). |
| `label` | str | Run directory name. |
| `created` | str | ISO-8601 local timestamp. |
| `notes` | str \| null | Free-text run note (`--notes`); the place to add run-level context. |
| `params` | object | `{players, agent, games, base_seed}` — the invocation parameters. |
| `agent_config` | object | The agent roster's config (§5). Homogeneous rosters today, so one object. |
| `games` | array | One entry per game: `{idx, seed, good_guys_won, reason, file}`. |

`reason` ∈ `{"all wires cut", "bomb detonated", "out of time"}`. `file` is the stem (§1),
no extension.

**Extensibility:** add new run-level keys at the top level (e.g. a future `roster` array
for heterogeneous agents) or inside `params`/`notes`. Never repurpose an existing key;
bump `schema_version` only if an existing key's meaning or presence guarantee changes.

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

### `cut`
| Field | Type | Meaning |
| ----- | ---- | ------- |
| `round` | int | Round index. |
| `cutter` | int | Player holding the cutters (chose the target). |
| `target` | int | Player whose card was cut (always `!= cutter`, had a face-down card). |
| `result` | str | `"wire"`, `"dud"`, or `"bomb"`. (Pre-rename logs used `"active wire"`/`"blank/inactive"`/`"BOMB"` — see §7.) |
| `reasoning` | str \| null | The cutter's **private** reasoning (LLM); null otherwise. |
| `message` | str \| null | The cutter's **public** table-talk — one short line said to everyone, shown in every later context. Null for non-speaking agents. |

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

## 4. `*.md` — the transcript (human read)

A rendered, chronological Markdown view of the same events: a header (players, revealed
roles, result, and the **agent roster**, §5), then per round the hidden deal, each
declaration with its reasoning and true count, and each numbered cut with its reasoning and
result. Derived from the `.jsonl`; never the source of truth for analysis.

---

## 5. Agent config object (`agent_config` / `game_start.agents[]`)

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

The format is designed so any metric is a fold over events. Recommended derived tables
(see `sim/analysis/analyze.py`, which builds these and exports CSVs):

- **Per game** — `{label, idx, seed, num_players, num_bad, good_guys_won, reason}`. For
  win-rate, broken down by `num_bad` or agent config.
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
