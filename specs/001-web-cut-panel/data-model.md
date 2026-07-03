# Data Model: Time Bomb Web Cut Panel

The whole game is an immutable **setup** plus an append-only **event log**. Everything
else — rounds, hand sizes, wire counts, whose hand is empty, game over — is *derived*
by replaying the log. Undo is truncation; reload is replay. The browser owns the stored
copy; the server derives state per request and stores nothing.

## Stored entities (browser `localStorage`)

### GameSetup

| Field | Type | Constraints |
|-------|------|-------------|
| `players` | array of string | 4–8 names, non-empty after trimming, unique |
| `bomb` | boolean | official play is `true`; `false` supported |
| `numBadOverride` | int or null | null = official role-deal for the player count (bad count may be uncertain, e.g. N=4 → 1 or 2); int = manual fixed count, must be one the solver accepts for this N |

### Event (one per user entry, append-only)

| Variant | Fields | Constraints |
|---------|--------|-------------|
| `declarations` | `values`: array of int | one per player, each in `[0, hand_size]` for the round it opens; only legal as the first event of a round |
| `cut` | `player`: int (seat index), `result`: `"safe" \| "nothing" \| "bomb"` | player must have a face-down card in the derived state; only legal after the round's declarations and before the round's cut budget is spent; illegal after game over |

### GameRecord (the persisted unit)

| Field | Type |
|-------|------|
| `setup` | GameSetup |
| `events` | array of Event |

## Derived state (recomputed on every replay — never stored)

| Quantity | Derivation (rules bookkeeping only, no probabilities) |
|----------|--------------------------------------------------------|
| round number | number of `declarations` events so far (round r hand size = 5, 4, 3, 2) |
| `hand_size` | `5 - (round - 1)`; the game ends after the hand-size-2 round |
| `total_active` (round start) | `active_wires` remaining when the round began; starts at N |
| `revealed[i]`, `found[i]` | per-round tallies from `cut` events (`safe` increments `found` and decrements `active_wires`; `nothing` only increments `revealed`) |
| round complete | N cuts entered, or game over mid-round |
| game over | `bomb` cut (bad guys win) · `active_wires == 0` (good guys win) · hand-size-2 round complete with wires left (bad guys win) |

## State transitions

```text
SETUP ──(valid GameSetup)──▶ AWAITING_DECLARATIONS(round 1)
AWAITING_DECLARATIONS ──(declarations event)──▶ CUTTING
CUTTING ──(cut event, budget left, no end)──▶ CUTTING
CUTTING ──(round budget spent, hand_size > 2)──▶ AWAITING_DECLARATIONS(next round)
CUTTING ──(bomb / last wire / final round exhausted)──▶ GAME_OVER
any state ──(undo = drop last event)──▶ replay of the shorter log
any state ──(new game)──▶ SETUP (stored record discarded)
```

## Response projection (from the solver, per replay)

All numeric fields below are raw solver outputs, passed through untouched
(Constitution I). See [contracts/api.md](contracts/api.md) for the wire format.

| Field | Solver source |
|-------|---------------|
| `p_bad[i]` | `JointBadBelief` over accumulated `RoundLogU` (completed rounds + current round provisionally), as in `General.Play` |
| `p_num_bad` | same `JointBadBelief` call |
| `panel[i] = [p_safe, p_bomb, one_ply, horizon]` | `CutPanel(..., max_depth)` on the current-round belief (`ProbDeclaration` → `ProbCut`) |
| `approx` | true when the round-horizon stat was depth-capped (`max_depth` < cuts remaining in the lookahead) |
| `warnings` | `Consistency` feasibility check on declarations; impossible-cut signal (`ProbCut` returning its prior); each at most once per round |
