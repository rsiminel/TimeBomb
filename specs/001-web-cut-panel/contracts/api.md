# HTTP Contract: Time Bomb Web Cut Panel

One stateless endpoint. The browser sends the full game record; the server replays it
and returns everything the page renders. There is no other API surface.

## `GET /` (and static assets)

Serves `web/static/` (`index.html`, `main.js`, `styles.css`). No parameters.

## `POST /api/panel`

### Request body

```json
{
  "setup": {
    "players": ["Alice", "Bob", "Clara", "Darryl"],
    "bomb": true,
    "numBadOverride": null
  },
  "events": [
    { "type": "declarations", "values": [2, 1, 1, 1] },
    { "type": "cut", "player": 2, "result": "safe" }
  ]
}
```

### Success response — `200`

```json
{
  "state": {
    "round": 1,
    "handSize": 5,
    "activeWires": 3,
    "cutsMade": 1,
    "cutsThisRound": 4,
    "revealed": [0, 0, 1, 0],
    "found": [0, 0, 1, 0],
    "awaiting": "cut",
    "gameOver": null
  },
  "belief": {
    "pBad": [0.31, 0.22, 0.19, 0.28],
    "pNumBad": { "1": 0.4, "2": 0.6 },
    "panel": [
      { "pSafe": 0.42, "pBomb": 0.12, "onePly": 0.91, "horizon": 0.55, "noCards": false }
    ],
    "approx": true,
    "maxDepth": 3
  },
  "warnings": []
}
```

Field semantics:

- `state.*` — rules bookkeeping derived from the replay (see data-model.md). `awaiting`
  is `"declarations" | "cut" | "over"`. `gameOver` is `null` or
  `{ "winner": "good" | "bad", "reason": "bomb" | "wires" | "time" }`.
- `belief.pBad`, `belief.pNumBad` — from `JointBadBelief` (current round folded in
  provisionally, as in `General.Play`). `pNumBad` has one key when the count is fixed.
- `belief.panel[i]` — row i of `CutPanel` untouched; `noCards: true` replaces the four
  numbers (solver NaN row) when player i has no face-down card.
- `belief.approx` — the round-horizon stat was depth-capped; the UI must label it.
- `belief` is `null` before the first declarations event. Between rounds and after a
  time-out ending (no round in progress), `belief.pBad`/`pNumBad` reflect the
  accumulated completed-round evidence and `belief.panel` is `null` — there is no
  live round to compute cut stats for.
- `warnings` — array of `{ "code": "impossible_declarations" | "impossible_cut",
  "message": "..." }`; the replay emits each at most once per round, mirroring the
  `Consistency` behaviour of the interactive `Play` loop.

### Error response — `422`

The record itself is invalid (violates data-model.md constraints — bad setup values,
out-of-range declaration, cut on an empty hand, event after game over, wrong event
order). The body pinpoints the first offending event so the client can offer undo:

```json
{ "error": "cut on player with no face-down cards", "eventIndex": 7 }
```

`eventIndex` is `-1` when the setup itself is invalid. Anything the UI prevents can
still arrive from a stale/edited localStorage record, so the server validates
everything.

### Error response — `400`

Malformed JSON / missing fields. Body: `{ "error": "..." }`.

## Contract invariants (tested in `web/tests/`)

1. **Solver parity (SC-003)**: for scripted games, every number in `belief` equals the
   value from calling `General.py` directly with the same inputs — byte-identical
   floats, no rounding server-side.
2. **Replay purity**: identical request bodies produce identical responses (no server
   state); a truncated log (undo) equals the response the shorter game had.
3. **Warning discipline**: impossible entries yield `200` with `warnings`, never `422`
   (they are legal inputs that are jointly impossible — the solver's fallback applies);
   malformed/rule-breaking entries yield `422`, never a crash.
