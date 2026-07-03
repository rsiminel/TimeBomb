# Quickstart: Time Bomb Web Cut Panel

Validation guide — how to run the assistant and prove the feature works end-to-end.
Wire format details: [contracts/api.md](contracts/api.md); state rules:
[data-model.md](data-model.md).

## Prerequisites

- The repo's `.venv` with Flask added:

  ```bash
  .venv/bin/pip install flask
  ```

## Run

```bash
.venv/bin/python web/app.py            # serves http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in a browser. To use a phone at the table, bind to the
LAN (`--host 0.0.0.0`) and open `http://<machine-ip>:5000` from the phone.

## Validation scenarios

### 1. First panel (US1 / SC-001)

1. Set up a 5-player game with default (official) roles and the bomb in play.
2. Enter five declarations.
3. **Expect**: the panel renders four numbers per player plus P(bad); nothing marks a
   "recommended" cut; total time from page open comfortably under 2 minutes.

### 2. Cut updates and latency (US1 / SC-002)

1. Record a cut (`safe`) on any player.
2. **Expect**: the panel changes and re-renders in under 2 seconds; the info stat shows
   its "approximate" label whenever the response says the lookahead was depth-capped.

### 3. Multi-round accumulation (US2)

1. Enter the round's full cut budget; advance; enter round-2 declarations.
2. **Expect**: hand size drops to 4; round-2 P(bad) differs from a fresh game given
   the same round-2 entries (evidence carried over).

### 4. Game end (US2)

1. Enter a `bomb` cut result.
2. **Expect**: "bad guys win" banner; entry controls disabled; new-game still offered.

### 5. Mistakes and undo (US3)

1. Try declaring 7 with hand size 5 → rejected inline, no request needed.
2. Enter jointly impossible declarations (e.g. everyone declares 5 of 5) →
   warning shown once for the round; panel still renders.
3. Press undo repeatedly → each press steps back exactly one entry, to the start.

### 6. Reload survival (SC-005)

1. Mid-round, reload the page.
2. **Expect**: the same game state and panel reappear (log replayed from
   localStorage).

## Automated checks

```bash
.venv/bin/python -m pytest web/tests -q          # web-layer suite
.venv/bin/python -m pytest web/tests/test_solver_parity.py -q   # SC-003 parity only
```

Expected: all green; parity tests assert exact float equality between API responses
and direct `General.py` calls over scripted games (no re-testing of the math itself —
Constitution, Constraints).
