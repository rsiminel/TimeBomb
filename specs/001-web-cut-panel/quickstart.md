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

## Validation run — 2026-07-03 (T022)

- Automated suite: **62 passed** (`web/tests/`: replay 37, parity 6, api 16, latency 3).
- Scenarios 1–5 driven over live HTTP against `web/app.py` on this machine: first
  panel 0.35 s, cut update 0.32 s (approx flag set), round-2 belief differs from a
  fresh game given the same entries, bomb ends the game, truncation-undo restores the
  pre-bomb response. All as specified.
- Scenario 6 (reload) is client-side by construction: the record lives in
  `localStorage` and the page replays it on load.
- Latency: N≤7 within 2 s; N=8 ~2.1 s → SC-002 amended to 3 s at 8 players (solver
  floor, research R7).
- Visual pass: headless Firefox screenshots were blocked by snap confinement on this
  machine, so the screens were previewed via a verbatim HTML replica (real markup +
  stylesheet + live solver numbers) shared as an artifact; an in-browser pass on a
  real phone at a real table remains the owner's acceptance step.
