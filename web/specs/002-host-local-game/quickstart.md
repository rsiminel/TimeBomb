# Quickstart: Hosted Local Time Bomb Game (002)

Runnable validation scenarios proving the feature end to end. Prerequisites: the
project venv (`.venv`, see repo README) and a browser.

## Run the app

```bash
cd /path/to/TimeBomb
.venv/bin/python web/app.py            # http://127.0.0.1:5000
```

## Scenario A — home page and untouched v1 (US3, SC-004)

1. Open `http://127.0.0.1:5000/` → home page with two doors.
2. Open the assistant door → v1 at `/assistant`; run a v1 flow (setup → declarations
   → cut → panel → undo). Expected: identical v1 behavior.
3. Regression: `.venv/bin/python -m pytest web/tests -q` — the v1 suite passes.

## Scenario B — solo game vs solver bots (US1, SC-001/002)

1. Home → Play → setup: 5 seats, seat 0 human, 4 × solver bot, official role-deal,
   seed `42`, panel opt-in **off**. Start.
2. Expected: bots declare within ~2 s; you are prompted to declare; cut phase shows
   legal targets only; card-flip reveals results; rounds advance 5→4→3→2 hands; the
   game announces the winner and reveals all roles.
3. Restart with the same setup and seed `42` → the deal (your role/hand) is
   identical; bot play may differ.

## Scenario C — hotseat privacy (US2, SC-003)

1. New game: 4 seats, 3 humans + 1 bot.
2. Expected: every private action is preceded by a named pass-the-device screen;
   role/hand appear only after tap-to-reveal; hide returns to the neutral table.
3. With devtools open (Network tab), play a round. Expected: no response body ever
   contains another seat's `role`, `wires`, or bomb fields (automated:
   `web/tests` privacy test asserts this over a scripted game).
4. Refresh mid-game → back at the shared table, privacy re-locked, no progress lost.

## Scenario D — opt-in panel parity (US4, SC-008)

1. New game with panel opt-in **on** → a side-drawer toggle exists (collapsed by
   default); open it after declarations. Expected: the four v1 stats per seat.
2. Parity check (automated): `web/tests` panel-parity test replays the same public
   events through `POST /api/panel` (v1) and `GET /api/game/panel` and asserts
   numeric identity.
3. New game with opt-in **off** → no panel control anywhere.

## Scenario E — save, restart, resume (US6, SC-007)

1. Mid-game (round 2), save as `friday-night`.
2. Stop the server (Ctrl-C), start it again, home → resume `friday-night`.
3. Expected: same round, same declarations, same pending turn; hidden info still
   hidden; play continues to a normal finish.

## Scenario F — post-game reveal & replay (US5, SC-009)

1. Finish any game. Expected: winner banner + all roles/hands revealed.
2. Enter replay; step backward/forward. Expected: every declaration annotated
   truthful/lie against the dealt hand; claims and cuts in order.

## Scenario G — exhibition mode (clarification Q5)

1. New game: 5 seats, all solver bots, panel opt-in on.
2. Expected: the table plays itself with everything open (roles, hands); useful as a
   live view of self-play.

## Scenario H — LLM seats (US7), optional

1. Without a key: setup shows LLM seats unavailable with a hint.
2. `export ANTHROPIC_API_KEY=...`, restart, assign one LLM seat, play a round.
   Expected: thinking indicator during its turns; on repeated failure a
   retry/substitute dialog appears.

## Engine & bot suites (SC-005/006 and arena regression)

```bash
.venv/bin/python -m pytest tbgame/tests web/tests -q   # new suites
.venv/bin/python -m pytest -q                          # frozen root suite still green
```

Expected: self-play (100 seeded games) completes with zero illegal moves and correct
judgments; the beats-random gate passes; `tests/test_arena_engine.py` passes
unmodified against the shims.
