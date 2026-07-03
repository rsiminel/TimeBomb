# Time Bomb Assistant

A probabilistic assistant and AI for the social-deduction board game **Time Bomb**.
From only the *public* information at the table — each player's declared wire count
and the result of each cut — it infers, for every player, the probability that they
are a bad guy, that they hold the Bomb, and that cutting one of their cards is safe.
It is **quantities-only**: it surfaces these calibrated decision inputs and leaves the
cut choice (and the player's own risk appetite) to the human, rather than dictating a
move. See [docs/model.md §3.5](docs/model.md) and
[ADR 0006](docs/decisions/0006-cut-recommendation-output.md).

See **[docs/model.md](docs/model.md)** for the mathematics and
**[docs/roadmap.md](docs/roadmap.md)** for the plan and status.

## Repository layout

| Path                | What it is                                                                  |
| ------------------- | -------------------------------------------------------------------------- |
| `timebomb/`         | The backend solver **`General.py`** (arbitrary bad-guy count + bomb, joint inference over the bad count), the shared `UsefulFunctions.py` / `Consistency.py`, and the `AI.py` RL agent (on hold). Also the four **frozen** hardcoded variants (`OneBadGuyNoBomb`, `TwoBadGuysNoBomb`, `OneBadGuyOneBomb`, `TwoBadGuysOneBomb`) — independent reference oracles that cross-check `General.py`; **do not modify them**. |
| `tests/`            | Test suites (independent `math.comb` brute-force references).               |
| `tbgame/`           | The shared, stepwise game engine (`tbgame.engine.TableGame`, a real Python package): rules, hidden information, event-sourced history. Promoted from the arena referee (specs/002); `sim/state.py`, `sim/engine.py`, `sim/agents/base.py` are re-export shims over it. Also `tbgame/agents/solver.py`, the offline solver bot. |
| `web/`              | The browser app: a hosted Time Bomb game at `/play` (hotseat humans and/or AI seats, transport-only over `tbgame`) plus the v1 assistant at `/assistant` (a static page + one stateless Flask endpoint replaying the event log through `General.py` — no math of its own). See "The web app" below. |
| `docs/`, `TODO.md`  | Model reference, roadmap, decision records (ADRs), and open work items.     |

## Playing

`General.py` (the backend solver) exposes two entry points:

- `Play(players, initial_hand_size=5)` — interactive: enter declarations and cut
  results at the prompt; the assistant prints the updated belief (P(bad), P(bomb),
  P(num_bad)) and the four-stat cut panel, and warns on inconsistent input.
- `PlayAuto(num_players, initial_hand_size=5, verbosity=2)` — simulate a full game
  with random play (useful for testing and benchmarking strategies).

```bash
# Simulate a 5-player game with the backend solver
PYTHONPATH=timebomb python3 -c "from General import PlayAuto; PlayAuto(num_players=5, verbosity=1)"
```

## The web app

Two doors from one home page (phone-friendly). It needs Flask in the venv
(`.venv/bin/python -m pip install flask`), then:

```bash
.venv/bin/python web/app.py            # http://127.0.0.1:5000
.venv/bin/python web/app.py --host 0.0.0.0   # reachable from a phone on the same Wi-Fi
```

**Play** (`/play`, specs/002-host-local-game) hosts the actual game: 4–8 seats, each
human or AI, dealt and adjudicated server-side by the shared `tbgame` engine. Hotseat
humans pass one device behind a privacy screen (secrets never leave the server);
solver bots play offline off `General.py`'s numbers; LLM seats are optional and
key-gated, with retry/substitute recovery when a call fails. Games support structured
table claims, an opt-in public-info stats panel (numerically identical to the
assistant), manual named saves that survive restarts (`web/saves/`), and a post-game
reveal with a truth-annotated replay. An all-AI setup runs as an open-information
exhibition.

**Assistant** (`/assistant`, specs/001-web-cut-panel) is the play-along tool for a
physical table: set up the game (4–8 players; the official role deal — including the
counts the deal leaves uncertain — or a fixed override), then enter each round's
declarations and cut results as they happen. After every entry the page shows, per
player, P(bad), P(bomb), P(safe wire), and the information value of cutting them —
computed by `General.py` exactly as the interactive `Play` loop does. Mistyped entries
can be undone all the way back; jointly impossible table claims warn but never block;
a reload restores the game (state lives in the browser, the endpoint is stateless).

Web tests: `.venv/bin/python -m pytest web/tests -q`. Engine tests:
`.venv/bin/python -m pytest tbgame/tests -q`.

## Testing

The environment is externally managed (PEP 668), so tests run against a
project-local virtualenv (`.venv`, git-ignored) created with
`--system-site-packages` to inherit the system `numpy`, with `pytest` + `scipy`
added on top. One-time setup:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install pytest pytest-xdist scipy
```

Run the tests (each test file also runs standalone without pytest):

```bash
.venv/bin/python -m pytest tests/test_OneBadGuyNoBomb.py -q   # or:
.venv/bin/python tests/test_OneBadGuyNoBomb.py
```

The suite is simulation-heavy but embarrassingly parallel, so `pytest.ini` runs it
across all cores by default (`-n auto`, via `pytest-xdist`) — the full suite is ~70s
instead of ~4.5 min serial. Pass `-n0` for a serial run (e.g. with `-s` or a debugger).

Each suite has two layers. **Correctness:** it reimplements the hypergeometric
likelihood, the posterior, and the expected-wire calculation independently (via
`math.comb`) so a shared bug cannot hide behind a matching assertion. **Predictive
usefulness:** an end-to-end test runs full simulated games and checks the final belief
identifies the true bad guy(s) far above the random baseline — a model can be
arithmetically correct yet uninformative, and only this catches that.
