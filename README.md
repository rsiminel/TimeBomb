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
| `timebomb/`         | The backend solver **`General.py`** (arbitrary bad-guy count + bomb, joint inference over the bad count), the shared `UsefulFunctions.py` / `Consistency.py`, and the `AI.py` RL agent (on hold). Also the four **frozen** hardcoded variants (`OneBadGuyNoBomb`, `TwoBadGuysNoBomb`, `OneBadGuyOneBomb`, `TwoBadGuysOneBomb`) — independent reference oracles that cross-check `General.py`; **do not modify them** (see [CLAUDE.md](CLAUDE.md)). |
| `tests/`            | Test suites (independent `math.comb` brute-force references).               |
| `web/`              | Flask API + browser "Time Bomb Assistant" UI, on hold.                      |
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
