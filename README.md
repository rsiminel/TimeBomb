# Time Bomb Assistant

A probabilistic assistant and AI for the social-deduction board game
[**Time Bomb**](https://www.daysofwonder.com/). From only the *public* information
at the table — each player's declared wire count and the result of each cut — it
infers, for every player, the probability that they are a Terrorist, that they
hold the Bomb, and that cutting one of their cards is safe, and recommends the next
cut.

See **[docs/model.md](docs/model.md)** for the mathematics and
**[docs/roadmap.md](docs/roadmap.md)** for the plan and status.

## Repository layout

| Path                   | What it is                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| variant `*.py`         | The four hardcoded solver variants plus the canonical `General.py`. See [docs/roadmap.md](docs/roadmap.md) for the variant list, configs, cleanup order, and status. |
| `UsefulFunctions.py`   | Shared combinatorics helpers (factorials, binomials, hypergeometric).      |
| `AI.py`                | REINFORCE cut agent (TensorFlow/Keras), on hold.                           |
| `web/`                 | Flask API + browser "Time Bomb Assistant" UI, on hold.                     |
| `test_*.py`            | Test suites (independent `math.comb` brute-force references).              |
| `docs/`, `TODO.md`     | Model reference, roadmap, decision records (ADRs), and open work items.    |

## Playing

Each variant exposes two entry points:

- `Play(players, initial_hand_size=5)` — interactive: enter declarations and cut
  results at the prompt; the assistant prints the updated belief state.
- `PlayAuto(num_players, initial_hand_size=5, verbosity=2)` — simulate a full game
  with random play (useful for testing and benchmarking strategies).

```bash
# Simulate a 5-player game of the simplest variant
python3 -c "from OneBadGuyNoBomb import PlayAuto; PlayAuto(num_players=5, verbosity=1)"
```

## Testing

The environment is externally managed (PEP 668), so tests run against a
project-local virtualenv (`.venv`, git-ignored) created with
`--system-site-packages` to inherit the system `numpy`, with `pytest` + `scipy`
added on top. One-time setup:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install pytest scipy
```

Run the tests (each test file also runs standalone without pytest):

```bash
.venv/bin/python -m pytest test_OneBadGuyNoBomb.py -q   # or:
.venv/bin/python test_OneBadGuyNoBomb.py
```

Test suites reimplement the hypergeometric likelihood, the posterior, and the
expected-wire calculation independently (via `math.comb`) so a shared bug cannot
hide behind a matching assertion.
