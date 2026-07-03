"""Beats-random gate for SolverBot's cut policy (specs/002-host-local-game SC-006).

Reimplements the self-calibrating z-test methodology of tests/baseline_stats.py
locally (the root suite cannot be imported-and-edited, per CLAUDE.md): a one-sided
z-test of a per-game statistic against a no-information baseline, so the threshold
scales with the sample's size and spread instead of a hand-tuned cutoff.

The claim under test: a good bot's ranked (panel-guided) cut choices find safe wires
above the rate a uniformly random legal target would, on average. The no-info baseline
for "a uniformly random facedown card is a wire" is ``active_wires / total_facedown``
(over every seat, not just legal targets) -- the model's own exchangeability
assumption (multivariate-hypergeometric deal, model.md 3.1/3.4.1), independent of any
belief the panel computes.
"""
import numpy as np
import pytest

from tbgame.engine import TableGame
from tbgame.state import AgentView, legal_targets
from tbgame.agents.solver import SolverBot

N_SEATS = 5
N_GAMES = 200


def _stats(values):
  v = np.asarray(values, dtype=float)
  return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def _gap_is_positive(diffs, z=5.0):
  mean, se = _stats(diffs)
  lower = mean - z * se
  return bool(lower > 0.0), float(mean), float(lower)


def _setup(seed, n=N_SEATS):
  return {"seats": [{"name": "P%d" % i, "occupant": "solver_bot"} for i in range(n)],
          "seed": seed}


def _agent_view(game, seat):
  return AgentView(public=game.public_state(), private=game.private_view(seat))


def _play_recording_good_cuts(game, agents):
  """Drive one game to completion, recording every GOOD bot's cut outcome plus the
  no-information baseline hit rate at that same decision point."""
  observed, baseline = [], []
  while not game.finished:
    pending = game.pending
    if pending.kind == "declare":
      for seat in list(pending.seats):
        view = _agent_view(game, seat)
        value = agents[seat].declare(view)
        claim = agents[seat].claim(view)
        game.submit({"seat": seat, "kind": "declare", "value": value, "claim": claim})
      continue
    seat = pending.seats[0]
    view = _agent_view(game, seat)
    pub, priv = view.public, view.private
    is_good = priv.my_role == 0
    if is_good and legal_targets(pub, seat):
      facedown_total = sum(pub.hand_size - pub.revealed[j] for j in range(pub.num_players))
      baseline.append(pub.active_wires / facedown_total)
    target = agents[seat].choose_cut(view)
    claim = agents[seat].claim(view)
    events = game.submit({"seat": seat, "kind": "cut", "value": target, "claim": claim})
    if is_good and baseline and len(baseline) != len(observed):
      cut_event = next(e for e in events if e["type"] == "cut")
      observed.append(1.0 if cut_event["result"] == "wire" else 0.0)
  return observed, baseline


@pytest.mark.slow
def test_good_bot_cuts_beat_random_baseline():
  per_game_gap = []
  for seed in range(N_GAMES):
    game = TableGame(_setup(seed))
    agents = [SolverBot() for _ in range(N_SEATS)]
    observed, baseline = _play_recording_good_cuts(game, agents)
    if not observed:
      continue
    per_game_gap.append(float(np.mean(observed)) - float(np.mean(baseline)))
  ok, mean, lower = _gap_is_positive(per_game_gap)
  assert ok, ("good-bot hit-rate minus baseline mean=%.3f (5-sigma lower %.3f) not > 0"
              % (mean, lower))


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q", "-m", "slow"]))
