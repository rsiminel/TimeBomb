# -*- coding: utf-8 -*-
"""Self-calibrating "beats the no-information baseline" checks for the end-to-end tests.

The end-to-end accuracy tests assert the belief is *informative* — a true bad guy's
P(bad) sits above the no-information baseline B/N, and clearly above a true good guy's.
Earlier these used hand-tuned cutoffs (``P(bad|bad) > 0.6`` etc.), which drift with the
model and read as arbitrary. Instead, derive the bar from the run itself: a one-sided
z-test of a per-game statistic against the baseline, so the threshold scales with the
sample's size and spread rather than a guess.

Aggregate **per game**, not per (game, player): the games are independent, so per-game
values are i.i.d. and the normal approximation is honest, whereas players within one game
are correlated (a shared deal / shared lies). ``z`` defaults to 5 — a deliberately wide
margin, since a genuine regression collapses the signal toward zero, far inside any sane
z, so the test still fails hard while never flaking on sampling noise.

Used by every variant's ``test_*_beats_random_chance`` and the General joint test.
"""
import numpy as np


def _stats(values):
  v = np.asarray(values, dtype=float)
  return v.mean(), v.std(ddof=1) / np.sqrt(len(v))  # mean, standard error of the mean


def exceeds_baseline(per_game, baseline, z=5.0):
  """``(ok, mean, lower)``: is the per-game sample mean more than ``z`` standard errors
  *above* ``baseline``? ``lower = mean - z*se`` is the lower end of the one-sided
  confidence bar. Use for "true bad guys (or the true bomb holder) score above the
  no-information rate" and for a top-k accuracy vs its random-guess rate."""
  mean, se = _stats(per_game)
  lower = mean - z * se
  return bool(lower > baseline), float(mean), float(lower)


def gap_is_positive(per_game_diffs, z=5.0):
  """``(ok, mean, lower)``: are the paired per-game differences more than ``z`` standard
  errors *above* zero? Feed per-game ``P(bad|bad) - P(bad|good)`` — the paired form
  cancels the shared per-game deal/lie noise, giving the honest informativeness gap. This
  is the robust cross-variant claim: unlike "P(bad|good) < baseline", it holds even for
  the bomb variants, where a good guy forced to lie by the bomb pushes P(bad|good) up to
  about the baseline."""
  mean, se = _stats(per_game_diffs)
  lower = mean - z * se
  return bool(lower > 0.0), float(mean), float(lower)
