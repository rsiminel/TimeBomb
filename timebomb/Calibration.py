# -*- coding: utf-8 -*-
"""Calibration measurement for the Time Bomb belief (model.md §3.5; prerequisite for
ADR 0006's cut panel and ADR 0005's per-round tempering).

A Bayesian posterior under a correctly-specified model is **calibrated**: among the
players the model calls ``P(bad) = p``, a fraction of about ``p`` really are bad (and
likewise for ``P(bomb)``). The panel's risk numbers are only worth trusting if this
holds, so these helpers measure it — a reliability table and the Expected Calibration
Error (ECE) — over simulated games. Because the simulator and the inference share the
uniform-lie model, a correct implementation must come out calibrated up to sampling
noise; a systematic gap is a bug (this is the check that would have caught the
declaration lie-count error of ADR 0007).

``Calibrate`` prints reliability tables for ``P(bad)`` and ``P(bomb)``; the individual
collectors return ``(predictions, outcomes)`` arrays for use in tests.
"""

import numpy as np
from random import Random

import General as gen


def reliability(predictions, outcomes, n_bins=10):
  """Reliability table + Expected Calibration Error for probabilistic predictions.

  Bins the predicted probabilities into ``n_bins`` equal-width intervals over ``[0, 1]``
  and, for each non-empty bin, reports ``(lo, hi, count, mean_prediction,
  empirical_frequency)``. The ECE is the count-weighted mean gap
  ``Σ_bins (count/N)·|mean_prediction − empirical_frequency|`` — 0 for a perfectly
  calibrated forecaster, up to 1 for an adversarial one. Returns ``(rows, ece)``.
  """
  predictions = np.asarray(predictions, dtype=float)
  outcomes = np.asarray(outcomes, dtype=float)
  total = len(predictions)
  edges = np.linspace(0.0, 1.0, n_bins + 1)
  rows = []
  ece = 0.0
  for b in range(n_bins):
    lo, hi = edges[b], edges[b + 1]
    if b < n_bins - 1:
      mask = (predictions >= lo) & (predictions < hi)
    else:
      mask = (predictions >= lo) & (predictions <= hi)  # last bin closed on the right
    count = int(mask.sum())
    if count == 0:
      rows.append((lo, hi, 0, float("nan"), float("nan")))
      continue
    mean_pred = float(predictions[mask].mean())
    emp_freq = float(outcomes[mask].mean())
    rows.append((lo, hi, count, mean_pred, emp_freq))
    ece += (count / total) * abs(mean_pred - emp_freq)
  return rows, ece


def collect_pbad(play_auto, num_players, num_games, seed=0):
  """Run ``play_auto(num_players=..., verbosity=0)`` for ``num_games`` games and pair
  every player's final ``P(bad)`` with whether they were actually bad. Works for any
  variant's ``PlayAuto`` and ``General``'s (all return ``(won, p_bad, roles, ...)``).
  Returns ``(predictions, outcomes)``."""
  import random
  random.seed(seed)
  np.random.seed(seed)
  preds, outs = [], []
  for _ in range(num_games):
    result = play_auto(num_players=num_players, verbosity=0)
    p_bad, roles = result[1], result[2]
    preds.extend(np.asarray(p_bad, dtype=float).tolist())
    outs.extend(np.asarray(roles, dtype=float).tolist())
  return np.array(preds), np.array(outs)


def collect_pbomb(prob_declaration, bomb_marginal, num_players, num_bad, num_rounds,
                  hand_size=5, seed=0):
  """Generate ``num_rounds`` round-starts under the uniform-lie model and pair every
  player's declaration-time ``P(bomb)`` with whether they actually hold it.
  ``prob_declaration(decls, H, A)`` returns the belief and ``bomb_marginal(belief)`` its
  per-player ``P(bomb)`` vector — so it works for any bomb variant or ``General`` at a
  fixed ``num_bad``. Returns ``(predictions, outcomes)``."""
  rng = Random(seed)
  active_wires = num_players
  preds, outs = [], []
  for _ in range(num_rounds):
    bomb = rng.randrange(num_players)
    # slot-uniform deal (multivariate hypergeometric), matching the model (§3.1/§3.4.1)
    slots = [(g, s) for g in range(num_players) for s in range(hand_size - (1 if g == bomb else 0))]
    wires = np.zeros(num_players, dtype=int)
    for (g, _s) in rng.sample(slots, active_wires):
      wires[g] += 1
    bad_set = rng.sample(range(num_players), num_bad)
    decls = wires.astype(float)
    for i in range(num_players):
      if i in bad_set or i == bomb:
        decls[i] = rng.randint(0, hand_size)
    p_bomb = np.asarray(bomb_marginal(prob_declaration(decls, hand_size, active_wires)),
                        dtype=float).reshape(-1)
    for i in range(num_players):
      preds.append(p_bomb[i])
      outs.append(1.0 if i == bomb else 0.0)
  return np.array(preds), np.array(outs)


def print_reliability(title, rows, ece):
  """Print a reliability table (one row per non-empty bin) and the ECE."""
  print(f"\n{title}   (ECE = {ece:.4f})")
  print(f"  {'bin':>11}  {'count':>6}  {'mean pred':>9}  {'emp freq':>9}  {'gap':>7}")
  for lo, hi, count, mean_pred, emp_freq in rows:
    if count == 0:
      continue
    print(f"  [{lo:.1f}, {hi:.1f}){'' :>1} {count:>6}  {mean_pred:>9.3f}  "
          f"{emp_freq:>9.3f}  {abs(mean_pred - emp_freq):>7.3f}")


def Calibrate(num_players=5, num_games=400, num_rounds=4000, seed=0):
  """Print reliability tables for ``P(bad)`` (full games) and ``P(bomb)`` (round-starts)
  at ``num_players``. A quick interactive read on whether the belief is calibrated."""
  pb_pred, pb_out = collect_pbad(gen.PlayAuto, num_players, num_games, seed)
  rows, ece = reliability(pb_pred, pb_out)
  print_reliability(f"P(bad) calibration  (N={num_players}, {num_games} games)", rows, ece)
  nb = gen.NUM_BAD_PRIOR(num_players)
  if len(nb) == 1:  # fixed bad count: P(bomb) reliability is well-defined here
    num_bad = next(iter(nb))
    pbm_pred, pbm_out = collect_pbomb(
        lambda d, h, a: gen.ProbDeclaration(d, h, a, num_bad, 1),
        lambda belief: gen.Separate(belief, num_bad, 1)[1],
        num_players, num_bad, num_rounds, seed=seed)
    rows, ece = reliability(pbm_pred, pbm_out)
    print_reliability(f"P(bomb) calibration  (N={num_players}, {num_rounds} rounds)", rows, ece)


if __name__ == "__main__":
  Calibrate()
