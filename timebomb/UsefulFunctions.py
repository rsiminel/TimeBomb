# -*- coding: utf-8 -*-
"""Small numeric helpers shared by every Time Bomb variant.

The combinatorial atoms (`C`, `A`, `Lklhd`) are the building blocks of the
hypergeometric likelihoods in docs/model.md §3.2. They take plain Python integers and
return plain numbers, so they compose cleanly inside the variants' belief functions.

Note on argument order: `C(k, n)` is "n choose k" -- the *lower* index comes first.
This matches how the variants call it, e.g. `C(found, revealed)` for C(revealed,
found). Out-of-range inputs return 0 rather than raising, so callers can sum over
infeasible splits without guarding every term.

Created on Sun Jun  5 14:16:22 2022
@author: Remy
"""

# Imports
import numpy as np
from random import sample


def Fact(x):
  """Integer factorial ``x!`` for ``x >= 0``. Returns ``-1`` as an error sentinel on
  negative input (a guard for the combinatorial helpers below, which never expect it).
  """
  factorial = 1
  if x < 0:
    factorial = -1
  for i in range(1, int(x) + 1):
    factorial *= i
  return factorial


def C(k, n):
  """Binomial coefficient ``C(n, k)`` -- the number of ways to choose ``k`` of ``n``
  items. Note the reversed argument order (lower index ``k`` first). Returns 0 when
  ``k > n`` (or ``k < 0``), so infeasible terms vanish from a sum."""
  if n < k or k < 0:
    return 0
  return Fact(n) / (Fact(k) * Fact(n - k))


def A(k, n):
  """Number of ordered arrangements ``n! / (n−k)!`` -- ``k``-permutations of ``n``
  items. Returns 0 when ``k > n``. Used to write `Lklhd` in permutation form."""
  if n < k or k < 0:
    return 0
  return Fact(n) / Fact(n - k)


def Lklhd(n, m, k, p):
  """Hypergeometric likelihood (docs/model.md §3.2): the probability that cutting ``k``
  of an ``n``-card hand holding ``m`` wires reveals exactly ``p`` of them. Written in
  permutation form, equal to ``C(m, p) · C(n−m, k−p) / C(n, k)``."""
  return C(p, k) * A(p, m) * A(k - p, n - m) / A(k, n)


def DistributeWires(num_players, hand_size, active_wires):
  """Deal ``active_wires`` wires uniformly at random among the ``num_players * hand_size``
  card **slots** -- the multivariate-hypergeometric deal the inference assumes (model.md
  §3.1/§3.4.1, every arrangement of which cards are wires equally likely). Returns the
  integer wire-count vector. Simulation-only helper for the ``PlayAuto`` generators.

  This must be slot-uniform, not player-uniform: a "pick a random player, add a wire if
  under capacity" loop draws from a different distribution (e.g. for N=2, H=2, A=2 it gives
  counts ``(1/4, 1/2, 1/4)`` instead of the hypergeometric ``(1/6, 2/3, 1/6)``), which makes
  the simulated games diverge from the model and shows up as ``P(bad)`` miscalibration."""
  slots = [(g, s) for g in range(num_players) for s in range(hand_size)]
  wires = np.zeros(num_players, dtype=int)
  for (g, _s) in sample(slots, int(active_wires)):
    wires[g] += 1
  return wires
