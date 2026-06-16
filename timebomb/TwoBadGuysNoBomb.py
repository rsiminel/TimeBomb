# -*- coding: utf-8 -*-
"""Time Bomb assistant -- variant: 2 bad guys, no bomb.

Exactly two players are bad guys (Terrorists) and there is no Bomb in play. Good
guys declare their true wire count; the bad guys declare uniformly at random. The
belief state is a lower-triangular matrix ``probs[i][j]`` (for i > j) holding
P(players i and j are the bad pair). Because the two bad hands share one wire pool,
the per-hand wire split follows the uniform-placement (multivariate-hypergeometric)
model of docs/model.md §3.4.1 -- the closed forms this module implements.

See docs/model.md for the full model. This module is a stepping stone toward the
general implementation in General.py.

Created on Sun May 26 19:38:34 2024
@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


def PlayAuto(num_players=6, initial_hand_size=5, verbosity=2):
  """Simulate one full game with random play, tracking the belief state.

  Assigns two random bad guys, then each round deals wires, generates declarations
  (good guys truthful, bad guys uniform) and cuts wires at random until the good
  guys find every active wire or time runs out. Returns ``(good_guys_won,
  final_marginals, roles)``: ``final_marginals`` is the per-player P(bad) vector
  (the pair matrix marginalised by ``DeMatrix``); ``roles`` marks the two bad guys.

  ``verbosity``: 0 silent, 1 prints game events, 2 also prints the belief tables.
  """
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  # Distributing roles
  roles = np.zeros(num_players)
  evil = 0
  while evil < 2:
    randy = randint(0, num_players - 1)
    if roles[randy] == 0:
      roles[randy] = 1
      evil += 1
  if verbosity > 0:
    print("Roles : ",  roles)
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    if verbosity > 0:
      print("Round ", initial_hand_size - hand_size + 1)
    # Distribute wires
    wires = uf.DistributeWires(num_players, hand_size, active_wires)
    if verbosity > 0:
      print("w : ", wires)
    # Declare your wires
    declarations = wires.copy()
    for i in range(num_players):
      if roles[i] == 1:
        declarations[i] = randint(0, min(hand_size, active_wires))
    if verbosity > 0:
      print("d : ", declarations)
    # Calculate and display probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    probabilities_list.append(probabilities.copy())
    total_probs = CombineProbs(probabilities_list)
    p_wire = P_wire(declarations, total_probs, np.zeros(num_players), np.zeros(num_players), hand_size, active_wires)
    if verbosity > 1:
      print("  p : ", DeMatrix(probabilities), H(DeMatrix(probabilities)), H2(probabilities))
      print(" tp : ", DeMatrix(total_probs), H(DeMatrix(probabilities)), H2(probabilities))
      print(" pw : ", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size))
      print(" em :", H_Min(declarations, total_probs, np.zeros(num_players),
                          np.zeros(num_players), hand_size, active_wires, 3))
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    for i in range(num_players):
      if verbosity > 0:
        print("Cut number", i + 1)
      cutee = randint(0, num_players - 1)
      while revealed[cutee] >= hand_size:
        cutee = randint(0, num_players - 1)
      randy = randint(1, hand_size - revealed[cutee])
      if randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r : ", revealed)
        print("f : ", found)
      # Update and display probabilities
      new_probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      probabilities_list[-1] = new_probs.copy()
      total_probs = CombineProbs(probabilities_list)
      p_wire = P_wire(declarations, total_probs, revealed, found, hand_size, active_wires)
      if verbosity > 1:
        print("  p :", DeMatrix(new_probs), H(DeMatrix(new_probs)), H2(new_probs))
        print(" tp :", DeMatrix(total_probs), H(DeMatrix(total_probs)), H2(total_probs))
        print(" pw : ", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size - np.sum(revealed)))
        print(" em :", H_Min(declarations, total_probs, revealed, found, hand_size, active_wires, min(3, num_players-i-1)))
      # Test for victory
      if active_wires <= 0:
        if verbosity > 0:
          print("Good guys win!")
        return (1, DeMatrix(CombineProbs(probabilities_list)), roles)
    # Next round
    hand_size -= 1
    if verbosity > 0:
      print("\n")
  if verbosity > 0:
    print("Bad guys win!")
  return (0, DeMatrix(CombineProbs(probabilities_list)), roles)


def DeMatrix(probabilities):
  """Marginalise the pair matrix to a per-player P(bad) vector.

  ``P(player i is bad) = Σ_j probabilities[i][j]`` over every pair containing i
  (summing both the row and the column of the lower-triangular matrix). Returns a
  length-N vector; its entries sum to 2 (each pair contributes to two players).
  """
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i):
      probability_line[i] += probabilities[i][j]
      probability_line[j] += probabilities[i][j]
  return probability_line


def CombineProbs(probabilities_list):
  """Combine each round's independent pair distribution into one.

  Treats the rounds as independent evidence: multiply the per-round pair matrices
  element-wise and renormalise. A pair ruled out in any round (0) stays ruled out.
  Returns the unnormalised (all-zero) matrix if every pair has been ruled out.
  """
  num_tests = len(probabilities_list)
  num_players = probabilities_list[0].shape[0]
  probabilities = np.full([num_players, num_players], 1.)
  for i in range(num_players):
    for j in range(num_players):
      for k in range(num_tests):
        probabilities[i][j] *= probabilities_list[k][i][j]
  if np.sum(probabilities) == 0:
    return probabilities
  probabilities /= np.sum(probabilities)
  return probabilities


def ProbDeclaration(decls, hand_size, active_wires):
  """Prior P(players i and j are the bad pair) from the declarations alone.

  The uniform-lie joint-Bayes prior for B=2 (docs/model.md §3.3). Marking {i, j} as
  the bad pair pins every other (truthful) hand to its declared count and forces the
  pair's true wire total to ``t_ij = decls[i] + decls[j] - excess``, where
  ``excess = sum(decls) - active_wires``. Under uniform placement the prior is
  proportional to the multivariate-hypergeometric weight of that forced deal; summing
  over the pair's internal wire splits collapses (Vandermonde) to a closed form:

    P({i, j}) ∝ C(2H, t_ij) / ( C(H, decls[i]) · C(H, decls[j]) )

  with ``C(2H, t) = 0`` for ``t < 0`` or ``t > 2H``. Unlike B=1, ``excess = 0`` does
  *not* give a uniform prior, so there is no special case. If no pair can account for
  the excess (every weight 0) the declarations are impossible under the model and the
  prior falls back to uniform over pairs (§3.3 / TODO Axis A3), never all-zeros.

  Returns a lower-triangular matrix (``probs[i][j]`` for i > j) summing to 1.
  """
  num_players = len(decls)
  excess = np.sum(decls) - active_wires
  weights = np.zeros([num_players, num_players])
  for i in range(num_players):
    for j in range(i):
      t_ij = decls[i] + decls[j] - excess  # the bad pair's true wire total
      if 0 <= t_ij <= 2 * hand_size:
        denom = uf.C(decls[i], hand_size) * uf.C(decls[j], hand_size)
        if denom != 0:
          weights[i][j] = uf.C(t_ij, 2 * hand_size) / denom
  total = np.sum(weights)
  if total == 0:  # Declarations impossible under the model: uniform over pairs
    for i in range(num_players):
      for j in range(i):
        weights[i][j] = 1.
    total = np.sum(weights)
  return weights / total


def L_bad_pair(hand_size, bg_wires, revealed, found, i, j):
  """Likelihood of the cut observation in bad hands i and j, given they are the bad
  pair holding ``bg_wires`` wires between them (docs/model.md §3.4.1).

  Under uniform placement the ``bg_wires`` wires fill the ``2H`` combined bad
  card-slots; the revealed cards are an exchangeable subset, so the per-hand split
  sum collapses to one multivariate-hypergeometric term:

    L = C(rev_i, f_i) · C(rev_j, f_j) · C(2H − rev_i − rev_j, bg − f_i − f_j) / C(2H, bg)

  Reduces exactly to the single-hand ``Lklhd`` when one hand is empty. Returns 0 for
  an infeasible configuration (wire total or hidden-wire count out of range).
  """
  hidden_slots = 2 * hand_size - revealed[i] - revealed[j]
  hidden_wires = bg_wires - found[i] - found[j]
  if not (0 <= bg_wires <= 2 * hand_size) or not (0 <= hidden_wires <= hidden_slots):
    return 0.0
  denom = uf.C(bg_wires, 2 * hand_size)  # C(2H, bg_wires)
  if denom == 0:
    return 0.0
  return (uf.C(found[i], revealed[i]) * uf.C(found[j], revealed[j])
          * uf.C(hidden_wires, hidden_slots)) / denom


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  """Bayesian posterior over which pair is bad after a cut is revealed.

  The hypotheses are the C(N, 2) bad pairs, mutually exclusive and exhaustive.
  Conditioned on pair {bad1, bad2}, every good hand is pinned to its declared count
  and the bad pair holds ``bg_wires = active_wires + Σfound − Σ_good decls`` between
  them, so the observation factorises into the §3.4.1 bad-pair likelihood times the
  good guys' independent per-hand hypergeometrics:

    L({bad1, bad2}) = L_bad_pair(...) · Π_{good} Lklhd(H, decls[good], rev, found)
    posterior(c) = prior(c) · L(c) / Σ_c′ prior(c′) · L(c′)

  Returns the prior unchanged on a zero marginal (observation impossible under every
  pair) or once a pair is certain. Inputs/outputs mirror ProbDeclaration's matrix.
  """
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players])
  marginal = 0
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      if prior[bad1][bad2] == 1:  # The two bad guys have already been found
        return prior
      good_decls = np.sum(decls) - decls[bad1] - decls[bad2]
      bg_wires = active_wires + np.sum(found) - good_decls
      lklhd[bad1][bad2] = L_bad_pair(hand_size, bg_wires, revealed, found, bad1, bad2)
      for good in range(num_players):  # Likelihood of everyone else's configurations
        if good != bad1 and good != bad2:
          lklhd[bad1][bad2] *= uf.Lklhd(hand_size, decls[good], revealed[good], found[good])
      marginal += prior[bad1][bad2] * lklhd[bad1][bad2]
  if marginal == 0:
    return prior
  posterior = prior.copy()
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      posterior[bad1][bad2] *= lklhd[bad1][bad2] / marginal
  return posterior


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  """Probability that cutting one of player i's face-down cards reveals a wire.

  Mixes the belief that i is good vs. bad over the pair distribution (docs/model.md
  §3.5). For a pair {i, j} the hidden wires ``bg − f_i − f_j`` are uniform over the
  ``2H − rev_i − rev_j`` combined hidden slots (§3.4.1), so each bad hand's per-card
  wire probability is exactly ``(bg − f_i − f_j) / (2H − rev_i − rev_j)``. The
  good-guy branch contributes ``(decls[i] − found[i]) / (H − rev_i)`` and, like
  OneBadGuyNoBomb, is gated so an infeasible "good" hand adds nothing rather than a
  negative count. Players with no face-down cards score 0.
  """
  num_players = decls.size
  p_wire = np.zeros(num_players)
  # Bad-pair branch: pooled-marginal wire probability under uniform placement.
  for i in range(num_players):
    for j in range(i):
      good_decls = np.sum(decls) - decls[i] - decls[j]
      bg_wires = active_wires + np.sum(found) - good_decls
      hidden_slots = 2 * hand_size - revealed[i] - revealed[j]
      hidden_wires = bg_wires - found[i] - found[j]
      if hidden_slots <= 0 or not (0 <= hidden_wires <= hidden_slots):
        continue
      per_card = hidden_wires / hidden_slots  # same for both bad hands
      if hand_size - revealed[i] > 0:
        p_wire[i] += probs[i][j] * per_card
      if hand_size - revealed[j] > 0:
        p_wire[j] += probs[i][j] * per_card
  # Good-guy branch: i holds exactly decls[i], gated for feasibility.
  lin_probs = DeMatrix(probs)
  for i in range(num_players):
    remaining_cards = hand_size - revealed[i]
    if remaining_cards <= 0:
      continue
    good_wires = decls[i] - found[i]
    if 0 <= good_wires <= remaining_cards:
      p_wire[i] += (1 - lin_probs[i]) * good_wires / remaining_cards
  return p_wire


def H(probs):
  """Shannon entropy (bits) of a 1-D vector -- e.g. the per-player marginals from
  ``DeMatrix``. 0 for a certain outcome; used to score information gain."""
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h


def H2(probs):
  """Shannon entropy (bits) of the 2-D pair matrix: sums ``p·log2(1/p)`` over every
  pair entry. The belief-state entropy minimised by the lookahead strategy."""
  h = 0
  for line in probs:
    for p in line:
      if p > 0.0001:
        h += p * np.log2(1/p)
  return h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop):
  """Recursive min-entropy lookahead: search up to ``stop`` cuts ahead for the cut
  sequence minimising the expected final entropy (``H2``) of the pair belief state.

  Returns ``(expected_entropy, path)`` where ``path`` is the recommended cut order
  (player indices, last to first). An information-greedy cut strategy.
  """
  if stop <= 0:
    return (H2(probs), [])
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  h = np.zeros(num_players)
  h_wire = 0
  h_not_wire = 0
  path = []
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players)
    reveal[cutee] += 1
    find = np.zeros(num_players)
    find[cutee] += 1
    if p_wire[cutee] > 0.0001:
      new_probs = ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires)
      (h_wire, path) = H_Min(decls, new_probs, revealed + reveal, found + find, hand_size, active_wires, stop - 1)
    if p_wire[cutee] < 0.9999:
      new_probs = ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires)
      (h_not_wire, path) = H_Min(decls, new_probs, revealed + reveal, found, hand_size, active_wires, stop - 1)
    h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_not_wire
  min_cutee = 0
  min_h = h[0]
  for cutee in range(num_players):
    if path != []:
      if cutee == path[-1]:
        continue
    if h[cutee] < min_h:
      min_cutee = cutee
      min_h = h[cutee]
  return (min_h, path + [min_cutee])

PlayAuto()

