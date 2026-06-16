# -*- coding: utf-8 -*-
"""Time Bomb assistant -- variant: 1 bad guy, no bomb.

The simplest configuration: exactly one player is the bad guy (Terrorist) and
there is no Bomb in play. Good guys always declare their true wire count; the bad
guy declares a uniformly random count. The belief state is a length-N vector
``probs[i] = P(player i is the bad guy)``.

See docs/model.md for the full model. This module is a stepping stone toward the
general implementation in General.py.

Created on Sun Jun  5 14:21:49 2022
@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


def PlayAuto(num_players=4, initial_hand_size=5, verbosity=2):
  """Simulate one full game with random play, tracking the belief state.

  Assigns one random bad guy, then each round deals wires, generates
  declarations (good guys truthful, bad guy random) and cuts wires at random
  until the good guys find every active wire or time runs out. Returns a tuple
  ``(good_guys_won, final_probs, roles)``: ``good_guys_won`` is 1/0,
  ``final_probs`` is the combined P(bad) vector, ``roles`` marks the bad guy.

  ``verbosity``: 0 silent, 1 prints game events, 2 also prints the belief tables.
  """
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  # Distributing roles
  roles = np.zeros(num_players)
  roles[randint(0, num_players - 1)] = 1
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    if verbosity > 0:
      print("Round ", initial_hand_size - hand_size + 1)
    wires = uf.DistributeWires(num_players, hand_size, active_wires)
    if verbosity > 0:
      print("w:", wires)
    # Declare your wires
    declarations = wires.copy()
    for i in range(num_players):
      if roles[i] == 1:
        declarations[i] = randint(0, min(hand_size, active_wires))
    if verbosity > 0:
      print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    probabilities_list.append(probabilities.copy())
    total_probs = CombineProbs(probabilities_list)
    p_wire = P_wire(declarations, total_probs, np.zeros(num_players),
                    np.zeros(num_players), hand_size, active_wires)
    if verbosity > 1:
      print("  p:", probabilities, H(probabilities))
      print(" tp:", total_probs, H(total_probs))
      print(" pw:", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size))
      print(" em:", H_Min(declarations, total_probs, np.zeros(num_players),
                          np.zeros(num_players), hand_size, active_wires, num_players))
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    for i in range(num_players):
      if verbosity > 0:
        print("Cut number", i + 1)
      cutee = randint(0, num_players - 1)  # cutee = max(prob, dclrtn)
      while revealed[cutee] >= hand_size:
        cutee = randint(0, num_players - 1)
      randy = randint(1, hand_size - revealed[cutee])
      if randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      probabilities_list[-1] = probs.copy()
      total_probs = CombineProbs(probabilities_list)
      p_wire = P_wire(declarations, total_probs, revealed, found, hand_size, active_wires)
      if verbosity > 1:
        print("  p:", probs, H(probs))
        print(" tp:", total_probs, H(total_probs))
        print(" pw : ", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size - np.sum(revealed)))
        print(" em:", H_Min(declarations, total_probs, revealed, found, hand_size,
                            active_wires, num_players - i - 1))
      # Test for victory
      if active_wires <= 0:
        if verbosity > 0:
          print("Good guys win!")
        return (1, CombineProbs(probabilities_list), roles)
    # Next round
    hand_size -= 1
    if verbosity > 0:
      print("\n")
  if verbosity > 0:
    print("Bad guys win!")
  return (0, CombineProbs(probabilities_list), roles)


def Play(players=["Alice", "Bob", "Clara", "Darryl"], initial_hand_size=5):
  """Interactive assistant for a real game: prompts for declarations and cut
  results at the terminal and prints the updated belief state after each event.
  """
  num_players = len(players)
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    print("Round ", initial_hand_size - hand_size + 1)
    # Declare your wires
    declarations = np.zeros(num_players)
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    probabilities_list.append(probabilities.copy())
    print(" p:", probabilities)
    print("tp:", CombineProbs(probabilities_list))
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    for i in range(num_players):
      print("Cut number", i + 1)
      cutee_str = input("Who's wire has been cut? ")
      while cutee_str not in players:
        cutee_str = input("You must have made a typo. Who? ")
      cutee = 0
      for j in range(num_players):
        if players[j] == cutee_str:
          cutee = j
      revealed[cutee] += 1
      num_wires -= 1
      shown = int(input("Did you reveal an\n" + " 1- inactive wire\n 2- active wire\n"))
      while shown not in [1, 2]:
        shown = int(input("Sorry, I'm looking for a 1 or a 2 here."))
      if shown == 2:
        found[cutee] += 1
        active_wires -= 1
      print("r:", revealed)
      print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      probabilities_list[-1] = probs.copy()
      print(" p:", probs)
      print("tp:", CombineProbs(probabilities_list))
      # Test for victory
      if active_wires <= 0:
        print("Good guys win!")
        return
    # Next round
    hand_size -= 1
    print("\n")
  print("Bad guys win!")
  return


def CombineProbs(probabilities_list):
  """Combine each round's independent P(bad) estimate into one distribution.

  Treats the rounds as independent evidence: multiply the per-round probabilities
  element-wise and renormalise. A player ruled out in any round (0) stays ruled
  out; a player pinned in some round (1) dominates the product.
  """
  num_tests = len(probabilities_list)
  num_players = len(probabilities_list[0])
  probabilities = np.full(num_players, 1.)
  for i in range(num_players):
    for j in range(num_tests):
      probabilities[i] *= probabilities_list[j][i]
  probabilities /= sum(probabilities)
  return probabilities


def ProbDeclaration(declarations, hand_size, active_wires):
  """Prior P(player i is the bad guy) from the round's declarations alone.

  Let ``excess = sum(declarations) - active_wires`` be the total over-declaration.
  With no excess the declarations are consistent with everyone telling the truth,
  so no information is available and the prior is uniform. Otherwise the lone liar
  must account for the excess, and each player is weighted by the number of
  card-arrangements consistent with their being that liar (C(n, k) = n choose k):

    excess > 0 (liar padded their count): weight C(declarations[i], excess)
    excess < 0 (liar hid wires):          weight C(hand_size - declarations[i], -excess)

  Returns a probability vector summing to 1, or all-zeros if the declarations are
  impossible under the model (no single liar can account for the excess).
  """
  num_players = declarations.size
  probabilities = np.full(num_players, 1 / num_players)
  excess = - active_wires + sum(declarations)
  if excess != 0:  # With no excess, no information can be extracted
    for i in range(num_players):
      if excess > 0:  # The extra decls are in i's declarations
        probabilities[i] = uf.C(excess, declarations[i])
      else:  # The missing decls are in what i did not declare
        probabilities[i] = uf.C(- excess, hand_size - declarations[i])
    if np.sum(probabilities) != 0:
      probabilities /= np.sum(probabilities)
  return probabilities


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  """Bayesian posterior over "who is the bad guy" after a cut is revealed.

  The hypotheses are mutually exclusive and exhaustive: exactly one player is
  the bad guy. Conditioned on hypothesis "player `bad` is the bad guy", every
  hand has a known wire count (the bad guy holds `bad_wires`; every good guy
  holds exactly their declared count, since good guys tell the truth), so the
  observation factorises into independent per-hand hypergeometric draws:

      L(bad) = P(obs | player `bad` is bad)
             = Lklhd(H, bad_wires, revealed[bad], found[bad])
               * prod over good != bad of Lklhd(H, decls[good], revealed[good], found[good])

  The posterior is then Bayes' theorem on that configuration space:

      posterior[bad] = prior[bad] * L(bad) / sum_k prior[k] * L(k)

  Inputs
    decls         declared wire count per player
    prior         current P(player i is the bad guy), sums to 1
    revealed      cards already cut from each player's hand this round
    found         active wires found in each player's hand this round
    hand_size     cards per hand this round (H)
    active_wires  safe wires still face-down across all hands (A)
  Returns
    posterior probability vector, summing to 1.
  """
  num_players = decls.size
  likelihood = np.zeros(num_players)
  marginal = 0
  for bad in range(num_players):
    if prior[bad] == 1:  # The bad guy has already been found; nothing to update
      return prior
    # The bad guy holds whatever wires the truthful good guys do not account for
    bad_wires = active_wires + np.sum(found) - np.sum(decls) + decls[bad]
    likelihood[bad] = uf.Lklhd(hand_size, bad_wires, revealed[bad], found[bad])
    for good in range(num_players):  # Truthful good guys hold their declared count
      if good != bad:
        likelihood[bad] *= uf.Lklhd(hand_size, decls[good], revealed[good], found[good])
    marginal += prior[bad] * likelihood[bad]
  if marginal == 0:  # Observation impossible under every hypothesis: keep prior
    return prior
  posterior = prior.copy()
  for i in range(num_players):
    posterior[i] *= likelihood[i] / marginal
  return posterior


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  """Probability that cutting one of player i's face-down cards reveals a wire.

  Player i has ``hand_size - revealed[i]`` cards still face-down. The active wires
  hidden among them depend on whether i is the bad guy:

    - good (prob 1 - probs[i]): holds their declared count, so decls[i] - found[i]
      wires remain;
    - bad  (prob probs[i]):     holds the wires the truthful good guys do not
      account for, so ``i_wires`` remain.

  The result mixes the two and divides by the number of remaining face-down cards
  (not the full hand -- a card already revealed can no longer be cut). Each branch
  contributes only when its implied wire count is feasible (between 0 and the
  remaining cards): an impossible hypothesis contributes nothing. Players with no
  cards left score 0.
  """
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for i in range(num_players):
    remaining_cards = hand_size - revealed[i]
    if remaining_cards <= 0:  # No face-down cards left to cut
      continue
    # Bad-guy branch: i holds the wires the truthful good guys do not account for
    bad_wires = active_wires + np.sum(found) - np.sum(decls) + decls[i] - found[i]
    if 0 <= bad_wires <= remaining_cards:
      p_wire[i] += probs[i] * bad_wires
    # Good-guy branch: i holds exactly their declared count
    good_wires = decls[i] - found[i]
    if 0 <= good_wires <= remaining_cards:
      p_wire[i] += (1 - probs[i]) * good_wires
    p_wire[i] /= remaining_cards
  return p_wire


def H(probs):
  """Shannon entropy (bits) of a probability vector: 0 for a certain outcome,
  log2(N) for the uniform distribution. Used to score information gain."""
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h


def NextH(decls, probs, revealed, found, hand_size, active_wires):
  """Expected posterior entropy after cutting each player's wire (one step ahead).

  For every cuttable player, average the entropy of the updated belief over the
  two possible outcomes (a wire with probability P_wire, nothing otherwise).
  Lower means that cut is expected to be more informative.
  """
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  h = np.zeros(num_players)
  info_wire = 0
  info_not_wire = 0
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players)
    reveal[cutee] += 1
    find = np.zeros(num_players)
    find[cutee] += 1
    if p_wire[cutee] > 0.0001:
      info_wire = H(ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires))
    if p_wire[cutee] < 0.9999:
      info_not_wire = H(ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires))
    h[cutee] = p_wire[cutee] * info_wire + (1 - p_wire[cutee]) * info_not_wire
  return h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop):
  """Recursive min-entropy lookahead: search up to ``stop`` cuts ahead for the
  sequence of cuts that minimises the expected final entropy of the belief state.

  Returns ``(expected_entropy, path)`` where ``path`` is the recommended cut order
  (player indices, last to first). An information-greedy cut strategy.
  """
  if stop <= 0:
    return (H(probs), [])
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


def ProbSus(players, probs):
  """Update beliefs from a subjective read on one player's behaviour.

  Prompts for a suspected player and the likelihood that they would behave as
  observed if they were good vs. bad, then applies Bayes' rule: the suspect's
  P(bad) is updated by the likelihood ratio and the remaining mass is rescaled
  proportionally across the other players. Modifies ``probs`` in place.
  """
  num_players = len(players)
  sus_str = input("Who is sus? ")
  while sus_str not in players:
    sus_str = input("You must have made a typo. Who? ")
  sus = 0
  for player in range(num_players):
    if players[player] == sus_str:
      sus = player
  lklhd_good = int(input("What is the likelihood that they would do this as a good guy?"))
  lklhd_bad = int(input("What is the likelihood that they would do this as a bad guy?"))
  new_prob = probs[sus] * lklhd_bad / (probs[sus] * lklhd_bad + (1 - probs[sus]) * lklhd_good)
  for i in range(num_players):
    if i != sus:
      probs[i] *= 1 - (new_prob - probs[sus]) / (1 - probs[sus])
  probs[sus] = new_prob
  return
