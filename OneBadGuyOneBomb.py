# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:24:34 2024

@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


# 1 bad guy, 1 bomb
# Suppositions : good guys tell the truth unless they have the bomb
#                if a good guy has the bomb, he declares fewer wires than he has
#                bad guys lie randomly unless they have the bomb
#                if a bad guy has the bomb, he declares more wires than he has
def PlayAuto(num_players=4, initial_hand_size=5, verbosity=2):
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  # Distributing roles
  roles = np.zeros(num_players)
  roles[randint(0, num_players - 1)] = 1
  if verbosity > 0:
    print("Roles : ",  roles)
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    if verbosity > 0:
      print("Round ", initial_hand_size - hand_size + 1)
    # Distribute wires
    wires = uf.DistributeWires(num_players, active_wires, hand_size)
    if verbosity > 0:
      print("w:", wires)
    bomb = np.zeros(num_players)
    randy = randint(0, num_players - 1)
    while wires[randy] >= hand_size:
      randy = randint(0, num_players - 1)
    bomb[randy] = 1
    if verbosity > 0:
      print("b:", bomb)
    # Declare your wires
    declarations = wires.copy()
    for i in range(num_players):
      maxx = min(hand_size, active_wires)
      if roles[i] == 1:
        if bomb[i] == 1:
          declarations[i] = randint(wires[i], maxx)
        else:
          declarations[i] = randint(0, maxx)
      elif bomb[i] == 1:
        declarations[i] = randint(0, wires[i])
    if verbosity > 0:
      print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, active_wires, hand_size)
    prob_bad, prob_bomb = DeMatrix(probabilities)
    probabilities_list.append(prob_bad.copy())
    comb_probs = CombineProbs(probabilities_list)
    if verbosity > 1:
      print("bp:", prob_bomb, np.sum(prob_bomb))
      print(" p:", prob_bad, np.sum(prob_bad))
      print("tp:", comb_probs, np.sum(comb_probs))
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    for i in range(num_players):
      if verbosity > 0:
        print("Cut number", i + 1)
      cutee = randint(0, num_players - 1)  # cutee = max(prob, dclrtn)
      while revealed[cutee] >= hand_size:
        cutee = randint(0, num_players - 1)
      randy = randint(1, hand_size - revealed[cutee])
      if bomb[cutee] == 1 and randy == hand_size - revealed[cutee]:
        if verbosity > 0:
          print("The Bomb was detonated. Bad guys win!")
        return (0, CombineProbs(probabilities_list), roles)
      elif randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      prob_bad, prob_bomb = DeMatrix(probs)
      probabilities_list[-1] = prob_bad.copy()
      comb_probs = CombineProbs(probabilities_list)
      if verbosity > 1:
        print("bp:", prob_bomb, np.sum(prob_bomb))
        print(" p:", prob_bad, np.sum(prob_bad))
        print("tp:", comb_probs, np.sum(comb_probs))
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


def DeMatrix(probabilities):
  num_players = len(probabilities)
  probability_bad = np.zeros(num_players)
  probability_bomb = np.zeros(num_players)
  for i in range(num_players):
    for j in range(num_players):
      probability_bad[i] += probabilities[i][j]
      probability_bomb[j] += probabilities[i][j]
  return (probability_bad, probability_bomb)


def ProbDeclaration(decls, active_wires, hand_size):
  num_players = decls.shape[0]
  probs = np.full([num_players, num_players], 1 / num_players**2)
  for i in range(num_players):
    bg_wires = int(active_wires - np.sum(decls) + decls[i])
    for j in range(num_players):
      if i == j:  # The bad guy has the bomb
        if bg_wires > decls[i]:  # i has more wires than declared
          probs[i][j] = 0  # i must have =fewer wires than declared (bg w. bomb)
        else:  # i has =fewer wires than he declared
          probs[i][j] = uf.C(bg_wires, decls[i])
      else:  # A good guy has the bomb
        if hand_size - decls[j] - 1 < 0:  # The bomb is not hidden in j's hand
          probs[i][j] = 0
        combinations = 0
        for i_wires in range(bg_wires + 1):
          j_wires = bg_wires - i_wires + decls[j]
          if i_wires < decls[i]:  # i has fewer wires than declared
            probs_i = uf.C(i_wires, decls[i])
          else:  # i has =more wires than declared
            probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
          # j must have =more wires than declared (gg w. bomb)
          probs_j = uf.C(j_wires - decls[j], hand_size - decls[j] - 1)
          combinations += uf.C(i_wires, bg_wires)
          probs[i][j] += uf.C(i_wires, bg_wires) * probs_i * probs_j
        if combinations != 0:
          probs[i][j] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players])
  for i in range(num_players):
    bg_wires = int(active_wires + np.sum(found) - np.sum(decls) + decls[i])
    for j in range(num_players):
      if prior[i][j] == 1:  # The bad guy and the bomb have been found
        return prior
      if i == j:  # A bad guy has the bomb
        if revealed[i] >= hand_size:  # All of i's hand is not bomb
          lklhd[i][j] = 0
          continue
        if bg_wires > decls[i]:  # j has more wires than declared
          lklhd[i][j] = 0  # i must have =fewer wires than declared (bg w. bomb)
          continue
        # Likelihood of configuration if i is the bad guy with the bomb
        lklhd[i][j] = uf.Lklhd(hand_size - 1, bg_wires, revealed[i], found[i])
      else:  # A good guy has the bomb
        if revealed[j] >= hand_size:  # All of j's hand is not bomb
          lklhd[i][j] = 0
          continue
        # Likelihood of configuration if i and j are lying
        combinations = 0
        for i_wires in range(bg_wires + 1):  # Consider all distributions
          j_wires = bg_wires - i_wires + decls[j]
          lklhd_i = uf.Lklhd(hand_size, i_wires, revealed[i], found[i])
          lklhd_j = uf.Lklhd(hand_size - 1, j_wires, revealed[j], found[j])
          combinations += uf.C(i_wires, bg_wires)
          lklhd[i][j] += uf.C(i_wires, bg_wires) * lklhd_i * lklhd_j
        if combinations != 0:
          lklhd[i][j] /= combinations
  marginal = 0
  for i in range(num_players):
    for j in range(num_players):
      marginal += prior[i][j] * lklhd[i][j]
  posterior = prior.copy()
  for i in range(num_players):
    for j in range(num_players):
      posterior[i][j] *= lklhd[i][j] / marginal
  return posterior


# Arithmetic average preserves 0s and 1s. Don't ask any other questions.
def CombineProbs(probabilities_list):
  num_tests = len(probabilities_list)
  num_players = len(probabilities_list[0])
  probabilities = np.full([num_players], 1.)
  for i in range(num_players):
    for j in range(num_tests):
      probabilities[i] *= probabilities_list[j][i]
  probabilities /= sum(probabilities)
  return probabilities
