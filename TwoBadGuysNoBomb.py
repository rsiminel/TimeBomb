# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:38:34 2024

@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


# 2 bad guys, no bomb
# Suppositions : good guys tell the truth, bad guys lie randomly
def PlayAuto(num_players=6, initial_hand_size=5, verbosity=2):
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
        declarations[i] = randint(0, hand_size + 1)
    if verbosity > 0:
      print("d : ", declarations)
    # Calculate and display probabilities
    probabilities = ProbDeclaration(declarations, active_wires, hand_size)
    probabilities_list.append(probabilities.copy())
    prob_line = DeMatrix(CombineProbs(probabilities_list))
    if verbosity > 1:
      print(" p : ", DeMatrix(probabilities))
      print("tp : ", prob_line, np.sum(prob_line))
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
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
        print("r:", revealed)
        print("f:", found)
      # Update and display probabilities
      new_probs = ProbCut(declarations, probabilities, revealed, found, num_players, active_wires)
      probabilities_list[-1] = new_probs.copy()
      if verbosity > 1:
        print(" p:", DeMatrix(new_probs))
        print("tp:", DeMatrix(CombineProbs(probabilities_list)))
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
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i + 1):
      probability_line[i] += probabilities[i][j]
      probability_line[j] += probabilities[i][j]
  return probability_line


def CombineProbs(probabilities_list):
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
  num_players = len(decls)
  probs = np.zeros([num_players, num_players])
  for i in range(num_players):
    for j in range(i):
      bg_wires = active_wires - sum(decls) + decls[i] + decls[j]
      combinations = 0
      for k in range(int(bg_wires) + 1):  # Consider all distributions
        i_wires = k
        j_wires = bg_wires - k
        if i_wires < decls[i]:  # i has fewer wires than he declared
          probs_i = uf.C(i_wires, decls[i])
        else:  # i has more wires than he declared
          probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
        if j_wires < decls[j]:  # j has fewer wires than he declared
          probs_j = uf.C(j_wires, decls[j])
        else:  # j has more wires than he declared
          probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
        probs[i][j] += uf.C(k, bg_wires) * probs_i * probs_j
        combinations += uf.C(k, bg_wires)
      if combinations != 0:
        probs[i][j] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)  # Normalize
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  num_players = len(decls)
  likelihood = np.zeros([num_players, num_players])
  for i in range(num_players):
    for j in range(i):
      if prior[i][j] == 1:  # The two bad guys have already been found
        return prior
      # How many wires do i and j have if they are bad
      other_decls = np.sum(decls) - decls[i] - decls[j]
      bg_wrs = active_wires + np.sum(found) - other_decls
      # Calculate likelihood of configuration supposing i and j bad guys
      combinations = 0
      for k in range(int(bg_wrs) + 1):  # Consider all distributions
        lklhd_i = uf.Lklhd(hand_size, k, revealed[i], found[i])
        lklhd_j = uf.Lklhd(hand_size, bg_wrs-k, revealed[j], found[j])
        combinations += uf.C(k, bg_wrs)
        likelihood[i][j] += uf.C(k, bg_wrs) * lklhd_i * lklhd_j
      if combinations != 0:
        likelihood[i][j] /= combinations
  marginal = 0
  for i in range(num_players):
    for j in range(i):
      marginal += prior[i][j] * likelihood[i][j]
  if marginal == 0:
    return prior
  posterior = prior.copy()
  for i in range(num_players):
    for j in range(i):
      posterior[i][j] *= likelihood[i][j] / marginal
  return posterior


# Test cases:
# Bad guy declares 0 wires and 1 is found
