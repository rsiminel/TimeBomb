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
      print(" pw : ", p_wire, np.sum(p_wire) / num_players - active_wires / num_players / hand_size)
      print(" em :", H_Min(declarations, total_probs, np.zeros(num_players),
                          np.zeros(num_players), hand_size, active_wires, 3))
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
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i):
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
      bg_wires = active_wires - np.sum(decls) + decls[i] + decls[j]
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
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players])
  marginal = 0
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      if prior[bad1][bad2] == 1:  # The two bad guys have already been found
        return prior
      # How many wires do the supposed bad guys have
      good_decls = np.sum(decls) - decls[bad1] - decls[bad2]
      bg_wires = active_wires + np.sum(found) - good_decls
      # Calculate likelihood of configuration with these bad guys
      combinations = 0
      for bad1_wires in range(int(bg_wires) + 1):  # Consider all distributions
        bad2_wires = bg_wires - bad1_wires
        lklhd_bad1 = uf.Lklhd(hand_size, bad1_wires, revealed[bad1], found[bad1])
        lklhd_bad2 = uf.Lklhd(hand_size, bad2_wires, revealed[bad2], found[bad2])
        comb = uf.C(bad1_wires, bg_wires)
        combinations += comb
        lklhd[bad1][bad2] += comb * lklhd_bad1 * lklhd_bad2
      if combinations != 0:
        lklhd[bad1][bad2] /= combinations
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
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i):
      bg_wires = active_wires + np.sum(found) - np.sum(decls) + decls[i] + decls[j]
      i_wires_avg = 0
      j_wires_avg = 0
      combinations = 0
      for k in range(int(bg_wires) + 1):  # Consider all distributions
        i_wires = k - found[i]
        j_wires = bg_wires - i_wires - found[j]
        if i_wires <= hand_size - revealed[i] + found[i] and i_wires >= found[i]:
          if j_wires <= hand_size - revealed[j] + found[j] and j_wires >= found[j]:
            i_wires_avg += i_wires * uf.C(k, bg_wires)
            j_wires_avg += j_wires * uf.C(k, bg_wires)
            combinations += uf.C(k, bg_wires)
      if combinations != 0:
        i_wires_avg /= combinations
        j_wires_avg /= combinations
      p_wire[i] += probs[i][j] * i_wires_avg
      p_wire[j] += probs[i][j] * j_wires_avg
  lin_probs = DeMatrix(probs)
  for i in range(num_players):
    p_wire[i] += (1 - lin_probs[i]) * (decls[i] - found[i])
    if hand_size - revealed[i] > 0:
      p_wire[i] /= hand_size - revealed[i]
  return p_wire


def H(probs):
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h


def H2(probs):
  h = 0
  for line in probs:
    for p in line:
      if p > 0.0001:
        h += p * np.log2(1/p)
  return h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop):
  if stop <= 0:
    return (H2(probs), [])
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  h = np.zeros(num_players)
  h_wire = 0
  h_not_wire = 0
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
