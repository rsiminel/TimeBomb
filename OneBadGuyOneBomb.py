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
    wires = uf.DistributeWires(num_players, hand_size, active_wires)
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
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    prob_bad, prob_bomb = DeMatrix(probabilities)
    probabilities_list.append(prob_bad.copy())
    comb_probs = CombineProbs(probabilities_list)
    total_probs = CombineNonHomoProbs(comb_probs, probabilities)
    p_wire = P_wire(declarations, total_probs, np.zeros(num_players), np.zeros(num_players), hand_size, active_wires)
    if verbosity > 1:
      print(" p:", prob_bad, H(prob_bad))
      print("tp:", comb_probs, H(comb_probs))
      print("bp:", prob_bomb, H(prob_bomb))
      print("pw:", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size))
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
      total_probs = CombineNonHomoProbs(comb_probs, probabilities)
      p_wire = P_wire(declarations, total_probs, revealed, found, hand_size, active_wires)
      if verbosity > 1:
        print(" p:", prob_bad, H(prob_bad))
        print("tp:", comb_probs, H(comb_probs))
        print("bp:", prob_bomb, H(prob_bomb))
        print("pw:", p_wire, np.sum(p_wire) / num_players - active_wires / (num_players * hand_size - np.sum(revealed)))
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


def CombineProbs(probabilities_list):
  num_tests = len(probabilities_list)
  num_players = len(probabilities_list[0])
  probabilities = np.full([num_players], 1.)
  for i in range(num_players):
    for test in range(num_tests):
      probabilities[i] *= probabilities_list[test][i]
  probabilities /= np.sum(probabilities)
  return probabilities


def CombineNonHomoProbs(prob_bad, probs):
  num_players = prob_bad.size
  new_probs = probs.copy()
  for bad in range(num_players):
    for bom in range(num_players):
      new_probs[bad][bom] *= prob_bad[bad]
  new_probs /= np.sum(new_probs)
  return new_probs


def ProbDeclaration(decls, hand_size, active_wires):
  num_players = decls.shape[0]
  probs = np.full([num_players, num_players], 1 / num_players**2)
  for bad in range(num_players):
    bg_wires = active_wires - np.sum(decls) + decls[bad]
    for bom in range(num_players):
      if bg_wires < 0:
        probs[bad][bom] = 0
      elif bad == bom:  # The bad guy has the bomb
        if bg_wires > decls[bad]:  # bad has more wires than declared
          probs[bad][bom] = 0  # i must have =fewer wires than declared (bg w. bomb)
        else:  # i has =fewer wires than he declared
          probs[bad][bom] = uf.C(bg_wires, decls[bad])
      else:  # A good guy has the bomb
        if hand_size - decls[bom] - 1 < 0:  # The bomb is not hidden in j's hand
          probs[bad][bom] = 0
          continue
        combinations = 0
        for bad_wires in range(int(bg_wires) + 1):
          bom_wires = bg_wires - bad_wires + decls[bom]
          if bad_wires < decls[bad]:  # i has fewer wires than declared
            probs_bad = uf.C(bad_wires, decls[bad])
          else:  # i has =more wires than declared
            probs_bad = uf.C(bad_wires - decls[bad], hand_size - decls[bad])
          # j must have =more wires than declared (gg w. bomb)
          probs_bom = uf.C(bom_wires - decls[bom], hand_size - decls[bom] - 1)
          combinations += uf.C(bad_wires, bg_wires)
          probs[bad][bom] += uf.C(bad_wires, bg_wires) * probs_bad * probs_bom
        if combinations != 0:
          probs[bad][bom] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players])
  marginal = 0
  for bad in range(num_players):
    bg_wires = int(active_wires + np.sum(found) - np.sum(decls) + decls[bad])
    for bom in range(num_players):
      if prior[bad][bom] == 1:  # The bad guy and the bomb have been found
        return prior
      if bad == bom:  # A bad guy has the bomb
        if revealed[bad] >= hand_size:  # All of bad's hand is not bomb
          lklhd[bad][bom] = 0
          continue
        if bg_wires > decls[bad]:  # bomber has more wires than declared
          lklhd[bad][bom] = 0  # bad must have =fewer wires than declared (bg w. bomb)
          continue
        # Likelihood of configuration if the supposed bad guy has the bomb
        lklhd[bad][bom] = uf.Lklhd(hand_size - 1, bg_wires, revealed[bad], found[bad])
      else:  # A good guy has the bomb
        if revealed[bom] >= hand_size:  # All of bomber's hand is not bomb
          lklhd[bad][bom] = 0
          continue
        # Likelihood of configuration if bad and bomb are lying
        combinations = 0
        for bad_wires in range(bg_wires + 1):  # Consider all distributions
          bom_wires = bg_wires - bad_wires + decls[bom]
          lklhd_bad = uf.Lklhd(hand_size, bad_wires, revealed[bad], found[bad])
          lklhd_bom = uf.Lklhd(hand_size - 1, bom_wires, revealed[bom], found[bom])
          combinations += uf.C(bad_wires, bg_wires)
          lklhd[bad][bom] += uf.C(bad_wires, bg_wires) * lklhd_bad * lklhd_bom
        if combinations != 0:
          lklhd[bad][bom] /= combinations
      for good in range(num_players):  # Likelihood of everyone else's configurations
        if good != bad and good != bom:
          lklhd[bad][bom] *= uf.Lklhd(hand_size, decls[good], revealed[good], found[good])
      marginal += prior[bad][bom] * lklhd[bad][bom]
  posterior = prior.copy()
  for bad in range(num_players):
    for bom in range(num_players):
      posterior[bad][bom] *= lklhd[bad][bom] / marginal
  return posterior


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for bad in range(num_players):
    for bom in range(num_players):
      bg_wires = active_wires + np.sum(found) - np.sum(decls) + decls[bad]
      if bad == bom:  # A bad guy has the bomb
        p_wire[bad] += probs[bad][bom] * bg_wires
      else:  # A good guy has the bomb
        bad_wires_avg = 0
        bom_wires_avg = 0
        combinations = 0
        for k in range(int(bg_wires) + 1):  # Consider all distributions
          bad_wires = k - found[bad]
          bom_wires = bg_wires - bad_wires - found[bom] + decls[bom]
          if bad_wires <= hand_size - revealed[bad] and bad_wires >= 0:
            if bom_wires <= hand_size - revealed[bom] and bom_wires >= 0:
              bad_wires_avg += bad_wires * uf.C(k, bg_wires)
              bom_wires_avg += bom_wires * uf.C(k, bg_wires)
              combinations += uf.C(k, bg_wires)
        if combinations != 0:
          bad_wires_avg /= combinations
          bom_wires_avg /= combinations
        p_wire[bad] += probs[bad][bom] * bad_wires_avg
        p_wire[bom] += probs[bad][bom] * bom_wires_avg
  (p_bad, _) = DeMatrix(probs)
  for good in range(num_players):
    p_wire[good] += (1 - p_bad[good]) * (decls[good] - found[good])
    if hand_size - revealed[good] > 0:
      p_wire[good] /= hand_size - revealed[good]
  return p_wire


def H(probs):
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h
