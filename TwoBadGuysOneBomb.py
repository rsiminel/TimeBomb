# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:37:52 2024

@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


def Play(players=["Alice", "Bob", "Clara", "Darryl", "Erica", "Fred"], initial_hand_size=5):
  num_players = len(players)
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Declare your wires
    declarations = np.zeros(num_players)
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, active_wires, hand_size)
    prob_bad, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_bad.copy())
    comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
    print("bp:", prob_bomb)
    print(" p:", DeMatrix(prob_bad))
    print("tp:", comb_probs_line)
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    for i in range(num_players):
      print("\n Cut number", i + 1)
      cutee_str = input("Who's wire has been cut? ")
      while cutee_str not in players:
        cutee_str = input("You must have made a typo. Who? ")
      cutee = 0
      for j in range(num_players):
        if players[j] == cutee_str:
          cutee = j
      revealed[cutee] += 1
      num_wires -= 1
      shown = int(input("Did you reveal\n" + " 0- an inactive wire\n 1- an active wire\n 2- the bomb"))
      while shown not in [0, 1, 2]:
        shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here."))
      if shown == 2:
        print("The Bomb was detonated. Bad guys win!")
        return
      if shown == 1:
        found[cutee] += 1
        active_wires -= 1
      print("r:", revealed)
      print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      prob_bad, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_bad.copy()
      comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
      print("bp:", prob_bomb)
      print(" p:", DeMatrix(prob_bad))
      print("tp:", comb_probs_line)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


# 2 bad guys, 1 bomb
# Suppositions : good guys tell the truth unless they have the bomb
#                if a good guy has the bomb, he declares fewer wires than he has
#                bad guys lie randomly unless they have the bomb
#                if a bad guy has the bomb, he declares more wires than he has
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
    prob_bad, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_bad.copy())
    comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
    if verbosity > 1:
      print("bp:", prob_bomb)
      print(" p:", DeMatrix(prob_bad))
      print("tp:", comb_probs_line)
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
      if bomb[cutee] == 1 and randy == hand_size - revealed[cutee]:
        if verbosity > 0:
          print("The Bomb was detonated. Bad guys win!")
        return (0, DeMatrix(CombineProbs(probabilities_list)), roles)
      elif randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, active_wires, hand_size)
      prob_bad, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_bad.copy()
      comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
      if verbosity > 1:
        print("bp:", prob_bomb)
        print(" p:", DeMatrix(prob_bad))
        print("tp:", comb_probs_line)
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


def DeMatrix(probabilities):
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i + 1):
      probability_line[i] += probabilities[i][j]
      probability_line[j] += probabilities[i][j]
  return probability_line


def DeTensor(probabilities):
  num_players = len(probabilities)
  probability_bad = np.zeros([num_players, num_players])
  probability_bomb = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i):
      for k in range(num_players):
        probability_bad[i][j] += probabilities[i][j][k]
        probability_bomb[k] += probabilities[i][j][k]
  return (probability_bad, probability_bomb)


def ProbDeclaration(decls, active_wires, hand_size):
  num_players = decls.shape[0]
  probs = np.zeros([num_players, num_players, num_players])
  for i in range(num_players):
    for j in range(i):
      bg_wires = int(active_wires - np.sum(decls) + decls[i] + decls[j])
      for k in range(num_players):
        combinations = 0
        if i == k:  # A bad guy (i) has the bomb
          for i_wires in range(bg_wires + 1):
            j_wires = bg_wires - i_wires
            if i_wires > decls[i]:  # i has more wires than declared
              probs_i = 0  # i must have =fewer wires than declared (bg w. bomb)
            else:  # i has =fewer wires than declared
              probs_i = uf.C(i_wires, decls[i])
            if j_wires < decls[j]:  # j has fewer wires than declared
              probs_j = uf.C(j_wires, decls[j])
            else:  # j has =more wires than declared
              probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
            probs[i][j][k] += uf.C(i_wires, bg_wires) * probs_i * probs_j
            combinations += uf.C(i_wires, bg_wires)
        elif j == k:  # A bad guy (j) has the bomb
          for i_wires in range(bg_wires + 1):
            j_wires = bg_wires - i_wires
            if i_wires < decls[i]:  # i has fewer wires than declared
              probs_i = uf.C(i_wires, decls[i])
            else:  # i has =more wires than declared
              probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
            if j_wires > decls[j]:  # j has more wires than declared
              probs_j = 0  # j must have =fewer wires than declared (bg w. bomb)
            else:  # j has =fewer wires than declared
              probs_j = uf.C(j_wires, decls[j])
            probs[i][j][k] += uf.C(i_wires, bg_wires) * probs_i * probs_j
            combinations += uf.C(i_wires, bg_wires)
        else:  # A good guy has the bomb
          if hand_size - decls[k] - 1 < 0:  # The bomb is not hidden in j's hand
            probs[i][j][k] = 0
            continue
          for i_wires in range(bg_wires + 1):
            for j_wires in range(bg_wires - i_wires + 1):
              k_wires = bg_wires - i_wires - j_wires + decls[k]
              if i_wires < decls[i]:  # i has fewer wires than declared
                probs_i = uf.C(i_wires, decls[i])
              else:  # i has =more wires than declared
                probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
              if j_wires < decls[j]:  # j has fewer wires than declared
                probs_j = uf.C(j_wires, decls[j])
              else:  # i has =more wires than declared
                probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
              # k must have =more wires than declared (gg w. bomb)
              probs_k = uf.C(k_wires - decls[k], hand_size - decls[k] - 1)
              probs[i][j][k] += uf.C3(i_wires, j_wires, bg_wires) * probs_i * probs_j * probs_k
              combinations += uf.C3(i_wires, j_wires, bg_wires)
        if combinations != 0:
          probs[i][j][k] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, active_wires, hand_size):
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players, num_players])
  marginal = 0
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      bg_wires = int(active_wires + np.sum(found) - np.sum(decls) + decls[bad1] + decls[bad2])
      for bom in range(num_players):
        if prior[bad1][bad2][bom] == 1:  # Bad guys and the bomb found
          return prior
        if revealed[bom] >= hand_size:  # All of bomber's hand is not bomb
            lklhd[bad1][bad2][bom] = 0
            continue
        if bad1 == bom:  # A bad guy (bad1) has the bomb
          if hand_size + decls[bad1] < bg_wires:
            lklhd[bad1][bad2][bom] = 0
            continue
          # Likelihood of conf with supposed bad guys if bad1 has a bomb
          combinations = 0
          for bad1_wires in range(bg_wires + 1):  # Consider all distributions
            bad2_wires = bg_wires - bad1_wires
            lklhd_bad1 = uf.Lklhd(hand_size - 1, bad1_wires, revealed[bad1], found[bad1])
            lklhd_bad2 = uf.Lklhd(hand_size, bad2_wires, revealed[bad2], found[bad2])
            combinations += uf.C(bad1_wires, bg_wires)
            lklhd[bad1][bad2][bom] += uf.C(bad1_wires, bg_wires) * lklhd_bad1 * lklhd_bad2
          if combinations != 0:
            lklhd[bad1][bad2][bom] /= combinations
        elif bad2 == bom:  # A bad guy (bad2) has the bomb
          if hand_size + decls[bad2] < bg_wires:
            lklhd[bad1][bad2][bom] = 0
            continue
          # Likelihood of conf with supposed bad guys if bad2 has a bomb
          combinations = 0
          for bad1_wires in range(bg_wires + 1):  # Consider all distributions
            bad2_wires = bg_wires - bad1_wires
            lklhd_bad1 = uf.Lklhd(hand_size, bad1_wires, revealed[bad1], found[bad1])
            lklhd_bad2 = uf.Lklhd(hand_size - 1, bad2_wires, revealed[bad2], found[bad2])
            combinations += uf.C(bad1_wires, bg_wires)
            lklhd[bad1][bad2][bom] += uf.C(bad1_wires, bg_wires) * lklhd_bad1 * lklhd_bad2
          if combinations != 0:
            lklhd[bad1][bad2][bom] /= combinations
        else:  # A good guy has the bomb
          # Likelihood of configuration if bad1, bad2 and bomber are lying
          combinations = 0
          for bad1_wires in range(bg_wires + 1):  # Consider all distributions
            for bad2_wires in range(bg_wires + 1 - bad1_wires):
              bom_wires = bg_wires - bad1_wires - bad2_wires + decls[bom]
              lklhd_bad1 = uf.Lklhd(hand_size, bad1_wires, revealed[bad1], found[bad1])
              lklhd_bad2 = uf.Lklhd(hand_size, bad2_wires, revealed[bad2], found[bad2])
              lklhd_k = uf.Lklhd(hand_size - 1, bom_wires, revealed[bom], found[bom])
              combinations += uf.C3(bad1_wires, bad2_wires, bg_wires)
              lklhd[bad1][bad2][bom] += uf.C3(bad1_wires, bad2_wires, bg_wires) * lklhd_bad1 * lklhd_bad2 * lklhd_k
          if combinations != 0:
            lklhd[bad1][bad2][bom] /= combinations
        for good in range(num_players):  # Likelihood of everyone else's configurations
          if good != bad1 and good != bad2 and good != bom:
            lklhd[bad1][bad2][bom] *= uf.Lklhd(hand_size, decls[good], revealed[good], found[good])
        marginal += prior[bad1][bad2][bom] * lklhd[bad1][bad2][bom]
  if marginal == 0:
    return prior
  posterior = prior.copy()
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      for bom in range(num_players):
        posterior[bad1][bad2][bom] *= lklhd[bad1][bad2][bom] / marginal
  return posterior
