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
    print("Round ", initial_hand_size - hand_size + 1)
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
    print("bp:", prob_bomb, np.sum(prob_bomb))
    print(" p:", DeMatrix(prob_bad), np.sum(prob_bad))
    print("tp:", comb_probs_line, np.sum(comb_probs_line))
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
      prob_bad, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_bad.copy()
      comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
      print("bp:", prob_bomb, np.sum(prob_bomb))
      print(" p:", DeMatrix(prob_bad), np.sum(prob_bad))
      print("tp:", comb_probs_line, np.sum(comb_probs_line))
      # Test for victory
      if active_wires <= 0:
        print("Good guys win!")
        return
    # Next round
    hand_size -= 1
    print("\n")
  print("Bad guys win!")
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
      print("bp:", prob_bomb, np.sum(prob_bomb))
      print(" p:", DeMatrix(prob_bad), np.sum(prob_bad))
      print("tp:", comb_probs_line, np.sum(comb_probs_line))
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
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      prob_bad, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_bad.copy()
      comb_probs_line = DeMatrix(CombineProbs(probabilities_list))
      if verbosity > 1:
        print("bp:", prob_bomb, np.sum(prob_bomb))
        print(" p:", DeMatrix(prob_bad), np.sum(prob_bad))
        print("tp:", comb_probs_line, np.sum(comb_probs_line))
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
  excess = - active_wires + sum(decls)
  cas_p = 0
  for i in range(num_players):
    for j in range(i):
      for k in range(num_players):
        if i == k:  # A bad guy (i) has the bomb
          if excess > 0:  # The excess is hiding in i and j's decls
            probs[i][j][k] = uf.C(excess, decls[i] + decls[j])
          else:  # The missing wires are in what j did not declare
            probs[i][j][k] = uf.C(-excess, hand_size - decls[j])
        elif j == k:  # A bad guy (j) has the bomb
          if excess > 0:  # The excess hiding in i and j's decls
            probs[i][j][k] = uf.C(excess, decls[i] + decls[j])
          else:  # The missing wires are in what i did not declare
            probs[i][j][k] = uf.C(-excess, hand_size - decls[i])
        else:  # A good guy has the bomb
          hide_bomb = hand_size - decls[k] - 1
          if hide_bomb < 0:  # The bomb is not in k's hand
            probs[i][j][k] = 0
          elif excess > 0:  # The excess hiding in i and j's decls
            probs[i][j][k] = uf.C(excess, decls[i] + decls[j])
          else:  # The wires are in what i, j and k did not declare
            empty = 2 * hand_size - decls[i] - decls[j]
            probs[i][j][k] = uf.C(- excess, empty + hide_bomb)
        cas_p += probs[i][j][k]
  probs /= cas_p
  return probs


def ProbDeclaration2(decls, active_wires, hand_size):
  num_players = decls.shape[0]
  probs = np.full([num_players, num_players], 1 / num_players**2)
  for i in range(num_players):
    for j in range(i):
      bg_wires = active_wires - sum(decls) + decls[i] + decls[j]
      for k in range(num_players):
        if i == k:  # A bad guy (i) has the bomb
          combinations = 0
          for a in range(int(bg_wires) + 1):
            i_wires = a
            j_wires = bg_wires - a
            if decls[i] < i_wires:  # i has more wires than he declared
              probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
            else:  # i has fewer wires than he declared
              probs_i = 0  # Bg with bomb declares extra wires
            if j_wires < decls[j]:  # j has fewer wires than he declared
              probs_j = uf.C(j_wires, decls[j])
            else:  # j has more wires than he declared
              probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
            probs[i][j][k] += uf.C(k, bg_wires) * probs_i * probs_j
            combinations += uf.C(k, bg_wires)
          if combinations != 0:
            probs[i][j][k] /= combinations
        if j == k:  # A bad guy (j) has the bomb
          combinations = 0
          for a in range(int(bg_wires) + 1):
            i_wires = a
            j_wires = bg_wires - a
            if i_wires < decls[i]:  # i has fewer wires than he declared
              probs_i = uf.C(i_wires, decls[i])
            else:  # i has more wires than he declared
              probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
            if decls[j] < j_wires:  # j has more wires than he declared
              probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
            else:  # j has fewer wires than he declared
              probs_j = 0  # Bg with bomb declares extra wires
            probs[i][j][k] += uf.C(k, bg_wires) * probs_i * probs_j
            combinations += uf.C(k, bg_wires)
          if combinations != 0:
            probs[i][j][k] /= combinations
        else:  # A good guy has the bomb
          hide_bomb = hand_size - decls[k] - 1
          if hide_bomb < 0:  # The bomb is not hidden in j's hand
            probs[i][j][k] = 0
          combinations = 0
          for a in range(int(bg_wires) + 1):
            for b in range(int(bg_wires) + 1 - a):
              i_wires = a
              j_wires = b
              k_wires = bg_wires - a - b + decls[k]
              if i_wires < decls[i]:  # i declared more wires than he has
                probs_i = uf.C(i_wires, decls[i])
              else:  # i declared fewer wires than he has
                probs_i = uf.C(i_wires - decls[i], hand_size - decls[i])
              if j_wires < decls[j]:  # j declared more wires than he has
                probs_j = uf.C(j_wires, decls[j])
              else:  # j declared fewer wires than he has
                probs_j = uf.C(j_wires - decls[j], hand_size - decls[j])
              # k can only declare fewer wires than he has
              probs_k = uf.C(k_wires - decls[k], hide_bomb)
              combinations += 1
              probs[i][j][k] += 1 * probs_i * probs_j * probs_k
          if combinations != 0:
            probs[i][j][k] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  num_players = decls.size
  lklhd = np.zeros([num_players, num_players, num_players])
  for i in range(num_players):
    for j in range(i):
      for k in range(num_players):
        if prior[i][j][k] == 1:  # Bad guys and the bomb found
          return prior
        if i == k:  # A bad guy (i) has the bomb
          if revealed[i] >= hand_size:  # All of i's hand is not bomb
            lklhd[i][j][k] = 0
            continue
          # How many wires do i and j have if they are bad guys
          m = active_wires + np.sum(found) - np.sum(decls) + decls[i] + decls[j]
          if hand_size + decls[i] < m:
            lklhd[i][j][k] = 0
            continue
          # Likelihood of conf if i has a bomb + i and j are bad guys
          combinations = 0
          for a in range(int(m) + 1):  # Consider all distributions
            lklhd_i = uf.Lklhd(hand_size - 1, a, revealed[i], found[i])
            lklhd_j = uf.Lklhd(hand_size, m - a, revealed[j], found[j])
            combinations += uf.C(a, m)
            lklhd[i][j][k] += uf.C(a, m) * lklhd_i * lklhd_j
          if combinations != 0:
            lklhd[i][j][k] /= combinations
        elif j == k:  # A bad guy (j) has the bomb
          if revealed[j] >= hand_size:  # All of j's hand is not bomb
            lklhd[i][j][k] = 0
            continue
          # How many wires do i and j have if they are bad guys
          m = active_wires + np.sum(found) - np.sum(decls) + decls[i] + decls[j]
          if hand_size + decls[j] < m:
            lklhd[i][j][k] = 0
            continue
          # Likelihood of conf if j has a bomb + i and j are bad guys
          combinations = 0
          for a in range(int(m) + 1):  # Consider all distributions
            lklhd_i = uf.Lklhd(hand_size, m - a, revealed[i], found[i])
            lklhd_j = uf.Lklhd(hand_size - 1, a, revealed[j], found[j])
            combinations += uf.C(a, m)
            lklhd[i][j][k] += uf.C(a, m) * lklhd_i * lklhd_j
          if combinations != 0:
            lklhd[i][j][k] /= combinations
        else:  # A good guy has the bomb
          if revealed[k] >= hand_size:  # All of k's hand is not bomb
            lklhd[i][j][k] = 0
            continue
          m = active_wires + np.sum(found) - np.sum(decls) + decls[i] + decls[j]
          # Likelihood of configuration if i, j and k are lying
          combinations = 0
          for a in range(int(m) + 1):  # Consider all distributions
            for b in range(int(m) + 1 - a):
              lklhd_i = uf.Lklhd(hand_size, m - a - b, revealed[i], found[i])
              lklhd_j = uf.Lklhd(hand_size, b, revealed[j], found[j])
              lklhd_k = uf.Lklhd(hand_size - 1, decls[k] + a, revealed[k], found[k])
              combinations += 1
              lklhd[i][k] += lklhd_i * lklhd_j * lklhd_k
          if combinations != 0:
            lklhd[i][k] /= combinations
  marginal = 0
  for i in range(num_players):
    for j in range(i):
      for k in range(num_players):
        marginal += prior[i][j][k] * lklhd[i][j][k]
  if marginal == 0:
    return prior
  posterior = prior.copy()
  for i in range(num_players):
    for j in range(i):
      for k in range(num_players):
        posterior[i][j][k] *= lklhd[i][j][k] / marginal
  return posterior
