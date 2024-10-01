# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:37:52 2024

@author: Remy
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf
from tabulate import tabulate


def DisplayProbs(players, probs, probs_list, decls, revealed, found, hand_size, active_wires):
  num_players = decls.size
  prob_bad, prob_bomb = DeTensor(probs)
  p_bomb = np.zeros(num_players)
  for i in range(num_players):
    if hand_size - revealed[i] != 0:
      p_bomb[i] = prob_bomb[i] / (hand_size - revealed[i])
  comb_probs = DeMatrix(CombineProbs(probs_list))
  total_probs = CombineNonHomoProbs(CombineProbs(probs_list[0:-1]), probs)
  p_wire = P_wire(decls, total_probs, revealed, found, hand_size, active_wires)
  curr_points = num_players - active_wires
  score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
  p_wire_rand = active_wires / (num_players * hand_size - np.sum(revealed))
  p_bomb_rand = 1 / (num_players * hand_size - np.sum(revealed))
  table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
            [players[i], p_wire[i]*100, p_bomb[i]*100, comb_probs[i]*100, score[i]] for i in range(num_players)] + [
            ["Average", p_wire_rand*100, p_bomb_rand*100, 2/num_players*100, np.sum(score)/num_players]]
  print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
  return


def Play(players=["Alice", "Bob", "Clara", "Darryl", "Erica", "Fred"], initial_hand_size=5):
  num_players = len(players)
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  zeros = np.zeros(num_players)
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Declare your wires
    declarations = zeros.copy()
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    prob_bad, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_bad.copy())
    DisplayProbs(players, probabilities, probabilities_list, declarations, zeros, zeros, hand_size, active_wires)
    # Cut wires
    found = zeros.copy()
    revealed = zeros.copy()
    for i in range(num_players):
      event = -1
      while event != 0:
        event = int(input("What happened?\n 0- a wire got cut\n 1- something sus\n"))
        if event == 1:
          ProbSus(players, probabilities)
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
      DisplayProbs(players, probs, probabilities_list, declarations, revealed, found, hand_size, active_wires)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


def ProbSus(players, probs):  # Modifies probs in-place
  num_players = len(players)
  sus_str = input("Who is sus? ")
  sus = -1
  while sus_str not in players:
    sus_str = input("You must have made a typo. Who? ")
  for player in range(num_players):
    if players[player] == sus_str:
      sus = player
  (probs_bad, probs_bom) = DeTensor(probs)
  probs_bad = DeMatrix(probs_bad)
  prob_bib = np.sum(probs[sus, :, sus]) + np.sum(probs[:, sus, sus])
  prob_bob = probs_bad[sus] - prob_bib
  prob_gib = probs_bom[sus] - prob_bib
  prob_gob = 1 - prob_bib - prob_bob - prob_gib
  lklhd_gob = int(input("What is the likelihood that they would do this as a good guy without the bomb ?"))/100
  lklhd_gib = int(input("What is the likelihood that they would do this as a good guy with the bomb ?"))/100
  lklhd_bob = int(input("What is the likelihood that they would do this as a bad guy without the bomb?"))/100
  lklhd_bib = int(input("What is the likelihood that they would do this as a bad guy with the bomb?"))/100
  marginal = prob_bib * lklhd_bib + prob_bob * lklhd_bob + prob_gib * lklhd_gib + prob_gob * lklhd_gob
  lklhd = np.full([num_players, num_players, num_players], lklhd_gob)
  lklhd[sus, :, :] = lklhd_bob
  lklhd[:, sus, :] = lklhd_bob
  lklhd[:, :, sus] = lklhd_gib
  lklhd[sus, :, sus] = lklhd_bib
  lklhd[:, sus, sus] = lklhd_bib
  probs *= lklhd / marginal
  return probs


# 2 bad guys, 1 bomb
# Suppositions : good guys tell the truth unless they have the bomb
#                if a good guy has the bomb, he declares fewer wires than he has
#                bad guys lie randomly unless they have the bomb
#                if a bad guy has the bomb, he declares more wires than he has
def PlayAuto(CutStrategy, num_players=6, initial_hand_size=5, verbosity=2):
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  zeros = np.zeros(num_players)
  # Distributing roles
  roles = zeros.copy()
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
    bomb = zeros.copy()
    randy = randint(0, num_players - 1)
    while wires[randy] >= hand_size:
      randy = randint(0, num_players - 1)
    bomb[randy] = 1
    if verbosity > 0:
      print("w:", wires)
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
    prob_bad, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_bad.copy())
    if verbosity > 1:
      players = []
      for i in range(num_players):
        if roles[i] == 1:
          if bomb[i] == 1:
            players += ["bad bomber"]
          else:
            players += ["bad guy"]
        elif bomb[i] == 1:
          players += ["good bomber"]
        else:
          players += ["good guy"]
      DisplayProbs(players, probabilities, probabilities_list, declarations, zeros, zeros, hand_size, active_wires)
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    probs = probabilities.copy()
    cutee = -1
    for i in range(num_players):
      if verbosity > 0:
        print("Cut number", i + 1)
      cutee = CutStrategy(declarations, probs, revealed, found, hand_size, active_wires, cutee)
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
      if verbosity > 1:
        DisplayProbs(players, probs, probabilities_list, declarations, revealed, found, hand_size, active_wires)
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
  if probabilities_list == []:
    return np.array([])
  num_tests = len(probabilities_list)
  num_players = probabilities_list[0].shape[0]
  probabilities = np.zeros([num_players, num_players])
  for i in range(num_players):
    for j in range(num_players):
      probabilities[i][j] = 1
      for k in range(num_tests):
        probabilities[i][j] *= probabilities_list[k][i][j]
  if np.sum(probabilities) == 0:
    return probabilities
  probabilities /= np.sum(probabilities)
  return probabilities


def CombineNonHomoProbs(prob_bad, probs):
  if prob_bad.size == 0:
    return probs
  num_players = prob_bad[0].size
  new_probs = probs.copy()
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      for bom in range(num_players):
        new_probs[bad1][bad2][bom] *= prob_bad[bad1][bad2]
  new_probs /= np.sum(new_probs)
  return new_probs


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


def ProbDeclaration(decls, hand_size, active_wires):
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


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
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


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for bad1 in range(num_players):
    for bad2 in range(bad1):
      bg_wires = active_wires + np.sum(found) - np.sum(decls) + decls[bad1] + decls[bad2]
      for bom in range(num_players):
        bad1_wires_avg = 0
        bad2_wires_avg = 0
        bom_wires_avg = 0
        combinations = 0
        if bad1 == bom:  # A bad guy (bad1) has the bomb
          for k in range(int(bg_wires) + 1):  # Consider all distributions
            bad1_wires = k - found[bad1]
            bad2_wires = bg_wires - bad1_wires - found[bad2]
            if bad1_wires + found[bad1] <= decls[bad1]:  # (bg w. bomb)
              if bad1_wires <= hand_size - revealed[bad1] - 1 and bad1_wires >= 0:
                if bad2_wires <= hand_size - revealed[bad2] and bad2_wires >= 0:
                  bad1_wires_avg += bad1_wires * uf.C(k, bg_wires)
                  bad2_wires_avg += bad2_wires * uf.C(k, bg_wires)
                  combinations += uf.C(k, bg_wires)
        elif bad2 == bom:  # A bad guy (bad2) has the bomb
          for k in range(int(bg_wires) + 1):  # Consider all distributions
            bad1_wires = k - found[bad1]
            bad2_wires = bg_wires - bad1_wires - found[bad2]
            if bad2_wires + found[bad2] <= decls[bad2]:  # (bg w. bomb)
              if bad1_wires <= hand_size - revealed[bad1] and bad1_wires >= 0:
                if bad2_wires <= hand_size - revealed[bad2] - 1 and bad2_wires >= 0:
                  bad1_wires_avg += bad1_wires * uf.C(k, bg_wires)
                  bad2_wires_avg += bad2_wires * uf.C(k, bg_wires)
                  combinations += uf.C(k, bg_wires)
        else:  # A good guy has the bomb
          for i in range(int(bg_wires) + 1):  # Consider all distributions
            for j in range(int(bg_wires) + 1 - i):
              bad1_wires = i - found[bad1]
              bad2_wires = j - found[bad2]
              bom_wires = bg_wires - bad1_wires - bad2_wires - found[bom] + decls[bom]
              if bom_wires + found[bom] >= decls[bom]:  # (gg w. bomb)
                if bad1_wires <= hand_size - revealed[bad1] and bad1_wires >= 0:
                  if bad2_wires <= hand_size - revealed[bad2] and bad2_wires >= 0:
                    if bom_wires <= hand_size - revealed[bom] and bom_wires >= 0:
                      comb = uf.C3(i, j, bg_wires)
                      bad1_wires_avg += bad1_wires * comb
                      bad2_wires_avg += bad2_wires * comb
                      bom_wires_avg += bom_wires * comb
                      combinations += comb
        if combinations != 0:
          bad1_wires_avg /= combinations
          bad2_wires_avg /= combinations
          bom_wires_avg /= combinations
        p_wire[bad1] += probs[bad1][bad2][bom] * bad1_wires_avg
        p_wire[bad2] += probs[bad1][bad2][bom] * bad2_wires_avg
        p_wire[bom] += probs[bad1][bad2][bom] * bom_wires_avg
  (p_bad, _) = DeTensor(probs)
  lin_probs = DeMatrix(p_bad)
  for good in range(num_players):
    p_wire[good] += (1 - lin_probs[good]) * (decls[good] - found[good])
    if hand_size - revealed[good] > 0:
      p_wire[good] /= hand_size - revealed[good]
  return p_wire


# Cut Strategies
def CutRandom(revealed, hand_size):
  num_players = revealed.size
  cutee = randint(0, num_players - 1)
  while revealed[cutee] >= hand_size:
    cutee = randint(0, num_players - 1)
  return cutee


def CutMaxWires(decls, probs, revealed, found, hand_size, active_wires):
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  cutee = -1
  max_prob = 0
  for i in range(num_players):
    if p_wire[i] > max_prob:
      cutee = i
      max_prob = p_wire[i]
  return cutee


def CutMaxBomb(probs, revealed, hand_size):
  num_players = revealed.size
  (prob_bad, prob_bomb) = DeTensor(probs)
  cutee = -1
  max_prob = 0
  for i in range(num_players):
    p_bomb = 0
    if hand_size - revealed[i] != 0:
      p_bomb = prob_bomb[i] / (hand_size - revealed[i])
    if p_bomb > max_prob:
      cutee = i
      max_prob = p_bomb
  return cutee


def CutMaxScore(decls, probs, revealed, found, hand_size, active_wires, curr_cut):
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  (prob_bad, prob_bomb) = DeTensor(probs)
  p_bomb = np.zeros(num_players)
  for i in range(num_players):
    if hand_size - revealed[i] != 0:
      p_bomb[i] = prob_bomb[i] / (hand_size - revealed[i])
  curr_points = num_players - active_wires
  score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
  cutee = -1
  max_score = 0
  for i in range(num_players):
    if revealed[i] < hand_size and i != curr_cut:
      if score[i] > max_score:
        cutee = i
        max_score = score[i]
  return cutee


def CutMinScore(decls, probs, revealed, found, hand_size, active_wires):
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  (prob_bad, prob_bomb) = DeTensor(probs)
  p_bomb = np.zeros(num_players)
  for i in range(num_players):
    if hand_size - revealed[i] != 0:
      p_bomb[i] = prob_bomb[i] / (hand_size - revealed[i])
  curr_points = num_players - active_wires
  score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
  cutee = -1
  min_score = curr_points + 2
  for i in range(num_players):
    if score[i] < min_score:
      cutee = i
      min_score = score[i]
  return cutee


# Entropy Calculations (deprecated)
def H(probs):
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h


def H2(probs):
  h = 0
  for line in probs:
    h += H(line)
  return h


def H3(probs):
  h = 0
  for plane in probs:
    h += H2(plane)
  return h


def NextH3(decls, probs, revealed, found, hand_size, active_wires):
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
      info_wire = H3(ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires))
    if p_wire[cutee] < 0.9999:
      info_not_wire = H3(ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires))
  h = p_wire * info_wire + (1 - p_wire) * info_not_wire
  return H3(probs) - h


def NextH2(decls, probs, revealed, found, hand_size, active_wires):
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
      (probs_bad, _) = DeTensor(ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires))
      info_wire = H2(probs_bad)
    if p_wire[cutee] < 0.9999:
      (probs_bad, _) = DeTensor(ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires))
      info_not_wire = H2(probs_bad)
  h = p_wire * info_wire + (1 - p_wire) * info_not_wire
  (probs_bad, _) = DeTensor(probs)
  return H2(probs_bad) - h
