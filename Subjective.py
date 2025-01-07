# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf
from tabulate import tabulate
import OneBadGuyNoBomb as obnb
import OneBadGuyOneBomb as obob
import TwoBadGuysNoBomb as tbnb
import TwoBadGuysOneBomb as tbob


def DisplayProbs(players, p_bad, p_bomb, p_wire, score, p_wire_rand, p_bomb_rand):
  num_players = len(players)
  table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
            [players[i], p_wire[i]*100, p_bomb[i]*100, p_bad[i]*100, score[i]] for i in range(num_players)] + [
            ["Average", p_wire_rand*100, p_bomb_rand*100, 2/num_players*100, np.sum(score)/num_players]]
  print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
  return


def Play(other_players=["Alice", "Bob", "Clara", "Darryl", "Erica"], initial_hand_size=5):
  # Initialize game
  num_players = len(other_players)
  hand_size = initial_hand_size
  num_wires = (num_players + 1) * hand_size
  active_wires = num_players + 1
  is_bad = int(input("Are you a\n 0- good guy\n 1- bad guy"))
  while is_bad not in [0, 1]:
    is_bad = int(input("Sorry, I'm looking for a 0 or a 1 here."))
  zeros = np.zeros(num_players)
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Wire declarations
    player_bomb = int(input("Do you\n 0- not have the bomb\n 1- have the bomb"))
    while player_bomb not in [0, 1]:
      player_bomb = int(input("Sorry, I'm looking for a 0 or a 1 here."))
    player_wires = int(input("How many wires do you have?"))
    decls = zeros.copy()
    for i in range(num_players):
      decls[i] = int(input("How many wires does " + other_players[i] + " say they have? "))
    print("d:", decls)
    # Calculate probabilities
    p_bomb = zeros.copy()
    p_wire = np.full(num_players, (active_wires - player_wires) / (num_players * hand_size))
    if player_bomb:
      if is_bad:
        probabilities = obnb.ProbDeclaration(decls, hand_size, active_wires - player_wires)
        probabilities_list.append(probabilities.copy())
        p_bad = obnb.CombineProbs(probabilities_list)
      else:
        probabilities = tbnb.ProbDeclaration(decls, hand_size, active_wires - player_wires)
        probabilities_list.append(probabilities.copy())
        p_bad = tbnb.DeMatrix(tbnb.CombineProbs(probabilities_list))
    else:
      if is_bad:
        probabilities = obob.ProbDeclaration(decls, hand_size, active_wires - player_wires)
        p_bad, p_bomb = obob.DeMatrix(probabilities)
        probabilities_list.append(p_bad.copy())
        p_bad = obob.CombineProbs(probabilities_list)
      else:
        probabilities = tbob.ProbDeclaration(decls, hand_size, active_wires - player_wires)
        total_probs = tbob.CombineNonHomoProbs(tbob.CombineProbs(probabilities_list[0:-1]), probabilities)
        p_wire = tbob.P_wire(decls, total_probs, zeros.copy(), zeros.copy(), hand_size, active_wires - player_wires)
        p_bad, p_bomb = tbob.DeTensor(probabilities)
        probabilities_list.append(p_bad.copy())
        p_bad = tbob.DeMatrix(tbob.CombineProbs(probabilities_list))
    curr_points = num_players - active_wires
    p_bomb_rand = player_bomb / (num_players * hand_size)
    p_wire_rand = (active_wires - player_wires) / (num_players * hand_size)
    score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
    table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
             [other_players[i], p_wire[i]*100, p_bomb[i]*100, p_bad[i]*100, score[i]] for i in range(num_players)] + [
             ["Average", p_wire_rand*100, p_bomb_rand*100, (2-is_bad)/num_players*100, np.sum(score)/num_players]]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
    # Cut wires
    found = zeros.copy()
    revealed = zeros.copy()
    for i in range(num_players + 1):
      print("\n Cut number", i + 1)
      cutee_str = input("Who's wire has been cut? ")
      cutee = -1
      for j in range(num_players):
        if other_players[j] == cutee_str:
          cutee = j
          revealed[cutee] += 1
      num_wires -= 1
      shown = int(input("Did you reveal\n 0- an inactive wire\n 1- an active wire\n 2- the bomb"))
      while shown not in [0, 1, 2]:
        shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here."))
      if shown == 2:
        print("The Bomb was detonated. Bad guys win!")
        return
      if shown == 1:
        active_wires -= 1
        if cutee == -1: player_wires -= 1
        else: found[cutee] += 1
      print("r:", revealed)
      print("f:", found)
      # Update probabilities
      p_wire = np.full(num_players, (active_wires - player_wires) / (num_players * hand_size - np.sum(revealed)))
      if player_bomb:
        if is_bad:
          probs = obnb.ProbCut(decls, probabilities, revealed, found, hand_size, active_wires - player_wires)
          probabilities_list[-1] = probs.copy()
          p_bad = obnb.CombineProbs(probabilities_list)
        else:
          probs = tbnb.ProbCut(decls, probabilities, revealed, found, hand_size, active_wires - player_wires)
          probabilities_list[-1] = probs.copy()
          p_bad = tbnb.DeMatrix(tbnb.CombineProbs(probabilities_list))
      else:
        if is_bad:
          probs = obob.ProbCut(decls, probabilities, revealed, found, hand_size, active_wires - player_wires)
          p_bad, p_bomb = obob.DeMatrix(probs)
          probabilities_list[-1] = p_bad.copy()
          p_bad = obob.CombineProbs(probabilities_list)
        else:
          probs = tbob.ProbCut(decls, probabilities, revealed, found, hand_size, active_wires - player_wires)
          total_probs = tbob.CombineNonHomoProbs(tbob.CombineProbs(probabilities_list[0:-1]), probs)
          p_wire = tbob.P_wire(decls, total_probs, revealed, found, hand_size, active_wires - player_wires)
          p_bad, p_bomb = tbob.DeTensor(probs)
          probabilities_list[-1] = p_bad.copy()
          p_bad = tbob.DeMatrix(tbob.CombineProbs(probabilities_list))
      curr_points = num_players - active_wires
      p_bomb_rand = player_bomb / (num_players * hand_size - np.sum(revealed))
      p_wire_rand = (active_wires - player_wires) / (num_players * hand_size - np.sum(revealed))
      score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
      table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
               [other_players[i], p_wire[i]*100, p_bomb[i]*100, p_bad[i]*100, score[i]] for i in range(num_players)] + [
               ["Average", p_wire_rand*100, p_bomb_rand*100, (2-is_bad)/num_players*100, np.sum(score)/num_players]]
      print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return
