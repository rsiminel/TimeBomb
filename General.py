import numpy as np
from copy import deepcopy
from random import randint
import UsefulFunctions as uf
from tabulate import tabulate
from itertools import combinations, combinations_with_replacement


def Cn(distribution):
  prod = 1
  for i in distribution:
    prod *= uf.Fact(i)
  return uf.Fact(sum(distribution)) / prod


def Flatten(probabilities):
  num_players = len(probabilities)
  num_dims = len(probabilities.shape)
  probability_line = np.zeros(num_players)
  for indices in combinations(range(num_players), num_dims):
    for index in indices:
      probability_line[index] += probabilities[indices]
  return probability_line


def Separate(probabilities, num_bad, num_bom):
  num_players = len(probabilities)
  probability_bad = np.zeros([num_players]*num_bad)
  probability_bom = np.zeros([num_players]*num_bom)
  for bad_indices in combinations(range(num_players), num_bad):
    for bom_indices in combinations(range(num_players), num_bom):
        probability_bad[bad_indices] += probabilities[bad_indices + bom_indices]
        probability_bom[bom_indices] += probabilities[bad_indices + bom_indices]
  return (probability_bad, probability_bom)


def CombineProbs(probabilities_list):
  if probabilities_list == []:
    return np.array([])
  num_tests = len(probabilities_list)
  num_players = probabilities_list[0].shape[0]
  num_dims = len(probabilities_list[0].shape)
  probabilities = np.zeros([num_players]*num_dims)
  for bad_set in combinations(range(num_players), num_dims):
    probabilities[bad_set] = 1
    for test in range(num_tests):
      probabilities[bad_set] *= probabilities_list[test][bad_set]
  if np.sum(probabilities) != 0:
    probabilities /= np.sum(probabilities)
  return probabilities


def CombineNonHomoProbs(prob_bad, probs, num_bad, num_bom):
  if prob_bad.size == 0:
    return probs
  num_players = prob_bad[0].size
  new_probs = deepcopy(probs)
  for bad_set in combinations(range(num_players), num_bad):
    for bom_set in combinations(range(num_players), num_bom):
      new_probs[bad_set + bom_set] *= prob_bad[bad_set]
  if np.sum(new_probs) != 0:
    new_probs /= np.sum(new_probs)
  return new_probs


def DisplayProbs(players, probs, probs_list, decls, revealed, found, hand_size, active_wires, num_bad, num_bom):
  num_players = decls.size
  _, prob_bomb = Separate(probs, num_bad, num_bom)
  p_bomb = np.zeros(num_players)
  for bom_set in combinations(range(num_players), num_bom):
    for bom in bom_set:
      if hand_size - revealed[bom] != 0:
        p_bomb[bom] += prob_bomb[bom_set] / (hand_size - revealed[bom])
  comb_probs = Flatten(CombineProbs(probs_list))
  total_probs = CombineNonHomoProbs(CombineProbs(probs_list[0:-1]), probs, num_bad, num_bom)
  p_wire = P_wire(decls, total_probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  curr_points = num_players - active_wires
  score = (1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points)
  p_wire_rand = active_wires / (num_players * hand_size - np.sum(revealed))
  p_bomb_rand = 1 / (num_players * hand_size - np.sum(revealed))
  table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
            [players[i], p_wire[i]*100, p_bomb[i]*100, comb_probs[i]*100, score[i]] for i in range(num_players)] + [
            ["Average", p_wire_rand*100, p_bomb_rand*100, 2/num_players*100, np.sum(score)/num_players]]
  print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
  return


def ProbDeclaration(decls, hand_size, active_wires, num_bad, num_bom):
  num_players = decls.shape[0]
  probs = np.zeros([num_players]*(num_bad + num_bom))
  for bad_set in combinations(range(num_players), num_bad):
    bad_wires = int(active_wires - np.sum(decls))
    for bad in bad_set:
      bad_wires += int(decls[bad])
    for bom_set in combinations(range(num_players), num_bom):
      bad_bom_set = tuple(set(bad_set) & set(bom_set))
      bad_nbom_set = tuple(set(bad_set) - set(bom_set))
      nbad_bom_set = tuple(set(bom_set) - set(bad_set))
      liars_impossible = False
      for nbad_bom in nbad_bom_set:
        if hand_size - decls[nbad_bom] - 1 < 0:  # The bomb is not hidden in nbad_bom's hand
          liars_impossible = True
          break
      if liars_impossible:
        continue
      combs = 0
      liar_set = bad_bom_set + bad_nbom_set + nbad_bom_set
      for short_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_wires_dist) != bad_wires:
          continue
        wires_dist = np.zeros(num_players)
        for wire in range(len(short_wires_dist)):
          wires_dist[liar_set[wire]] = short_wires_dist[wire]
        prob = 1
        wires_impossible = False
        for bad_bom in bad_bom_set:
          if wires_dist[bad_bom] > decls[bad_bom]:  # bad_bom has more wires than declared
            wires_impossible = True
            break
          else:  # bad_bom has =fewer wires than declared
            prob *= uf.C(wires_dist[bad_bom], decls[bad_bom])
        if wires_impossible:
          continue
        for bad_nbom in bad_nbom_set:
          if wires_dist[bad_nbom] < decls[bad_nbom]:  # bad_nbom has fewer wires than declared
            prob *= uf.C(wires_dist[bad_nbom], decls[bad_nbom])
          else:  # bad_nbom has =more wires than declared
            prob *= uf.C(wires_dist[bad_nbom] - decls[bad_nbom], hand_size - decls[bad_nbom])
        for nbad_bom in nbad_bom_set:
          wires_dist[nbad_bom] += decls[nbad_bom]
          prob *= uf.C(wires_dist[nbad_bom] - decls[nbad_bom], hand_size - decls[nbad_bom] - 1)
        new_combs = Cn(wires_dist)
        probs[bad_set + bom_set] += prob * new_combs
        combs += new_combs
      if combs != 0:
        probs[bad_set + bom_set] /= combs
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires, num_bad, num_bom):
  num_players = decls.size
  lklhds = np.zeros([num_players]*(num_bad + num_bom))
  marginal = 0
  for bad_set in combinations(range(num_players), num_bad):
    bad_wires = int(active_wires - np.sum(decls))
    for bad in bad_set:
      bad_wires += int(decls[bad])
    for bom_set in combinations(range(num_players), num_bom):
      if prior[bad_set + bom_set] == 1:  # Bad guys and the bomb found
        return prior
      liars_impossible = False
      for bom in bom_set:
        if revealed[bom] >= hand_size:  # All of bomber's hand is not bomb
          liars_impossible = True
          break
        if bom in bad_set and hand_size + decls[bom] < bad_wires:
            liars_impossible = True
            break
      if liars_impossible:
        continue
      combs = 0
      liar_set = tuple(set(bad_set) | set(bom_set) - set(bad_set) & set(bom_set))
      for short_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_wires_dist) != bad_wires:
          continue
        wires_dist = np.zeros(num_players)
        for wire in range(len(short_wires_dist)):
          wires_dist[liar_set[wire]] = short_wires_dist[wire]
        lklhd = 1
        for player in range(num_players):
          if player in bom_set and player in bad_set:
            lklhd *= uf.Lklhd(hand_size - 1, wires_dist[player], revealed[player], found[player])
          elif player in bom_set:
            lklhd *= uf.Lklhd(hand_size - 1, wires_dist[player] + decls[player], revealed[bom], found[bom])
          elif player in bad_set:
            lklhd *= uf.Lklhd(hand_size, wires_dist[player], revealed[player], found[player])
          else:
            lklhd *= uf.Lklhd(hand_size, decls[player], revealed[player], found[player])
        new_combs = Cn(wires_dist)
        combs += new_combs
        lklhds[bad_set + bom_set] += new_combs * lklhd
      if combs != 0:
        lklhds[bad_set + bom_set] /= combs
      marginal += prior[bad_set + bom_set] * lklhds[bad_set + bom_set]
  if marginal == 0:
    return prior
  return prior * lklhds / marginal


def P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom):
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for bad_set in combinations(range(num_players), num_bad):
    bad_wires = int(active_wires - np.sum(decls))
    for bad in bad_set:
      bad_wires += int(decls[bad])
    for bom_set in combinations(range(num_players), num_bom):
      combs = 0
      wires_avg = np.zeros([num_players])
      liar_set = tuple(set(bad_set) | set(bom_set) - set(bad_set) & set(bom_set))
      for short_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_wires_dist) != bad_wires:
          continue
        wires_dist = np.zeros(num_players)
        for wire in range(len(short_wires_dist)):
          wires_dist[liar_set[wire]] = short_wires_dist[wire]
        new_combs = Cn(wires_dist)
        wires_impossible = False
        for player in liar_set:
          if (
            (wires_dist[player] > hand_size - revealed[player] - int(player in bom_set)) or
            (player in bom_set and (
              (player in bad_set and wires_dist[player] + found[player] > decls[player]) or
              (player not in bad_set and wires_dist[player] + found[player] < decls[player])))
          ):
            wires_impossible = True
            break
        if wires_impossible:
          continue
        wires_avg += wires_dist * new_combs
        combs += new_combs
      if combs != 0:
        wires_avg /= combs
      p_wire += wires_avg * probs[bad_set + bom_set]
  (p_bad, _) = Separate(probs, num_bad, num_bom)
  lin_probs = Flatten(p_bad)
  for good in range(num_players):
    p_wire[good] += (1 - lin_probs[good]) * (decls[good] - found[good])
    if hand_size - revealed[good] > 0:
      p_wire[good] /= hand_size - revealed[good]
  return p_wire


def PlayAuto(CutStrategy, num_players, initial_hand_size=5, verbosity=2):
  num_bom = 1
  if num_players < 4:
    print("Not enough players")
    return
  elif num_players == 4:
    num_bad = 2 - int(randint(0, 4) < 2)
  elif num_players < 7:
    num_bad = 2
  elif num_players == 7:
    num_bad = 3 - int(randint(0, 7) < 3)
  elif num_players == 8:
    num_bad = 3
  else:
    print("Too many players")
    return
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  zeros = np.zeros(num_players)
  # Distributing roles
  roles = zeros.copy()
  evil = 0
  while evil < num_bad:
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
    bombs = zeros.copy()
    bom = 0
    while bom < num_bom:
      randy = randint(0, num_players - 1)
      if bombs[randy] == 0 and wires[randy] < hand_size:
        bombs[randy] = 1
        bom += 1
    if verbosity > 0:
      print("w:", wires)
      print("b:", bombs)
    # Declare your wires
    declarations = wires.copy()
    for player in range(num_players):
      maxx = min(hand_size, active_wires)
      if roles[player] == 1:
        if bombs[player] == 1:
          declarations[player] = randint(wires[player], maxx)
        else:
          declarations[player] = randint(0, maxx)
      elif bombs[player] == 1:
        declarations[player] = randint(0, wires[player])
    if verbosity > 0:
      print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires, num_bad, num_bom)
    prob_bad, _ = Separate(probabilities, num_bad, num_bom)
    probabilities_list.append(deepcopy(prob_bad))
    if verbosity > 1:
      players = []
      for player in range(num_players):
        if roles[player] == 1:
          if bombs[player] == 1:
            players += ["liar - bomb"]
          else:
            players += ["liar"]
        elif bombs[player] == 1:
          players += ["good - bomb"]
        else:
          players += ["good"]
      DisplayProbs(players, probabilities, probabilities_list, declarations, zeros, zeros, hand_size, active_wires, num_bad, num_bom)
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    probs = deepcopy(probabilities)
    cutee = -1
    for player in range(num_players):
      if verbosity > 0:
        print("Cut number", player + 1)
      cutee = CutStrategy(declarations, probs, revealed, found, hand_size, active_wires, cutee, num_bad, num_bom)
      randy = randint(1, hand_size - revealed[cutee])
      if bombs[cutee] == 1 and randy == hand_size - revealed[cutee]:
        if verbosity > 0:
          print("The Bomb was detonated. Bad guys win!")
        return (0, Flatten(CombineProbs(probabilities_list)), roles)
      elif randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires, num_bad, num_bom)
      prob_bad, _ = Separate(probs, num_bad, num_bom)
      probabilities_list[-1] = deepcopy(prob_bad)
      if verbosity > 1:
        DisplayProbs(players, probs, probabilities_list, declarations, revealed, found, hand_size, active_wires, num_bad, num_bom)
      # Test for victory
      if active_wires <= 0:
        if verbosity > 0:
          print("Good guys win!")
        return (1, Flatten(CombineProbs(probabilities_list)), roles)
    # Next round
    hand_size -= 1
    if verbosity > 0:
      print("\n")
  if verbosity > 0:
    print("Bad guys win!")
  return (0, Flatten(CombineProbs(probabilities_list)), roles)


def CutMaxScore(decls, probs, revealed, found, hand_size, active_wires, curr_cut, num_bad, num_bom):
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  _, prob_bomb = Separate(probs, num_bad, num_bom)
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
