#%%
import numpy as np
from copy import deepcopy
from random import randint
from scipy.stats import beta
import UsefulFunctions as uf
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.special import gammaln, psi
from scipy.optimize import minimize_scalar
from sympy.utilities.iterables import multiset_permutations
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


def dirichlet_entropy(alpha):
  alpha0 = np.sum(alpha)
  log_beta = np.sum(gammaln(alpha)) - gammaln(alpha0)
  return log_beta + (alpha0 - len(alpha)) * psi(alpha0) - np.sum((alpha - 1) * psi(alpha))


def ProbsGraph(players, probs, e=0.001):
  num_players = len(players)
  x = np.linspace(0, 1, 500)
  # fig, axs = plt.subplots(num_players)
  for i in range(num_players):
    res = minimize_scalar(
      lambda c: - dirichlet_entropy(np.array([c * probs[i], c * (1 - probs[i])])),
      bounds=[1, 100], method='bounded'
    )
    print(res.x)
    # axs[i].set_title(players[i])
    # axs[i].plot(x, beta.pdf(x, res.x * probs[i] + e, res.x * (1 - probs[i]) + num_players * e))
    plt.plot(x, beta.pdf(x, res.x * probs[i] + e, res.x * (1 - probs[i]) + num_players * e))
  plt.show()
  return


def DisplayProbs(players, probs, probs_list, decls, revealed, found, hand_size, active_wires, pos_bad, num_bom):
  num_players = decls.size
  p_wire = np.zeros(num_players)
  p_bomb = np.zeros(num_players)
  comb_probs = np.zeros(num_players)
  score = np.zeros(num_players)
  p_wir_rand = 0
  p_bom_rand = 0
  p_bad_rand = 0
  for i in range(len(pos_bad)):
    _, prob_bomb = Separate(probs[i], pos_bad[i][0], num_bom)
    for bom_set in combinations(range(num_players), num_bom):
      for bom in bom_set:
        if hand_size - revealed[bom] != 0:
          p_bomb[bom] += pos_bad[i][1] * prob_bomb[bom_set] / (hand_size - revealed[bom])
    comb_probs += pos_bad[i][1] * Flatten(CombineProbs(probs_list[i]))
    total_probs = CombineNonHomoProbs(CombineProbs(probs_list[i][0:-1]), probs[i], pos_bad[i][0], num_bom)
    p_wire += pos_bad[i][1] * P_wire(decls, total_probs, revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
    curr_points = num_players - active_wires
    score += pos_bad[i][1] * ((1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points))
    p_wir_rand += pos_bad[i][1] * active_wires / (num_players * hand_size - np.sum(revealed))
    p_bom_rand += pos_bad[i][1] * num_bom / (num_players * hand_size - np.sum(revealed))
    p_bad_rand += pos_bad[i][1] * pos_bad[i][0] / num_players
  table = [["Player", "P_wire", "P_bomb", "P_bad", "Score"]] + [
            [players[i], p_wire[i]*100, p_bomb[i]*100, comb_probs[i]*100, score[i]] for i in range(num_players)] + [
            ["Average", p_wir_rand*100, p_bom_rand*100, p_bad_rand*100, np.sum(score)/num_players]]
  print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', floatfmt=(".1f", ".1f", ".1f", ".1f", ".3f")))
  ProbsGraph(players, comb_probs)
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
      for short_ord_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != bad_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
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
      for short_ord_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != bad_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
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
      wires_avg = np.zeros(num_players)
      liar_set = tuple(set(bad_set) | set(bom_set) - set(bad_set) & set(bom_set))
      for short_ord_wires_dist in combinations_with_replacement(range(bad_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != bad_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
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


def Play(players=["Alice", "Bob", "Clara", "Darryl", "Erica", "Fred"], initial_hand_size=5):
  num_players = len(players)
  num_bom = 1
  if num_players < 4:
    print("Not enough players")
    return
  elif num_players == 4:
    pos_bad = [[1, 2/5], [2, 3/5]]
  elif num_players < 7:
    pos_bad = [[2, 1.0]]
  elif num_players == 7:
    pos_bad = [[2, 3/8], [3, 5/8]]
  elif num_players == 8:
    pos_bad = [[3, 1.0]]
  else:
    print("Too many players")
    return
  hand_size = initial_hand_size
  num_wires = num_players * hand_size
  active_wires = num_players
  zeros = np.zeros(num_players)
  # Initialize probabilities
  probabilities_list = [[] for _ in range(len(pos_bad))]
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Declare your wires
    declarations = zeros.copy()
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    print("d:", declarations)
    # Calculate probabilities
    probabilities = [0 for _ in range(len(pos_bad))]
    prob_bad = [0 for _ in range(len(pos_bad))]
    for i in range(len(pos_bad)):
      probabilities[i] = ProbDeclaration(declarations, hand_size, active_wires, pos_bad[i][0], num_bom)
      prob_bad[i], _ = Separate(probabilities[i], pos_bad[i][0], num_bom)
      probabilities_list[i].append(deepcopy(prob_bad[i]))
    DisplayProbs(players, probabilities, probabilities_list, declarations, zeros, zeros, hand_size, active_wires, pos_bad, num_bom)
    # Cut wires
    found = zeros.copy()
    revealed = zeros.copy()
    for cut in range(num_players):
      print("\n Cut number", cut + 1)
      cutee_str = input("Who's wire has been cut? ")
      while cutee_str not in players:
        cutee_str = input("You must have made a typo. Who? ")
      for player in range(num_players):
        if players[player] == cutee_str:
          cutee = player
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
      probs = [0 for _ in range(len(pos_bad))]
      prob_bad = [0 for _ in range(len(pos_bad))]
      for i in range(len(pos_bad)):
        probs[i] = ProbCut(declarations, probabilities[i], revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
        prob_bad[i], _ = Separate(probs[i], pos_bad[i][0], num_bom)
        probabilities_list[i][-1] = deepcopy(prob_bad[i])
      DisplayProbs(players, probs, probabilities_list, declarations, revealed, found, hand_size, active_wires, pos_bad, num_bom)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


def PlaySubjective(other_players=["Alice", "Bob", "Clara", "Darryl", "Erica"], initial_hand_size=5):
  # Initialize game
  num_players = len(other_players)
  hand_size = initial_hand_size
  num_wires = (num_players + 1) * hand_size
  active_wires = num_players + 1
  is_bad = int(input("Are you a\n 0- good guy\n 1- bad guy"))
  while is_bad not in [0, 1]:
    is_bad = int(input("Sorry, I'm looking for a 0 or a 1 here."))
  num_bom = 1
  if num_players < 4:
    print("Not enough players")
    return
  elif num_players == 4:
    pos_bad = [[1, 2/5], [2, 3/5]]
  elif num_players < 7:
    pos_bad = [[2, 1.0]]
  elif num_players == 7:
    pos_bad = [[2, 3/8], [3, 5/8]]
  elif num_players == 8:
    pos_bad = [[3, 1.0]]
  else:
    print("Too many players")
    return
  for i in range(len(pos_bad)):
    pos_bad[i][0] -= is_bad
  zeros = np.zeros(num_players)
  probabilities_list = [[] for _ in range(len(pos_bad))]
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Wire declarations
    player_bomb = int(input("Do you\n 0- not have the bomb\n 1- have the bomb"))
    while player_bomb not in [0, 1]:
      player_bomb = int(input("Sorry, I'm looking for a 0 or a 1 here."))
    pos_bomb = num_bom - player_bomb
    player_wires = int(input("How many wires do you have?"))
    pos_wires = active_wires - player_wires
    decls = zeros.copy()
    for i in range(num_players):
      decls[i] = int(input("How many wires does " + other_players[i] + " say they have? "))
    print("d:", decls)
    # Calculate probabilities
    probabilities = [0 for _ in range(len(pos_bad))]
    prob_bad = [0 for _ in range(len(pos_bad))]
    for i in range(len(pos_bad)):
      probabilities[i] = ProbDeclaration(decls, hand_size, pos_wires, pos_bad[i][0], pos_bomb)
      prob_bad[i], _ = Separate(probabilities[i], pos_bad[i][0], pos_bomb)
      probabilities_list[i].append(deepcopy(prob_bad[i]))
    DisplayProbs(other_players, probabilities, probabilities_list, decls, zeros, zeros, hand_size, pos_wires, pos_bad, pos_bomb)
    # Cut wires
    found = zeros.copy()
    revealed = zeros.copy()
    for cut in range(num_players):
      print("\n Cut number", cut + 1)
      cutee_str = input("Who's wire has been cut? ")
      num_wires -= 1
      if cutee_str == "me":
        shown = int(input("Did you reveal\n 0- an inactive wire\n 1- an active wire\n 2- the bomb"))
        while shown not in [0, 1, 2]:
          shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here."))
        if shown == 2:
          print("The Bomb was detonated. Bad guys win!")
          return
        if shown == 1:
          active_wires -= 1
        continue
      while cutee_str not in other_players:
        cutee_str = input("You must have made a typo. Who? ")
      for player in range(num_players):
        if other_players[player] == cutee_str:
          cutee = player
      shown = int(input("Did you reveal\n 0- an inactive wire\n 1- an active wire\n 2- the bomb"))
      while shown not in [0, 1, 2]:
        shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here."))
      if shown == 2:
        print("The Bomb was detonated. Bad guys win!")
        return
      if shown == 1:
        active_wires -= 1
        found[cutee] += 1
      print("r:", revealed)
      print("f:", found)
      # Update probabilities
      probs = [0 for _ in range(len(pos_bad))]
      prob_bad = [0 for _ in range(len(pos_bad))]
      for i in range(len(pos_bad)):
        probs[i] = ProbCut(decls, probabilities[i], revealed, found, hand_size, pos_wires + np.sum(found), pos_bad[i][0], pos_bomb)
        prob_bad[i], _ = Separate(probs[i], pos_bad[i][0], pos_bomb)
        probabilities_list[i][-1] = deepcopy(prob_bad[i])
      DisplayProbs(other_players, probs, probabilities_list, decls, revealed, found, hand_size, pos_wires, pos_bad, pos_bomb)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


def PlayAuto(CutStrategy, num_players, initial_hand_size=5, verbosity=2):
  num_bom = 1
  if num_players < 4:
    print("Not enough players")
    return
  elif num_players == 4:
    num_bad = 2 - int(randint(0, 4) < 2)
    pos_bad = [[1, 2/5], [2, 3/5]]
  elif num_players < 7:
    num_bad = 2
    pos_bad = [[2, 1.0]]
  elif num_players == 7:
    num_bad = 3 - int(randint(0, 7) < 3)
    pos_bad = [[2, 3/8], [3, 5/8]]
  elif num_players == 8:
    num_bad = 3
    pos_bad = [[3, 1.0]]
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
  probabilities_list = [[] for _ in range(len(pos_bad))]
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
    probabilities = [0 for _ in range(len(pos_bad))]
    prob_bad = [0 for _ in range(len(pos_bad))]
    for i in range(len(pos_bad)):
      probabilities[i] = ProbDeclaration(declarations, hand_size, active_wires, pos_bad[i][0], num_bom)
      prob_bad[i], _ = Separate(probabilities[i], pos_bad[i][0], num_bom)
      probabilities_list[i].append(deepcopy(prob_bad[i]))
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
      DisplayProbs(players, probabilities, probabilities_list, declarations, zeros, zeros, hand_size, active_wires, pos_bad, num_bom)
    # Cut wires
    found = np.zeros(num_players)
    revealed = np.zeros(num_players)
    probs = deepcopy(probabilities)
    cutee = -1
    for cut in range(num_players):
      if verbosity > 0:
        print("Cut number", cut + 1)
      cutee = CutStrategy(declarations, probs, revealed, found, hand_size, active_wires, cutee, pos_bad, num_bom)
      randy = randint(1, hand_size - revealed[cutee])
      if bombs[cutee] == 1 and randy == hand_size - revealed[cutee]:
        if verbosity > 0:
          print("The Bomb was detonated. Bad guys win!")
        final_probs = np.zeros(num_players)
        for i in range(len(pos_bad)):
          final_probs += pos_bad[i][1] * Flatten(CombineProbs(probabilities_list[i]))
        return (0, final_probs, roles)
      elif randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      num_wires -= 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities
      probs = [0 for _ in range(len(pos_bad))]
      prob_bad = [0 for _ in range(len(pos_bad))]
      for i in range(len(pos_bad)):
        probs[i] = ProbCut(declarations, probabilities[i], revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
        prob_bad[i], _ = Separate(probs[i], pos_bad[i][0], num_bom)
        probabilities_list[i][-1] = deepcopy(prob_bad[i])
      if verbosity > 1:
        DisplayProbs(players, probs, probabilities_list, declarations, revealed, found, hand_size, active_wires, pos_bad, num_bom)
      # Test for victory
      if active_wires <= 0:
        if verbosity > 0:
          print("Good guys win!")
        final_probs = np.zeros(num_players)
        for i in range(len(pos_bad)):
          final_probs += pos_bad[i][1] * Flatten(CombineProbs(probabilities_list[i]))
        return (1, final_probs, roles)
    # Next round
    hand_size -= 1
    if verbosity > 0:
      print("\n")
  if verbosity > 0:
    print("Bad guys win!")
  final_probs = np.zeros(num_players)
  for i in range(len(pos_bad)):
    final_probs += pos_bad[i][1] * Flatten(CombineProbs(probabilities_list[i]))
  return (0, final_probs, roles)


def CutRandom(decls, probs, revealed, found, hand_size, active_wires, curr_cut, pos_bad, num_bom):
  num_players = revealed.size
  cutee = randint(0, num_players - 1)
  while revealed[cutee] >= hand_size or cutee == curr_cut:
    cutee = randint(0, num_players - 1)
  return cutee


def CutMaxScore(decls, probs, revealed, found, hand_size, active_wires, curr_cut, pos_bad, num_bom):
  num_players = decls.size
  score = np.zeros(num_players)
  for i in range(len(pos_bad)):
    p_wire = P_wire(decls, probs[i], revealed, found, hand_size, active_wires + np.sum(found), pos_bad[i][0], num_bom)
    _, prob_bomb = Separate(probs[i], pos_bad[i][0], num_bom)
    p_bomb = np.zeros(num_players)
    for j in range(num_players):
      if hand_size - revealed[j] != 0:
        p_bomb[j] = prob_bomb[j] / (hand_size - revealed[j])
    curr_points = num_players - active_wires
    score += pos_bad[i][1] * ((1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points))
  cutee = -1
  max_score = 0
  for i in range(num_players):
    if revealed[i] < hand_size and i != curr_cut:
      if score[i] > max_score:
        cutee = i
        max_score = score[i]
  return cutee


def Test(strategies, num_players, num_games, init_hand_size=5):
  win_rate = [0]  * len(strategies)
  suspicion = [0] * len(strategies)
  for strat in range(len(strategies)):
    for _ in range(num_games):
      (is_win, probs, roles) = PlayAuto(strategies[strat], num_players, init_hand_size, 0)
      win_rate[strat] += is_win
      culprits = []
      for player in range(num_players):
        if roles[player] == 1:
          culprits.append(player)
      for j in range(len(culprits)):
        suspicion[strat] += probs[culprits[j]] / len(culprits)
    win_rate[strat] /= num_games
    suspicion[strat] /= num_games
  return (win_rate, suspicion)

# %%
