import numpy as np
from random import randint
import UsefulFunctions as uf
from tabulate import tabulate
from itertools import combinations


def Cn(distribution):
  prod = 1
  for i in distribution:
    prod *= uf.Fact(i)
  return uf.Fact(sum(distribution)) / prod


def Flatten(probabilities, num_b):
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for indices in combinations(range(num_players), num_b):
    for index in indices:
      probability_line[index] += probabilities[indices]
  return probability_line


def Seperate(probabilities, num_bad, num_bom):
  num_players = len(probabilities)
  probability_bad = np.zeros([num_players]*num_bad)
  probability_bom = np.zeros([num_players]*num_bom)
  for bad_indices in combinations(range(num_players), num_bad):
    for bom_indices in combinations(range(num_players), num_bom):
        probability_bad[bad_indices] += probabilities[bad_indices + bom_indices]
        probability_bom[bom_indices] += probabilities[bad_indices + bom_indices]
  return (probability_bad, probability_bom)


def ProbDeclaration(decls, hand_size, active_wires, num_bad, num_bom):
  num_players = decls.shape[0]
  probs = np.zeros([num_players]*(num_bad + num_bom))
  for bad_set in combinations(range(num_players), num_bad):
    bad_wires = int(active_wires - np.sum(decls))
    for bad in bad_set:
      bad_wires += int(decls[bad])
    for bom_set in combinations(range(num_players), num_bom):
      combinations = 0
      bad_bom_set = tuple(set(bad_set) & set(bom_set))
      bad_nbom_set = tuple(set(bad_set) - set(bom_set))
      nbad_bom_set = tuple(set(bom_set) - set(bad_set))
      num_liars = len(bad_set) + len(bom_set) - len(bad_bom_set)
      for wires_dist in (bad_wires, num_liars):
        prob = 1
        for nbad_bom in nbad_bom_set:
          if hand_size - decls[nbad_bom] - 1 < 0:  # The bomb is not hidden in nbad_bom's hand
            probs[bad_set + bom_set] = 0
            continue  #! Should continue 3 (next bom_set)
          prob *= uf.C(wires_dist[nbad_bom] - decls[nbad_bom], hand_size - decls[nbad_bom] - 1)
        for bad_bom in bad_bom_set:
          if wires_dist[bad_bom] > decls[bad_bom]:  # bad_bom has more wires than declared
            prob *= 0
            continue  #! Should continue 2 (next wire_dist)
          else:  # bad_bom has =fewer wires than declared
            prob *= uf.C(wires_dist[bad_bom], decls[bad_bom])
        for bad_nbom in bad_nbom_set:
          if wires_dist[bad_nbom] < decls[bad_nbom]:  # bad_nbom has fewer wires than declared
            prob *= uf.C(wires_dist[bad_nbom], decls[bad_nbom])
          else:  # bad_nbom has =more wires than declared
            prob *= uf.C(wires_dist[bad_nbom] - decls[bad_nbom], hand_size - decls[bad_nbom])
        new_combinations = Cn(wires_dist)
        probs[bad_set + bom_set] += prob * new_combinations
        combinations += new_combinations
      if combinations != 0:
        probs[bad_set + bom_set] /= combinations
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs
