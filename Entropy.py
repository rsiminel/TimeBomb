import numpy as np


def H(probs):
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1/p)
  return h


def NextH(decls, probs, revealed, found, hand_size, active_wires, P_wire, ProbCut):
  num_players = decls.size
  p_wire = P_wire(decls, probs, found, hand_size, active_wires)
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
      info_wire = H(ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires))
    if p_wire[cutee] < 0.9999:
      info_not_wire = H(ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires))
    h[cutee] = p_wire[cutee] * info_wire + (1 - p_wire[cutee]) * info_not_wire
  return h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop, P_wire, ProbCut):
  if stop <= 0:
    return (H(probs), -1)
  num_players = decls.size
  p_wire = P_wire(decls, probs, found, hand_size, active_wires)
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
      (h_wire, _) = H_Min(decls, new_probs, revealed + reveal, found + find, hand_size, active_wires, stop - 1)
    if p_wire[cutee] < 0.9999:
      new_probs = ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires)
      (h_not_wire, _) = H_Min(decls, new_probs, revealed + reveal, found, hand_size, active_wires, stop - 1)
    h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_not_wire
  min_cutee = 0
  min_h = h[0]
  for cutee in range(num_players):
    if h[cutee] < min_h:
      min_cutee = cutee
      min_h = h[cutee]
  return (min_h, min_cutee)
