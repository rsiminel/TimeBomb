from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from copy import deepcopy
from scipy.special import factorial, comb
from sympy.utilities.iterables import multiset_permutations
from itertools import combinations, combinations_with_replacement


app = Flask(__name__)
CORS(app)


def Lklhd(n, m, k, p):
  c = comb(n, m)
  return comb(k, p) * comb(n - k, m - p) / c if c else 0


def Cn(distribution):
  return factorial(np.sum(distribution)) / np.prod(factorial(distribution))


def Flatten(probabilities):
  num_players = len(probabilities)
  num_dims = len(probabilities.shape)
  probability_line = np.zeros(num_players)
  for indices in combinations(range(num_players), num_dims):
    probability_line[list(indices)] += probabilities[indices]
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


def ProbDeclaration(decls, hand_size, active_wires, num_evil, num_bomb):
  num_players = decls.shape[0]
  probs = np.zeros([num_players]*(num_evil + num_bomb))
  for evil_set in combinations(range(num_players), num_evil):
    evil_wires = int(active_wires - np.sum(decls) + np.sum(decls[list(evil_set)]))
    for bomb_set in combinations(range(num_players), num_bomb):
      evil_bomb_set = tuple(set(evil_set) & set(bomb_set))
      evil_no_bomb_set = tuple(set(evil_set) - set(bomb_set))
      good_bomb_set = tuple(set(bomb_set) - set(evil_set))
      if np.any(hand_size - decls[list(good_bomb_set)] - 1 < 0):  # The bomb is not hidden in good_bomb's hand
        continue
      combs = 0
      liar_set = evil_bomb_set + evil_no_bomb_set + good_bomb_set
      for short_ord_wires_dist in combinations_with_replacement(range(evil_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != evil_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
          wires_dist = decls.copy()
          for wire in range(len(short_wires_dist)):
            if liar_set[wire] in good_bomb_set:
              wires_dist[liar_set[wire]] += short_wires_dist[wire]
            else:
              wires_dist[liar_set[wire]] = short_wires_dist[wire]
          prob = 1
          wires_impossible = False
          for evil_bomb in evil_bomb_set:
            if wires_dist[evil_bomb] > decls[evil_bomb]:  # evil_bomb has more wires than declared
              wires_impossible = True
              break
            else:  # evil_bomb has =fewer wires than declared
              prob *= comb(decls[evil_bomb], wires_dist[evil_bomb])
          if wires_impossible:
            continue
          for evil_no_bomb in evil_no_bomb_set:
            if wires_dist[evil_no_bomb] < decls[evil_no_bomb]:  # evil_no_bomb has fewer wires than declared
              prob *= comb(decls[evil_no_bomb], wires_dist[evil_no_bomb])
            else:  # evil_no_bomb has =more wires than declared
              prob *= comb(hand_size - decls[evil_no_bomb], wires_dist[evil_no_bomb] - decls[evil_no_bomb])
          for good_bomb in good_bomb_set:
            wires_dist[good_bomb] += decls[good_bomb]
            prob *= comb(hand_size - decls[good_bomb] - 1, wires_dist[good_bomb] - decls[good_bomb])
          new_combs = Cn(wires_dist)
          probs[evil_set + bomb_set] += prob * new_combs
          combs += new_combs
      if combs != 0:
        probs[evil_set + bomb_set] /= combs
  if np.sum(probs) != 0:
    probs /= np.sum(probs)
  return probs


def ProbCut(decls, prior, revealed, found, hand_size, active_wires, num_evil, num_bomb):
  num_players = decls.size
  lklhds = np.zeros([num_players]*(num_evil + num_bomb))
  marginal = 0
  for evil_set in combinations(range(num_players), num_evil):
    evil_wires = int(active_wires - np.sum(decls) + np.sum(decls[list(evil_set)]))
    for bomb_set in combinations(range(num_players), num_bomb):
      if prior[evil_set + bomb_set] == 1:  # Bad guys and the bomb found
        return prior
      liars_impossible = False
      for bomb in bomb_set:
        if revealed[bomb] >= hand_size:  # All of bomber's hand is not bomb
          liars_impossible = True
          break
        if bomb in evil_set and hand_size + decls[bomb] < evil_wires:
          liars_impossible = True
          break
      if liars_impossible:
        continue
      combs = 0
      liar_set = tuple(set(evil_set) | set(bomb_set) - set(evil_set) & set(bomb_set))
      for short_ord_wires_dist in combinations_with_replacement(range(evil_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != evil_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
          wires_dist = decls.copy()
          for wire in range(len(short_wires_dist)):
            if liar_set[wire] in bomb_set:
              wires_dist[liar_set[wire]] += short_wires_dist[wire]
            if liar_set[wire] in evil_set:
              wires_dist[liar_set[wire]] = short_wires_dist[wire]
          lklhd = 1
          for player in range(num_players):
            if player in bomb_set:
              lklhd *= Lklhd(hand_size - 1, wires_dist[player], revealed[player], found[player])
            else:
              lklhd *= Lklhd(hand_size, wires_dist[player], revealed[player], found[player])
          new_combs = Cn(wires_dist)
          combs += new_combs
          lklhds[evil_set + bomb_set] += new_combs * lklhd
      if combs != 0:
        lklhds[evil_set + bomb_set] /= combs
      marginal += prior[evil_set + bomb_set] * lklhds[evil_set + bomb_set]
  if marginal == 0:
    return prior
  return prior * lklhds / marginal


def P_wire(decls, probs, revealed, found, hand_size, active_wires, num_evil, num_bomb):
  num_players = decls.size
  p_wire = np.zeros(num_players)
  for evil_set in combinations(range(num_players), num_evil):
    evil_wires = int(active_wires - np.sum(decls) + np.sum(decls[list(evil_set)]))
    for bomb_set in combinations(range(num_players), num_bomb):
      combs = 0
      wires_avg = np.zeros(num_players)
      liar_set = tuple(set(evil_set) | set(bomb_set) - set(evil_set) & set(bomb_set))
      for short_ord_wires_dist in combinations_with_replacement(range(evil_wires + 1), len(liar_set)):
        if sum(short_ord_wires_dist) != evil_wires:
          continue
        for short_wires_dist in multiset_permutations(short_ord_wires_dist):
          wires_dist = decls.copy()
          for wire in range(len(short_wires_dist)):
            if liar_set[wire] in bomb_set:
              wires_dist[liar_set[wire]] += short_wires_dist[wire]
            if liar_set[wire] in evil_set:
              wires_dist[liar_set[wire]] = short_wires_dist[wire]
          new_combs = Cn(wires_dist)
          wires_impossible = False
          for player in liar_set:
            if (
              (wires_dist[player] > hand_size - revealed[player] - int(player in bomb_set)) or
              (player in bomb_set and (
                (player in evil_set and wires_dist[player] + found[player] > decls[player]) or
                (player not in evil_set and wires_dist[player] + found[player] < decls[player])))
            ):
              wires_impossible = True
              break
          if wires_impossible:
            continue
          wires_avg += wires_dist * new_combs
          combs += new_combs
      if combs != 0:
        wires_avg /= combs
      p_wire += wires_avg * probs[evil_set + bomb_set]
  for player in range(num_players):
    if hand_size - revealed[player] > 0:
      p_wire[player] /= hand_size - revealed[player]
  return p_wire


@app.route('/declaration', methods=['POST'])
def Declaration():
  try:
    inputs = request.get_json()
    probs_list = [[np.array(prob_evil) for prob_evil in pos] for pos in inputs.get('probs_list', [])]
    decls = np.array(inputs['decls'])
    hand_size = inputs['hand_size']
    active_wires = inputs['active_wires']
    pos_evil = inputs['pos_evil']
    num_bomb = inputs['num_bomb']
    num_players = len(decls)
    zeros = np.zeros(num_players)
    probs = [0 for _ in range(len(pos_evil))]
    p_evil = zeros.copy()
    p_bomb = zeros.copy()
    p_wire = zeros.copy()
    score = zeros.copy()
    for pos in range(len(pos_evil)):
      probs[pos] = ProbDeclaration(decls, hand_size, active_wires, pos_evil[pos][0], num_bomb)
      prob_evil, prob_bomb = Separate(probs[pos], pos_evil[pos][0], num_bomb)
      probs_list[pos].append(deepcopy(prob_evil))
      for bom_set in combinations(range(num_players), num_bomb):
        for bom in bom_set:
          p_bomb[bom] += pos_evil[pos][1] * prob_bomb[bom_set] / hand_size
      p_evil += pos_evil[pos][1] * Flatten(CombineProbs(probs_list[pos]))
      total_probs = CombineNonHomoProbs(CombineProbs(probs_list[pos][0:-1]), probs[pos], pos_evil[pos][0], num_bomb)
      p_wire += pos_evil[pos][1] * P_wire(decls, total_probs, zeros, zeros, hand_size, active_wires, pos_evil[pos][0], num_bomb)
      curr_points = num_players - active_wires
      score += pos_evil[pos][1] * ((1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points))
    return jsonify({
      'wire': p_wire.tolist(),
      'bomb': p_bomb.tolist(),
      'evil': p_evil.tolist(),
      'score': score.tolist(),
      'probs_list': [[prob_evil.tolist() for prob_evil in pos] for pos in probs_list],
      'prior': [prob.tolist() for prob in probs]
    }), 200
  except Exception as e:
    print(e)
    return jsonify({'error': str(e)}), 500


@app.route('/cut', methods=['POST'])
def Cut():
  try:
    inputs = request.get_json()
    probs_list = [[np.array(prob_evil) for prob_evil in pos] for pos in inputs.get('probs_list', [])]
    prior = np.array(inputs['prior'])
    decls = np.array(inputs['decls'])
    revealed = np.array(inputs['revealed'])
    found = np.array(inputs['found'])
    hand_size = inputs['hand_size']
    active_wires = inputs['active_wires']
    pos_evil = inputs['pos_evil']
    num_bomb = inputs['num_bomb']
    num_players = len(decls)
    probs = [0 for _ in range(len(pos_evil))]
    p_evil = np.zeros(num_players)
    p_bomb = np.zeros(num_players)
    p_wire = np.zeros(num_players)
    score = np.zeros(num_players)
    for pos in range(len(pos_evil)):
      probs[pos] = ProbCut(decls, prior[pos], revealed, found, hand_size, active_wires, pos_evil[pos][0], num_bomb)
      prob_evil, prob_bomb = Separate(probs[pos], pos_evil[pos][0], num_bomb)
      probs_list[pos][-1] = deepcopy(prob_evil)
      for bom_set in combinations(range(num_players), num_bomb):
        for bom in bom_set:
          if hand_size - revealed[bom] != 0:
            p_bomb[bom] += pos_evil[pos][1] * prob_bomb[bom_set] / (hand_size - revealed[bom])
      p_evil += pos_evil[pos][1] * Flatten(CombineProbs(probs_list[pos]))
      total_probs = CombineNonHomoProbs(CombineProbs(probs_list[pos][0:-1]), probs[pos], pos_evil[pos][0], num_bomb)
      p_wire += pos_evil[pos][1] * P_wire(decls, total_probs, revealed, found, hand_size, active_wires + np.sum(found), pos_evil[pos][0], num_bomb)
      curr_points = num_players - active_wires
      score += pos_evil[pos][1] * ((1 - p_bomb) * (p_wire * (curr_points + 1) + (1 - p_wire) * curr_points))
    return jsonify({
      'wire': p_wire.tolist(),
      'bomb': p_bomb.tolist(),
      'evil': p_evil.tolist(),
      'score': score.tolist(),
      'probs_list': [[prob_evil.tolist() for prob_evil in pos] for pos in probs_list]
    }), 200
  except Exception as e:
    print(e)
    return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
