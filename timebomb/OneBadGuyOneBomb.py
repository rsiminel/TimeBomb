# -*- coding: utf-8 -*-
"""One bad guy, one bomb (B=1, M=1).

The belief state is the full N x N matrix ``probs[b][h]`` = P(player b is bad and
player h holds the bomb), with the diagonal ``b == h`` allowed (the bad guy may be
dealt his own bomb). The model is the uniform-lie bomb model of docs/model.md §2 and
§3.8: a player declares truthfully iff good *and* bomb-free; everyone else (any bad
guy, or a good guy holding the bomb) declares uniformly over {0..H}. The bomb occupies
one card slot, so a bomb hand has H-1 wire-able slots, and live inference conditions on
"no bomb drawn yet" (cutting the bomb ends the game, bad guys win).

P(bad) is the row marginal and **accumulates** across rounds (via ``CombineProbs``);
P(bomb) is the column marginal, read from the **current round only** -- the bomb is
re-dealt every round, so it must never be combined across rounds (§3.8.3).
"""

# Imports
import numpy as np
from random import randint
import UsefulFunctions as uf


def PlayAuto(num_players=4, initial_hand_size=5, verbosity=2):
  """Simulate one full game with random play, tracking the belief state.

  Assigns one random bad guy. Each round deals the bomb to a random hand and the
  active wires among the remaining slots, generates declarations under the uniform-lie
  bomb model (good and bomb-free => truthful; bad or bomb-holder => uniform on {0..H}),
  then cuts cards at random until the good guys find every active wire, the bomb is
  cut, or time runs out. Returns ``(good_guys_won, final_probs, roles)``:
  ``good_guys_won`` is 1/0, ``final_probs`` is the combined P(bad) vector (the bomb
  marginalised out before combining rounds, §3.8.3), ``roles`` marks the bad guy.

  ``verbosity``: 0 silent, 1 prints game events, 2 also prints the belief marginals.
  """
  hand_size = initial_hand_size
  active_wires = num_players
  # Distributing roles (fixed for the whole game)
  roles = np.zeros(num_players)
  roles[randint(0, num_players - 1)] = 1
  if verbosity > 0:
    print("Roles:", roles)
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    if verbosity > 0:
      print("Round ", initial_hand_size - hand_size + 1)
    # Deal the bomb to a random hand, then the wires among the remaining slots
    # (the bomb hand holds at most H-1 wires).
    bomb = np.zeros(num_players, dtype=int)
    bomb[randint(0, num_players - 1)] = 1
    capacity = np.full(num_players, hand_size, dtype=int) - bomb
    wires = np.zeros(num_players, dtype=int)
    given = 0
    while given < active_wires:
      c = randint(0, num_players - 1)
      if wires[c] < capacity[c]:
        wires[c] += 1
        given += 1
    if verbosity > 0:
      print("w:", wires)
      print("b:", bomb)
    # Declare your wires: truthful iff good and bomb-free, else uniform on {0..H}.
    declarations = wires.copy()
    for i in range(num_players):
      if roles[i] == 1 or bomb[i] == 1:
        declarations[i] = randint(0, hand_size)
    if verbosity > 0:
      print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    prob_bad, prob_bomb = DeMatrix(probabilities)
    probabilities_list.append(prob_bad.copy())
    if verbosity > 1:
      print("  P(bad): ", CombineProbs(probabilities_list))
      print("  P(bomb):", prob_bomb)
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
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
        return (0, CombineProbs(probabilities_list), roles)
      if randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities (Bayes on the round's declaration prior with the cuts so far)
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      prob_bad, prob_bomb = DeMatrix(probs)
      probabilities_list[-1] = prob_bad.copy()
      if verbosity > 1:
        print("  P(bad): ", CombineProbs(probabilities_list))
        print("  P(bomb):", prob_bomb)
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
    print("Out of time. Bad guys win!")
  return (0, CombineProbs(probabilities_list), roles)


def Play(players=["Alice", "Bob", "Clara", "Darryl"], initial_hand_size=5):
  """Interactive assistant for a real game: prompts for declarations and cut results
  at the terminal and prints the updated belief (P(bad), P(bomb)) after each event.
  """
  num_players = len(players)
  hand_size = initial_hand_size
  active_wires = num_players
  # Initialize probabilities
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    print("\n\n Round ", initial_hand_size - hand_size + 1)
    # Declare your wires
    declarations = np.zeros(num_players, dtype=int)
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    print("d:", declarations)
    # Calculate probabilities
    probabilities = ProbDeclaration(declarations, hand_size, active_wires)
    prob_bad, prob_bomb = DeMatrix(probabilities)
    probabilities_list.append(prob_bad.copy())
    print(" P(bad): ", CombineProbs(probabilities_list))
    print(" P(bomb):", prob_bomb)
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    for i in range(num_players):
      print("\n Cut number", i + 1)
      cutee_str = input("Whose wire has been cut? ")
      while cutee_str not in players:
        cutee_str = input("You must have made a typo. Who? ")
      cutee = players.index(cutee_str)
      revealed[cutee] += 1
      shown = int(input("Did you reveal:\n 0- an inactive wire\n 1- an active wire\n 2- the bomb\n"))
      while shown not in [0, 1, 2]:
        shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here. "))
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
      prob_bad, prob_bomb = DeMatrix(probs)
      probabilities_list[-1] = prob_bad.copy()
      print(" P(bad): ", CombineProbs(probabilities_list))
      print(" P(bomb):", prob_bomb)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


def DeMatrix(probabilities):
  """Marginalise the N x N config matrix into its two readouts (§3.8.3):
  ``P(bad = b) = Σ_h probs[b][h]`` (row sums) and ``P(bomb = h) = Σ_b probs[b][h]``
  (column sums). Returns ``(prob_bad, prob_bomb)``.
  """
  probabilities = np.asarray(probabilities)
  return (probabilities.sum(axis=1), probabilities.sum(axis=0))


def CombineProbs(probabilities_list):
  """Combine per-round P(bad) vectors into one accumulated belief.

  Roles are fixed for the game while wires and the bomb are re-dealt each round, so
  the per-round P(bad) marginals are conditionally independent evidence about the same
  fixed bad guy: multiply them elementwise and renormalise. The bomb marginal is
  per-round and is **never** passed in here (§3.8.3).
  """
  if probabilities_list == []:
    return np.array([])
  probabilities = np.ones(len(probabilities_list[0]))
  for round_probs in probabilities_list:
    probabilities *= round_probs
  total = np.sum(probabilities)
  if total == 0:
    return np.full(len(probabilities), 1 / len(probabilities))
  return probabilities / total


def ProbDeclaration(decls, hand_size, active_wires):
  """Prior P(player b bad, player h has the bomb) from declarations alone (§3.8.1).

  Marking the config ``(b, h)`` pins every truthful hand ``j not in {b, h}`` to its
  declared count and forces the liar/bomb hands' free wire total. Under a uniform deal
  the prior is the multivariate-hypergeometric probability of that forced deal; the
  liars' own declarations are constant factors (uniform lie) that cancel in
  normalisation. With ``excess = sum(decls) - active_wires`` the closed form is

    b != h:  P(b, h) ∝ C(2H-1, decls[b]+decls[h]-excess) / (C(H, decls[b])·C(H, decls[h]))
    b == h:  P(b, b) ∝ C(H-1,  decls[b]-excess)          /  C(H, decls[b])

  The bomb eats one wire slot, so two distinct liar hands offer ``2H-1`` free slots for
  the free wires and the self-bomb case offers ``H-1`` (``C(n, k) = 0`` for ``k < 0``
  or ``k > n``). Falls back to the uniform N x N prior if every config has zero weight
  (degeneracy, §3.3). Returns an N x N matrix summing to 1.
  """
  num_players = decls.size
  H = int(hand_size)
  decls = decls.astype(int)
  excess = int(np.sum(decls)) - int(active_wires)
  probs = np.zeros([num_players, num_players])
  for bad in range(num_players):
    for bom in range(num_players):
      if bad == bom:  # The bad guy holds his own bomb: one free hand, H-1 slots
        denom = uf.C(decls[bad], H)
        t = decls[bad] - excess
        if denom != 0 and 0 <= t <= H - 1:
          probs[bad][bom] = uf.C(t, H - 1) / denom
      else:  # A good guy holds the bomb: two free hands offering 2H-1 slots
        denom = uf.C(decls[bad], H) * uf.C(decls[bom], H)
        t = decls[bad] + decls[bom] - excess
        if denom != 0 and 0 <= t <= 2 * H - 1:
          probs[bad][bom] = uf.C(t, 2 * H - 1) / denom
  total = np.sum(probs)
  if total == 0:  # Declarations impossible under the model: fall back to uniform
    return np.full([num_players, num_players], 1 / num_players**2)
  return probs / total


def L_bomb_hand(found_h, revealed_h, bomb_wires, hand_size):
  """Cut likelihood for the bomb hand: P(find ``found_h`` wires AND draw no bomb in
  ``revealed_h`` cuts) given the hand holds ``bomb_wires`` wires, one bomb, and
  ``H-1-bomb_wires`` blanks -- the must-not-draw hypergeometric of model.md §3.8.2:

    C(bomb_wires, found_h) · C(H-1-bomb_wires, revealed_h-found_h) / C(H, revealed_h)

  Returns 0.0 on an infeasible wire count or an empty denominator.
  """
  H = int(hand_size)
  if not (0 <= bomb_wires <= H - 1):
    return 0.0
  denom = uf.C(revealed_h, H)
  if denom == 0:
    return 0.0
  return uf.C(found_h, bomb_wires) * uf.C(revealed_h - found_h, H - 1 - bomb_wires) / denom


def L_config(decls, revealed, found, hand_size, active_wires, bad, bom):
  """Likelihood of the cut observation under config ``(bad, bom)`` (model.md §3.8.2).

  Truthful hands (pinned to their declared count, no bomb) contribute a plain
  hypergeometric each. The free wire total ``t_free`` of the liar/bomb hands is split
  under the §3.4.1 uniform-placement law: the bad hand draws an ordinary
  hypergeometric and the bomb hand the must-not-draw term ``L_bomb_hand``. When
  ``bad == bom`` there is a single free hand (the split is deterministic). The split is
  summed explicitly; whether it collapses to a closed form is left open -- this is the
  honest, directly-checkable form.
  """
  num_players = decls.size
  H = int(hand_size)
  # Truthful hands: pinned to their declared count, no bomb.
  like = 1.0
  for j in range(num_players):
    if j != bad and j != bom:
      like *= uf.Lklhd(H, decls[j], revealed[j], found[j])
  if bad == bom:
    truthful = int(np.sum(decls)) - decls[bad]
    t_free = int(active_wires) + int(np.sum(found)) - truthful
    return like * L_bomb_hand(found[bad], revealed[bad], t_free, H)
  truthful = int(np.sum(decls)) - decls[bad] - decls[bom]
  t_free = int(active_wires) + int(np.sum(found)) - truthful
  if not (0 <= t_free <= 2 * H - 1):
    return 0.0
  denom = uf.C(t_free, 2 * H - 1)
  if denom == 0:
    return 0.0
  split = 0.0
  for bad_wires in range(t_free + 1):
    bom_wires = t_free - bad_wires
    place = uf.C(bad_wires, H) * uf.C(bom_wires, H - 1)  # 0 when out of range
    if place == 0:
      continue
    split += (place * uf.Lklhd(H, bad_wires, revealed[bad], found[bad])
              * L_bomb_hand(found[bom], revealed[bom], bom_wires, H))
  return like * split / denom


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  """Bayesian posterior over the (bad, bomb) config after a cut (model.md §3.8.2).

  The N x N configs ``(b, h)`` are mutually exclusive and exhaustive. Conditioned on a
  config, the observation factorises into the truthful hands' hypergeometrics and the
  liar/bomb hands' split-marginalised likelihood (``L_config``), all conditioned on
  "no bomb drawn yet". Bayes' theorem then gives

    posterior(b, h) = prior(b, h) · L(b, h) / Σ prior · L

  Returns the prior unchanged on a zero marginal (impossible observation) or once a
  config is already certain. Inputs mirror the no-bomb variant; ``active_wires`` is the
  post-cut remaining safe-wire total. Returns an N x N matrix summing to 1.
  """
  num_players = decls.size
  decls = decls.astype(int)
  lklhd = np.zeros([num_players, num_players])
  marginal = 0.0
  for bad in range(num_players):
    for bom in range(num_players):
      if prior[bad][bom] == 1:  # config already certain; nothing to update
        return prior
      lklhd[bad][bom] = L_config(decls, revealed, found, hand_size, active_wires, bad, bom)
      marginal += prior[bad][bom] * lklhd[bad][bom]
  if marginal == 0:  # Observation impossible under every config: keep the prior
    return prior
  posterior = prior.copy()
  for bad in range(num_players):
    for bom in range(num_players):
      posterior[bad][bom] *= lklhd[bad][bom] / marginal
  return posterior


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  """Probability that cutting one of player i's face-down cards reveals a wire.

  Mix over the config posterior ``probs[b][h]``. In each config a player is either
  truthful (holds ``decls[i] - found[i]`` remaining wires), the bad hand, or the bomb
  hand; for the two free hands the expected remaining wires are taken over the §3.4.1
  split posterior given the observation. Each contribution is divided by the remaining
  face-down cards ``H - revealed[i]`` (which still counts the bomb card in the bomb
  hand) and is gated to a feasible wire count in ``[0, H - revealed[i]]`` -- an
  impossible hypothesis contributes nothing. Players with no cards left score 0.
  """
  num_players = decls.size
  H = int(hand_size)
  decls = decls.astype(int)
  sum_found = int(np.sum(found))
  p_wire = np.zeros(num_players)
  for bad in range(num_players):
    for bom in range(num_players):
      p = probs[bad][bom]
      if p == 0:
        continue
      remaining = np.full(num_players, np.nan)  # nan = skip this hand for this config
      for j in range(num_players):  # Truthful hands: pinned to their declaration
        if j != bad and j != bom:
          remaining[j] = decls[j] - found[j]
      if bad == bom:
        t_free = int(active_wires) + sum_found - (int(np.sum(decls)) - decls[bad])
        if 0 <= t_free <= H - 1:
          remaining[bad] = t_free - found[bad]
      else:
        t_free = int(active_wires) + sum_found - (int(np.sum(decls)) - decls[bad] - decls[bom])
        bad_sum = bom_sum = norm = 0.0
        if 0 <= t_free <= 2 * H - 1:
          for bad_wires in range(t_free + 1):
            bom_wires = t_free - bad_wires
            place = uf.C(bad_wires, H) * uf.C(bom_wires, H - 1)
            if place == 0:
              continue
            weight = (place * uf.Lklhd(H, bad_wires, revealed[bad], found[bad])
                      * L_bomb_hand(found[bom], revealed[bom], bom_wires, H))
            bad_sum += weight * (bad_wires - found[bad])
            bom_sum += weight * (bom_wires - found[bom])
            norm += weight
        if norm > 0:
          remaining[bad] = bad_sum / norm
          remaining[bom] = bom_sum / norm
      for i in range(num_players):
        cards_left = H - revealed[i]
        r = remaining[i]
        if cards_left > 0 and not np.isnan(r) and 0 <= r <= cards_left:
          p_wire[i] += p * r / cards_left
  return p_wire


def ProbSus(players, probs):  # Modifies probs in-place
  """Fold a free-form "suspicious behaviour" observation about one player into the
  belief matrix. Prompts for the likelihood of the behaviour under each of the four
  role/bomb hypotheses for the suspect and applies Bayes' rule over the N x N matrix.
  Interactive; used by ``Play``.
  """
  num_players = len(players)
  sus_str = input("Who is sus? ")
  while sus_str not in players:
    sus_str = input("You must have made a typo. Who? ")
  sus = players.index(sus_str)
  (probs_bad, probs_bomb) = DeMatrix(probs)
  probs_bob = probs_bad[sus] - probs[sus][sus]   # bad, no bomb
  probs_gib = probs_bomb[sus] - probs[sus][sus]  # good, bomb
  probs_gob = 1 - probs[sus][sus] - probs_bob - probs_gib  # good, no bomb
  lklhd_gob = int(input("Likelihood as a good guy without the bomb? ")) / 100
  lklhd_gib = int(input("Likelihood as a good guy with the bomb? ")) / 100
  lklhd_bob = int(input("Likelihood as a bad guy without the bomb? ")) / 100
  lklhd_bib = int(input("Likelihood as a bad guy with the bomb? ")) / 100
  marginal = (probs[sus][sus] * lklhd_bib + probs_bob * lklhd_bob
              + probs_gib * lklhd_gib + probs_gob * lklhd_gob)
  if marginal == 0:
    return probs
  lklhd = np.full([num_players, num_players], lklhd_gob)
  lklhd[sus, :] = lklhd_bob   # sus is the bad guy
  lklhd[:, sus] = lklhd_gib   # sus holds the bomb
  lklhd[sus, sus] = lklhd_bib
  probs *= lklhd / marginal
  return probs


# Entropy Calculations (deprecated)
def H(probs):
  """Shannon entropy (bits) of a 1-D probability vector."""
  h = 0
  for p in probs:
    if p > 0.0001:
      h += p * np.log2(1 / p)
  return h


def H2(probs):
  """Shannon entropy (bits) of the N x N config matrix."""
  h = 0
  for line in probs:
    for p in line:
      if p > 0.0001:
        h += p * np.log2(1 / p)
  return h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop):
  """Min-entropy lookahead: the cut sequence (depth ``stop``) minimising the expected
  entropy of the config belief, averaging each cut's wire/no-wire outcomes by
  ``P_wire``. Returns ``(expected_entropy, path)``. Deprecated exploratory strategy.
  """
  if stop <= 0:
    return (H2(probs), [])
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  h = np.zeros(num_players)
  h_wire = 0
  h_not_wire = 0
  path = []
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] += 1
    find = np.zeros(num_players, dtype=int)
    find[cutee] += 1
    if p_wire[cutee] > 0.0001:
      new_probs = ProbCut(decls, probs, revealed + reveal, found + find, hand_size, active_wires)
      (h_wire, path) = H_Min(decls, new_probs, revealed + reveal, found + find, hand_size, active_wires, stop - 1)
    if p_wire[cutee] < 0.9999:
      new_probs = ProbCut(decls, probs, revealed + reveal, found, hand_size, active_wires)
      (h_not_wire, path) = H_Min(decls, new_probs, revealed + reveal, found, hand_size, active_wires, stop - 1)
    h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_not_wire
  min_cutee = 0
  min_h = h[0]
  for cutee in range(num_players):
    if path != [] and cutee == path[-1]:
      continue
    if h[cutee] < min_h:
      min_cutee = cutee
      min_h = h[cutee]
  return (min_h, path + [min_cutee])
