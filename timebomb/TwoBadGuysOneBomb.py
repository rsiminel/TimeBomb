# -*- coding: utf-8 -*-
"""Two bad guys, one bomb (B=2, M=1).

This variant unifies the two structures already validated separately: the B>1 bad
*pair* (docs/model.md §3.4.1, uniform-placement multivariate hypergeometric) and the
bomb sub-model (§3.2–§3.4, the uniform-lie bomb model with the bomb as a must-not-draw
card). A configuration is the triple ``(b1, b2, h)`` -- ``{b1, b2}`` is the unordered
bad pair (stored lower-triangular, ``b1 > b2``) and ``h`` holds the bomb, with ``h``
allowed to coincide with a bad guy. The belief state is the ``N x N x N`` tensor
``probs[b1][b2][h]``, non-zero only for ``b1 > b2``.

A hand is *truthful* iff its owner is good and bomb-free, i.e. every hand outside the
free set ``{b1, b2, h}``. The free hands hold ``t_free`` wires between them, placed
uniformly over their non-bomb slots (the bomb hand offering ``H-1``, the others ``H``).

P(bad pair) is the tensor's pair marginal and **accumulates** across rounds (via
``CombineProbs`` on the pair matrix, as the roles are fixed); P(bomb) is the bomb
marginal, read from the **current round only** -- the bomb is re-dealt every round, so
it must never be combined across rounds (§3.5).
"""

# Imports
import numpy as np
from random import randint, sample
import UsefulFunctions as uf


def PlayAuto(num_players=6, initial_hand_size=5, verbosity=2):
  """Simulate one full game with random play, tracking the belief state.

  Assigns two random bad guys (fixed for the whole game). Each round deals the bomb to
  a random hand and the active wires among the remaining slots, generates declarations
  under the uniform-lie bomb model (good and bomb-free => truthful; bad or bomb-holder
  => uniform on {0..H}), then cuts cards at random until the good guys find every
  active wire, the bomb is cut, or time runs out. Returns ``(good_guys_won,
  final_marginals, roles)``: ``good_guys_won`` is 1/0, ``final_marginals`` is the
  combined per-player P(bad) vector (the bomb marginalised out, then the pair matrix
  combined across rounds and reduced to players, §3.5), ``roles`` marks the bad pair.

  ``verbosity``: 0 silent, 1 prints game events, 2 also prints the belief marginals.
  """
  hand_size = initial_hand_size
  active_wires = num_players
  # Distributing roles (fixed for the whole game)
  roles = np.zeros(num_players)
  for i in sample(range(num_players), 2):
    roles[i] = 1
  if verbosity > 0:
    print("Roles:", roles)
  # Initialize probabilities (one accumulated P(bad pair) matrix per round)
  probabilities_list = []
  # Starting turns
  while hand_size > 1:
    if verbosity > 0:
      print("Round ", initial_hand_size - hand_size + 1)
    # Deal the bomb to a random hand, then the wires among the remaining slots
    # (the bomb hand holds at most H-1 wires).
    bomb = np.zeros(num_players, dtype=int)
    bomb[randint(0, num_players - 1)] = 1
    # slot-uniform deal (multivariate hypergeometric); the bomb hand offers H-1 slots
    slots = [(g, s) for g in range(num_players) for s in range(hand_size - int(bomb[g]))]
    wires = np.zeros(num_players, dtype=int)
    for (g, _s) in sample(slots, active_wires):
      wires[g] += 1
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
    prob_pair, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_pair.copy())
    if verbosity > 1:
      print("  P(bad): ", DeMatrix(CombineProbs(probabilities_list)))
      print("  P(bomb):", prob_bomb)
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    if verbosity > 1:
      PrintPanel(["A", "B", "C", "D", "E", "F"], declarations, probabilities, revealed, found, hand_size, active_wires)
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
      if randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      if verbosity > 0:
        print("r:", revealed)
        print("f:", found)
      # Update probabilities (Bayes on the round's declaration prior with the cuts so far)
      probs = ProbCut(declarations, probabilities, revealed, found, hand_size, active_wires)
      prob_pair, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_pair.copy()
      if verbosity > 1:
        print("  P(bad): ", DeMatrix(CombineProbs(probabilities_list)))
        print("  P(bomb):", prob_bomb)
      # Test for victory
      if active_wires <= 0:
        if verbosity > 0:
          print("Good guys win!")
        return (1, DeMatrix(CombineProbs(probabilities_list)), roles)
      if verbosity > 1:
        PrintPanel(["A", "B", "C", "D", "E", "F"], declarations, probs, revealed, found, hand_size, active_wires)
    # Next round
    hand_size -= 1
    if verbosity > 0:
      print("\n")
  if verbosity > 0:
    print("Out of time. Bad guys win!")
  return (0, DeMatrix(CombineProbs(probabilities_list)), roles)


def Play(players=["Alice", "Bob", "Clara", "Darryl", "Erica", "Fred"], initial_hand_size=5):
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
    prob_pair, prob_bomb = DeTensor(probabilities)
    probabilities_list.append(prob_pair.copy())
    print(" P(bad): ", DeMatrix(CombineProbs(probabilities_list)))
    print(" P(bomb):", prob_bomb)
    # Cut wires
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    PrintPanel(players, declarations, probabilities, revealed, found, hand_size, active_wires)
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
      prob_pair, prob_bomb = DeTensor(probs)
      probabilities_list[-1] = prob_pair.copy()
      print(" P(bad): ", DeMatrix(CombineProbs(probabilities_list)))
      print(" P(bomb):", prob_bomb)
      # Test for victory
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
      if i < num_players - 1:  # show the panel for the next cut decision this round
        PrintPanel(players, declarations, probs, revealed, found, hand_size, active_wires)
    # Next round
    hand_size -= 1
  print("Out of time. Bad guys win!")
  return


def DeTensor(probabilities):
  """Marginalise the N x N x N config tensor into its two readouts (§3.5):
  the **pair matrix** ``P(bad pair = {b1, b2}) = Σ_h probs[b1][b2][h]`` (a
  lower-triangular N x N matrix) and the **bomb vector**
  ``P(bomb = h) = Σ_{b1 > b2} probs[b1][b2][h]``. Returns ``(prob_pair, prob_bomb)``.
  """
  probabilities = np.asarray(probabilities)
  num_players = probabilities.shape[0]
  prob_pair = probabilities.sum(axis=2)  # sum over the bomb axis
  prob_pair = np.tril(prob_pair, -1)     # keep only b1 > b2 (defensive)
  prob_bomb = np.zeros(num_players)
  for b1 in range(num_players):
    for b2 in range(b1):
      prob_bomb += probabilities[b1][b2]
  return (prob_pair, prob_bomb)


def DeMatrix(probabilities):
  """Marginalise the pair matrix to a per-player P(bad) vector.

  ``P(player i is bad) = Σ_j probabilities[i][j]`` over every pair containing i
  (summing both the row and the column of the lower-triangular matrix). Returns a
  length-N vector; its entries sum to 2 (each pair contributes to two players).
  """
  num_players = len(probabilities)
  probability_line = np.zeros(num_players)
  for i in range(num_players):
    for j in range(i):
      probability_line[i] += probabilities[i][j]
      probability_line[j] += probabilities[i][j]
  return probability_line


def CombineProbs(probabilities_list, eps=1e-9):
  """Combine per-round P(bad pair) matrices into one accumulated belief (ADR 0005).

  Roles are fixed for the game while wires and the bomb are re-dealt each round, so
  the per-round pair marginals are conditionally independent evidence about the same
  fixed bad pair: the exact posterior is their elementwise product, renormalised. The
  bomb marginal is per-round and is **never** passed in here (§3.5).

  This is the General.py-bound *robust* form of that product (ADR 0005), prototyped in
  this variant to de-risk the General.py work -- a deliberate, authorised deviation
  from ADR 0005's "land it in General.py, don't retrofit the pinned variants" (it is
  numerically identical to the plain product for normal play, so the variant's pinned
  tests are unaffected). Two hardenings over the bare product:

  * **eps-floor.** Each round's vector is mixed with ``eps * uniform`` before
    combining, so a single round's hard ``0`` -- which an idealised lie model can emit
    for a real but "impossible" observation -- cannot *permanently* eliminate a pair;
    later evidence can revive it.
  * **Log-space accumulation.** Sum the per-round logs and softmax-normalise, so many
    rounds / large N cannot underflow the product to all-zeros (which the old code
    silently dumped to uniform, discarding real evidence).

  Falls back to a uniform distribution over the C(N, 2) pairs only for a genuinely
  degenerate round (a per-round vector that is all zeros over the valid pairs).
  """
  if probabilities_list == []:
    return np.array([])
  num_players = probabilities_list[0].shape[0]
  mask = np.tril(np.ones([num_players, num_players], dtype=bool), -1)  # b1 > b2 cells
  uniform = mask / mask.sum()  # uniform over the C(N, 2) real pairs
  log_acc = np.zeros([num_players, num_players])
  for round_probs in probabilities_list:
    r = np.tril(np.asarray(round_probs, dtype=float), -1)
    s = r.sum()
    r = r / s if s > 0 else uniform.copy()  # defensively normalise each round's vector
    r = (1 - eps) * r + eps * uniform        # eps-floor: no pair is ever hard-zeroed
    log_acc[mask] += np.log(r[mask])
  # Softmax over the valid cells (log-space => underflow-proof).
  shifted = log_acc[mask] - log_acc[mask].max()
  weights = np.zeros([num_players, num_players])
  weights[mask] = np.exp(shifted)
  return weights / weights.sum()


def ProbDeclaration(decls, hand_size, active_wires):
  """Prior P(bad pair {b1, b2}, bomb in hand h) from declarations alone (§3.3).

  Marking the config ``(b1, b2, h)`` pins every truthful hand ``j not in {b1, b2, h}``
  to its declared count and forces the free hands' wire total
  ``t_free = A - Σ_{truthful} decls[j]`` into their non-bomb slots. Under a uniform
  deal the prior is the multivariate-hypergeometric probability of that forced deal,
  times the probability each free hand (a uniform liar over ``{0..H}``) declared what
  it did -- a factor ``(H+1)^{-|F|}`` for the ``|F|`` free hands. That lie factor does
  **not** cancel here (ADR 0007): ``|F| = 2`` when the bomb sits on a bad guy but
  ``|F| = 3`` when it sits on a good guy (one extra good liar), so it down-weights the
  bomb-on-good configs by ``1/(H+1)``. Collapsing the wire split via Vandermonde gives

    P(b1, b2, h) ∝ (H+1)^{-|F|} · C(free_slots, t_free) / Π_{g in {b1,b2,h}} C(H, decls[g])

  where ``free_slots = Σ_{g free} (H - [g == h])`` -- the bomb eats one wire slot, so
  the free set offers ``2H-1`` slots when h is a bad guy and ``3H-1`` when h is a good
  guy (``C(n, k) = 0`` for ``k < 0`` or ``k > n``). Falls back to the uniform tensor
  over valid pairs if every config has zero weight (degeneracy, §3.3). Returns an
  ``N x N x N`` tensor (non-zero only for b1 > b2) summing to 1.
  """
  num_players = decls.size
  H = int(hand_size)
  decls = decls.astype(int)
  total_decls = int(np.sum(decls))
  probs = np.zeros([num_players, num_players, num_players])
  for b1 in range(num_players):
    for b2 in range(b1):
      for h in range(num_players):
        free = {b1, b2, h}
        t_free = int(active_wires) - (total_decls - sum(int(decls[g]) for g in free))
        free_slots = sum(H - (1 if g == h else 0) for g in free)
        if not (0 <= t_free <= free_slots):
          continue
        denom = 1.0
        for g in free:
          denom *= uf.C(decls[g], H)
        if denom != 0:
          lie_factor = (H + 1) ** (-len(free))  # each free hand is a uniform liar (ADR 0007)
          probs[b1][b2][h] = lie_factor * uf.C(t_free, free_slots) / denom
  total = np.sum(probs)
  if total == 0:  # Declarations impossible under the model: uniform over valid configs
    for b1 in range(num_players):
      for b2 in range(b1):
        for h in range(num_players):
          probs[b1][b2][h] = 1.0
    return probs / np.sum(probs)
  return probs / total


def L_bomb_hand(found_h, revealed_h, bomb_wires, hand_size):
  """Cut likelihood for the bomb hand: P(find ``found_h`` wires AND draw no bomb in
  ``revealed_h`` cuts) given the hand holds ``bomb_wires`` wires, one bomb, and
  ``H-1-bomb_wires`` blanks -- the must-not-draw hypergeometric of model.md §3.2:

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


def L_bad_pair(hand_size, bg_wires, revealed, found, i, j):
  """Likelihood of the cut observation in bad hands i and j (no bomb), given they hold
  ``bg_wires`` wires between them (docs/model.md §3.4.1). Under uniform placement over
  the ``2H`` combined slots the per-hand split collapses to one multivariate-
  hypergeometric term:

    C(rev_i, f_i) · C(rev_j, f_j) · C(2H − rev_i − rev_j, bg − f_i − f_j) / C(2H, bg)

  Returns 0.0 for an infeasible configuration (wire total or hidden-wire count out of
  range).
  """
  hidden_slots = 2 * hand_size - revealed[i] - revealed[j]
  hidden_wires = bg_wires - found[i] - found[j]
  if not (0 <= bg_wires <= 2 * hand_size) or not (0 <= hidden_wires <= hidden_slots):
    return 0.0
  denom = uf.C(bg_wires, 2 * hand_size)
  if denom == 0:
    return 0.0
  return (uf.C(found[i], revealed[i]) * uf.C(found[j], revealed[j])
          * uf.C(hidden_wires, hidden_slots)) / denom


def L_config(decls, revealed, found, hand_size, active_wires, b1, b2, h):
  """Likelihood of the cut observation under config ``(b1, b2, h)`` (model.md §3.4).

  Truthful hands (pinned to their declared count, no bomb) contribute a plain
  hypergeometric each. The free hands' wire total ``t_free`` is split under the §3.4.1
  uniform-placement law, with the bomb hand using the must-not-draw term ``L_bomb_hand``:

    * h a good guy (h not in {b1, b2}): split t_free between the bad *pair*
      (``L_bad_pair`` over 2H slots) and the bomb hand (H-1 slots), summing
      ``C(2H, w_bad)·C(H-1, w_bomb)`` over the bomb split, normalised by C(3H-1, t_free).
    * h a bad guy (h in {b1, b2}): the free set is the bad pair, one of whom holds the
      bomb; split t_free between that bomb hand (H-1 slots, ``L_bomb_hand``) and the
      other bad hand (H slots, plain ``Lklhd``), normalised by C(2H-1, t_free).

  The split is summed explicitly; the Vandermonde collapse of the bad pair is reused
  via ``L_bad_pair`` but the bomb split is not closed-form. Returns 0.0 when ``t_free``
  is infeasible.
  """
  num_players = decls.size
  H = int(hand_size)
  free = {b1, b2, h}
  # Truthful hands: pinned to their declared count, no bomb.
  like = 1.0
  for j in range(num_players):
    if j not in free:
      like *= uf.Lklhd(H, decls[j], revealed[j], found[j])
  truthful = int(np.sum([decls[j] for j in range(num_players) if j not in free]))
  t_free = int(active_wires) + int(np.sum(found)) - truthful
  if h != b1 and h != b2:  # the bomb is held by a good guy: three free hands
    if not (0 <= t_free <= 3 * H - 1):
      return 0.0
    denom = uf.C(t_free, 3 * H - 1)
    if denom == 0:
      return 0.0
    split = 0.0
    for w_bomb in range(t_free + 1):
      w_bad = t_free - w_bomb
      place = uf.C(w_bad, 2 * H) * uf.C(w_bomb, H - 1)  # 0 when out of range
      if place == 0:
        continue
      split += (place * L_bad_pair(H, w_bad, revealed, found, b1, b2)
                * L_bomb_hand(found[h], revealed[h], w_bomb, H))
    return like * split / denom
  # The bomb is held by a bad guy: the free set is the bad pair (two hands).
  other = b1 if h == b2 else b2
  if not (0 <= t_free <= 2 * H - 1):
    return 0.0
  denom = uf.C(t_free, 2 * H - 1)
  if denom == 0:
    return 0.0
  split = 0.0
  for w_bomb in range(t_free + 1):
    w_other = t_free - w_bomb
    place = uf.C(w_bomb, H - 1) * uf.C(w_other, H)  # 0 when out of range
    if place == 0:
      continue
    split += (place * L_bomb_hand(found[h], revealed[h], w_bomb, H)
              * uf.Lklhd(H, w_other, revealed[other], found[other]))
  return like * split / denom


def ProbCut(decls, prior, revealed, found, hand_size, active_wires):
  """Bayesian posterior over the (bad pair, bomb) config after a cut (model.md §3.4).

  The configs ``(b1, b2, h)`` are mutually exclusive and exhaustive. Conditioned on a
  config, the observation factorises into the truthful hands' hypergeometrics and the
  free hands' split-marginalised likelihood (``L_config``), all conditioned on "no bomb
  drawn yet". Bayes' theorem gives

    posterior(c) = prior(c) · L(c) / Σ prior · L

  Returns the prior unchanged on a zero marginal (impossible observation) or once a
  config is already certain. ``active_wires`` is the post-cut remaining safe-wire
  total. Returns an ``N x N x N`` tensor summing to 1.
  """
  num_players = decls.size
  decls = decls.astype(int)
  lklhd = np.zeros([num_players, num_players, num_players])
  marginal = 0.0
  for b1 in range(num_players):
    for b2 in range(b1):
      for h in range(num_players):
        if prior[b1][b2][h] == 1:  # config already certain; nothing to update
          return prior
        lklhd[b1][b2][h] = L_config(decls, revealed, found, hand_size, active_wires, b1, b2, h)
        marginal += prior[b1][b2][h] * lklhd[b1][b2][h]
  if marginal == 0:  # Observation impossible under every config: keep the prior
    return prior
  posterior = prior.copy()
  for b1 in range(num_players):
    for b2 in range(b1):
      for h in range(num_players):
        posterior[b1][b2][h] *= lklhd[b1][b2][h] / marginal
  return posterior


def P_wire(decls, probs, revealed, found, hand_size, active_wires):
  """Probability that cutting one of player i's face-down cards reveals a wire.

  Mix over the config posterior ``probs[b1][b2][h]``. In each config a player is either
  truthful (holds ``decls[i] - found[i]`` remaining wires) or a free hand; for the free
  hands the expected remaining wires are taken over the §3.4.1 split posterior given the
  observation (the bomb hand using the must-not-draw weights). Each contribution is
  divided by the remaining face-down cards ``H - revealed[i]`` (which still counts the
  bomb card in the bomb hand) and gated to a feasible wire count in
  ``[0, H - revealed[i]]`` -- an impossible hypothesis contributes nothing. Players
  with no cards left score 0.
  """
  num_players = decls.size
  H = int(hand_size)
  decls = decls.astype(int)
  sum_found = int(np.sum(found))
  p_wire = np.zeros(num_players)
  for b1 in range(num_players):
    for b2 in range(b1):
      for h in range(num_players):
        p = probs[b1][b2][h]
        if p == 0:
          continue
        free = {b1, b2, h}
        remaining = np.full(num_players, np.nan)  # nan = skip this hand for this config
        for j in range(num_players):  # Truthful hands: pinned to their declaration
          if j not in free:
            remaining[j] = decls[j] - found[j]
        truthful = int(np.sum([decls[j] for j in range(num_players) if j not in free]))
        t_free = int(active_wires) + sum_found - truthful
        if h != b1 and h != b2:  # three free hands: b1, b2 (bad), h (good w/ bomb)
          b1_sum = b2_sum = h_sum = norm = 0.0
          if 0 <= t_free <= 3 * H - 1:
            for w1 in range(t_free + 1):
              for w2 in range(t_free - w1 + 1):
                wh = t_free - w1 - w2
                place = uf.C(w1, H) * uf.C(w2, H) * uf.C(wh, H - 1)
                if place == 0:
                  continue
                weight = (place * uf.Lklhd(H, w1, revealed[b1], found[b1])
                          * uf.Lklhd(H, w2, revealed[b2], found[b2])
                          * L_bomb_hand(found[h], revealed[h], wh, H))
                b1_sum += weight * (w1 - found[b1])
                b2_sum += weight * (w2 - found[b2])
                h_sum += weight * (wh - found[h])
                norm += weight
          if norm > 0:
            remaining[b1] = b1_sum / norm
            remaining[b2] = b2_sum / norm
            remaining[h] = h_sum / norm
        else:  # two free hands: the bad pair, one of whom (h) holds the bomb
          other = b1 if h == b2 else b2
          bomb_sum = other_sum = norm = 0.0
          if 0 <= t_free <= 2 * H - 1:
            for w_bomb in range(t_free + 1):
              w_other = t_free - w_bomb
              place = uf.C(w_bomb, H - 1) * uf.C(w_other, H)
              if place == 0:
                continue
              weight = (place * L_bomb_hand(found[h], revealed[h], w_bomb, H)
                        * uf.Lklhd(H, w_other, revealed[other], found[other]))
              bomb_sum += weight * (w_bomb - found[h])
              other_sum += weight * (w_other - found[other])
              norm += weight
          if norm > 0:
            remaining[h] = bomb_sum / norm
            remaining[other] = other_sum / norm
        for i in range(num_players):
          cards_left = H - revealed[i]
          r = remaining[i]
          if cards_left > 0 and not np.isnan(r) and 0 <= r <= cards_left:
            p_wire[i] += p * r / cards_left
  return p_wire


# --- Cut recommendation: the quantities-only four-stat panel (model.md §3.5) ----
#
# These functions are the General.py-bound reference for the cut panel of ADR 0006.
# Stats 1-2 are P_wire and the DeTensor bomb marginal (above); stats 3-4 are the
# information lookaheads below. They are defined here, with the bomb present, so the
# General.py implementation mirrors a verified reference rather than designing from
# scratch. The panel is quantities-only: it presents calibrated decision inputs and
# leaves the explore/exploit/risk tradeoff (the player's risk appetite) to the human.


def H(probs):
  """Shannon entropy in bits of a 1-D probability vector: 0 for a certain outcome,
  log2(k) for the uniform distribution over k outcomes (model.md §3.5).

  This ``H`` is Shannon entropy and is unrelated to the hand size also written ``H``
  in §2's table; the code reuses the letter, as the model doc warns. Masses at or
  below 1e-12 are skipped to avoid ``0 * log 0``.
  """
  h = 0.0
  for p in np.asarray(probs).ravel():
    if p > 1e-12:
      h -= p * np.log2(p)
  return h


def EntropyBad(probabilities):
  """Shannon entropy (bits) of the role posterior P(bad pair) (model.md §3.5).

  The genuine role uncertainty is the distribution over the C(N, 2) candidate bad
  *pairs* -- the lower-triangular pair marginal of the config tensor, which sums to 1
  (for B=1 this would reduce to entropy over the N players). The cut panel's
  information stats drive this toward 0. The bomb axis is marginalised out first (via
  ``DeTensor``), so it measures persistent-role uncertainty, not per-round bomb noise.
  Maximum value ``log2(C(N, 2))``.
  """
  prob_pair, _ = DeTensor(probabilities)
  return H(prob_pair[prob_pair > 0])


def NextHBad(decls, probs, revealed, found, hand_size, active_wires):
  """Stat 3 -- the 1-ply expected post-cut role entropy (model.md §3.5, ADR 0006).

  For each player with a face-down card, hypothesise cutting one uniformly random
  face-down card and average the resulting ``EntropyBad`` over its two non-terminal
  outcomes: a wire (weight ``P_wire``; the safe-wire total drops by one and a wire is
  found) or a non-wire dud (weight ``1 - P_wire``). Both branches reuse the bomb-aware
  ``ProbCut``, which conditions on "no bomb cut yet", so detonation is excluded from
  the rollout -- the panel's information stats ignore bomb risk by design (ADR 0006),
  which is why stat 2 must always be read beside them. Lower means the cut is expected
  to teach more about the fixed roles. Players with no face-down card left are
  ``np.nan``.

  This is honestly myopic -- it prices only the next cut. Stat 4 (``RoundHorizonH``)
  values a cut as the opening move of an information-gathering line, which this cannot.
  """
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  exp_h = np.full(num_players, np.nan)
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      # A wire is found: increment found AND drop the remaining safe-wire total, so the
      # round's invariant free-wire count stays put (mirrors PlayAuto's cut bookkeeping).
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1)
      h_wire = EntropyBad(probs_wire)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires)
      h_dud = EntropyBad(probs_dud)
    exp_h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
  return exp_h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop):
  """Information-greedy min-entropy lookahead over the role posterior (model.md §3.5).

  Searches up to ``stop`` cuts ahead, assuming every cut is chosen to be maximally
  informative, and returns the minimum achievable expected ``EntropyBad`` at the end
  of that window. Each cut branches into a wire (weight ``P_wire``; safe-wire total -1)
  and a dud (weight ``1 - P_wire``), both via the bomb-aware ``ProbCut`` conditioned on
  "no bomb cut yet" -- the rollout ignores bomb risk (ADR 0006), exactly as ``NextHBad``.

  Cost is exponential in ``stop`` -- ``O((2N)^stop)`` ``ProbCut`` calls -- so callers
  cap the depth for large N (a beam / analytic approximation is the General.py
  follow-up). Returns ``EntropyBad`` of the current belief when ``stop <= 0`` or no
  player has a card left to cut.
  """
  if stop <= 0:
    return EntropyBad(probs)
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  best = None
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1)
      h_wire = H_Min(decls, probs_wire, revealed + reveal, found + reveal,
                     hand_size, active_wires - 1, stop - 1)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires)
      h_dud = H_Min(decls, probs_dud, revealed + reveal, found,
                    hand_size, active_wires, stop - 1)
    expected = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
    if best is None or expected < best:
      best = expected
  return EntropyBad(probs) if best is None else best


def RoundHorizonH(decls, probs, revealed, found, hand_size, active_wires, max_depth=None):
  """Stat 4 -- the round-horizon expected role entropy under info-greedy continuation
  (model.md §3.5, ADR 0006). The headline explore stat.

  For each player with a face-down card, value *opening* the round-remainder with a cut
  on that player: expand its wire / dud outcomes one level (as ``NextHBad``), then let
  ``H_Min`` continue info-greedily for the rest of the round. A round is exactly N cuts
  (``PlayAuto``), so the natural horizon is ``cuts_left = N - sum(revealed)``;
  ``max_depth`` caps it for a responsive display (``None`` = exact to end of round).
  Lower means the cut best opens an information-gathering line. Players with no
  face-down card left are ``np.nan``.

  Ships with two caveats (ADR 0006): it is an information *potential* (the player does
  not control every cut, so info-greedy continuation is counterfactual), and the
  rollout ignores bomb risk -- always read it beside stat 2.
  """
  num_players = decls.size
  cuts_left = num_players - int(np.sum(revealed))
  depth = cuts_left if max_depth is None else min(cuts_left, max_depth)
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  round_h = np.full(num_players, np.nan)
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1)
      h_wire = H_Min(decls, probs_wire, revealed + reveal, found + reveal,
                     hand_size, active_wires - 1, depth - 1)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires)
      h_dud = H_Min(decls, probs_dud, revealed + reveal, found,
                    hand_size, active_wires, depth - 1)
    round_h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
  return round_h


def CutPanel(decls, probs, revealed, found, hand_size, active_wires, max_depth=None):
  """Assemble the quantities-only four-stat cut panel (model.md §3.5, ADR 0006).

  Returns an ``N x 4`` array whose row i, for a player with a face-down card, is
  ``[P(safe wire), P(bomb), 1-ply E[H(bad)], round-horizon H(bad)]`` -- exploit, risk,
  immediate role-info, strategic role-info. Rows for players with no face-down card
  left are all ``np.nan`` (they cannot be cut). ``max_depth`` caps stat 4's lookahead
  (see ``RoundHorizonH``). The four stats are deliberately *not* combined into a single
  score: the explore/exploit/risk integration needs a risk appetite that belongs to the
  human, not the assistant.
  """
  num_players = decls.size
  p_safe = P_wire(decls, probs, revealed, found, hand_size, active_wires)
  _, p_bomb = DeTensor(probs)
  one_ply = NextHBad(decls, probs, revealed, found, hand_size, active_wires)
  horizon = RoundHorizonH(decls, probs, revealed, found, hand_size, active_wires, max_depth)
  panel = np.full((num_players, 4), np.nan)
  for i in range(num_players):
    if revealed[i] >= hand_size:
      continue
    panel[i] = [p_safe[i], p_bomb[i], one_ply[i], horizon[i]]
  return panel


def PrintPanel(players, decls, probs, revealed, found, hand_size, active_wires, max_depth=3):
  """Pretty-print the four-stat cut panel (model.md §3.5) before a human chooses a cut.

  Stat 4's round-horizon entropy is depth-capped at ``max_depth`` for a responsive
  display, since the exact lookahead is exponential (see ``H_Min``).
  """
  decls = np.asarray(decls)
  panel = CutPanel(decls, probs, revealed, found, hand_size, active_wires, max_depth)
  print("  cut panel   [ P(safe wire) | P(bomb) | 1-ply H(bad) | round-horizon H(bad) ]")
  for i in range(len(players)):
    if np.all(np.isnan(panel[i])):
      print(f"    {players[i]:<8} (no face-down cards left)")
    else:
      ps, pb, dh, rh = panel[i]
      print(f"    {players[i]:<8} {ps:11.3f}   {pb:6.3f}   {dh:9.3f}   {rh:14.3f}")