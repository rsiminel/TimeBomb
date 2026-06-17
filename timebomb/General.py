# -*- coding: utf-8 -*-
"""General Time Bomb model — arbitrary ``B`` bad guys and ``M in {0, 1}`` bomb.

This is the unification target the four hardcoded variants converge toward
(docs/roadmap.md): the same uniform-lie model (docs/model.md §3.1–§3.5), written once
over the **general configuration space**. A configuration is a bad set ``S`` of size
``num_bad`` plus, when ``M = 1``, the player ``h`` holding the bomb this round; the
belief state is the ``[N]*(num_bad + num_bom)`` tensor indexed by the sorted bad-set
tuple followed by the bomb tuple (so only sorted-index cells are populated, e.g. lower
triangle for ``num_bad = 2``).

Every hardcoded variant is a projection of these functions: ``OneBadGuyNoBomb`` is
``(num_bad, num_bom) = (1, 0)``, ``TwoBadGuysOneBomb`` is ``(2, 1)``, etc. The closed
forms here are the corrected ones — the declaration prior carries the lie-count factor
``(H+1)^{-|F|}`` (ADR 0007), which is what makes per-configuration weights absolute and
comparable across different bad counts (needed for joint ``num_bad`` inference).

Vectorisation: the iteration over configurations is an ``itertools.combinations`` loop
(there is no numpy primitive that sums over all ``B``-subsets with a per-subset free
set), but the per-hand likelihoods, the §3.4.1 bad-group collapse, and the tensor
assembly/normalisation inside it are numpy/array operations.
"""

# Imports
import itertools
import numpy as np
from random import randint, sample, randrange
import UsefulFunctions as uf


# --- per-hand cut likelihood atoms (docs/model.md §3.2, §3.4.1) ----------------

def L_bomb_hand(found_h, revealed_h, bomb_wires, hand_size):
  """Cut likelihood for the bomb hand: P(find ``found_h`` wires AND draw no bomb in
  ``revealed_h`` cuts) given the hand holds ``bomb_wires`` wires, one bomb, and
  ``H-1-bomb_wires`` blanks — the must-not-draw hypergeometric of model.md §3.2,
  ``C(bomb_wires, found_h)·C(H-1-bomb_wires, revealed_h-found_h)/C(H, revealed_h)``.
  Returns 0.0 on an infeasible wire count or an empty denominator.
  """
  H = int(hand_size)
  if not (0 <= bomb_wires <= H - 1):
    return 0.0
  denom = uf.C(revealed_h, H)
  if denom == 0:
    return 0.0
  return uf.C(found_h, bomb_wires) * uf.C(revealed_h - found_h, H - 1 - bomb_wires) / denom


def L_bad(hand_size, group_wires, revealed, found, group):
  """§3.4.1 collapse for a group ``G`` of bomb-free bad hands holding ``group_wires``
  wires between them. Under uniform placement over the ``|G|·H`` combined slots the
  per-hand split collapses to one multivariate-hypergeometric term:

    Π_{g in G} C(revealed[g], found[g]) · C(|G|·H − Σrevealed, group_wires − Σfound)
                                        / C(|G|·H, group_wires)

  Reduces to a single ``Lklhd`` when ``|G| = 1`` and to ``1`` for an empty group with
  ``group_wires = 0`` (and ``0`` otherwise). Returns 0.0 for an infeasible split.
  """
  H = int(hand_size)
  slots = len(group) * H
  hidden_slots = slots - sum(int(revealed[g]) for g in group)
  hidden_wires = group_wires - sum(int(found[g]) for g in group)
  if not (0 <= group_wires <= slots) or not (0 <= hidden_wires <= hidden_slots):
    return 0.0
  denom = uf.C(group_wires, slots)
  if denom == 0:
    return 0.0
  num = 1.0
  for g in group:
    num *= uf.C(int(found[g]), int(revealed[g]))
  return num * uf.C(hidden_wires, hidden_slots) / denom


def _free_splits(free, t_free, slots):
  """Yield every wire split ``{g: w_g}`` of the free hands summing to ``t_free`` with
  each ``w_g`` within ``slots[g]`` (the hand's non-bomb slot count). Used to enumerate
  the free-hand wire placement where no closed form collapses it (``P_wire``)."""
  free = list(free)
  caps = [slots[g] for g in free]

  def rec(i, remaining):
    if i == len(free) - 1:
      if 0 <= remaining <= caps[i]:
        yield {free[i]: remaining}
      return
    for w in range(min(caps[i], remaining) + 1):
      for rest in rec(i + 1, remaining - w):
        rest[free[i]] = w
        yield rest

  if free:
    yield from rec(0, t_free)
  elif t_free == 0:
    yield {}


def L_config(decls, revealed, found, hand_size, active_wires, bad_set, bom_set):
  """Likelihood of the cut observation under config ``(bad_set, bom_set)`` (model.md
  §3.4), conditioned on "no bomb cut yet". Truthful hands (good, bomb-free, pinned to
  their declared count) each contribute a plain hypergeometric; the free hands'
  wire total ``t_free`` is split under the §3.4.1 uniform-placement law.

  ``M = 0``: the free set is exactly ``bad_set`` and ``L_bad`` collapses the whole
  split in closed form. ``M = 1``: the bomb hand (``H-1`` slots, ``L_bomb_hand``) cannot
  join the collapse, so the split between the bomb-free bad group and the bomb hand is
  summed explicitly while ``L_bad`` still collapses the bad group internally.
  Returns 0.0 when ``t_free`` is infeasible.
  """
  num_players = decls.size
  H = int(hand_size)
  free = set(bad_set) | set(bom_set)
  like = 1.0
  for j in range(num_players):
    if j not in free:
      like *= uf.Lklhd(H, int(decls[j]), int(revealed[j]), int(found[j]))
  truthful = sum(int(decls[j]) for j in range(num_players) if j not in free)
  t_free = int(active_wires) + int(np.sum(found)) - truthful
  if len(bom_set) == 0:  # M = 0: the free hands are the bad set; closed-form collapse
    if not (0 <= t_free <= len(bad_set) * H):
      return 0.0
    return like * L_bad(H, t_free, revealed, found, bad_set)
  # M = 1: one bomb holder h; split t_free between the bad group and the bomb hand
  h = bom_set[0]
  group = [g for g in bad_set if g != h]  # bomb-free bad hands
  group_slots = len(group) * H
  free_slots = group_slots + (H - 1)
  if not (0 <= t_free <= free_slots):
    return 0.0
  denom = uf.C(t_free, free_slots)
  if denom == 0:
    return 0.0
  split = 0.0
  for w_bomb in range(t_free + 1):
    w_grp = t_free - w_bomb
    place = uf.C(w_grp, group_slots) * uf.C(w_bomb, H - 1)  # 0 when out of range
    if place == 0:
      continue
    split += (place * L_bad(H, w_grp, revealed, found, group)
              * L_bomb_hand(int(found[h]), int(revealed[h]), w_bomb, H))
  return like * split / denom


# --- the three belief functions (model.md §3.3, §3.4, §3.5) --------------------

def ProbDeclaration(decls, hand_size, active_wires, num_bad, num_bom):
  """Prior over configurations from the round's declarations alone (model.md §3.3).

  Closed form (ADR 0007, with the lie-count factor kept):

    P(config) ∝ (H+1)^{-|F|} · C(free_slots, t_free) / Π_{g in F} C(H, decls[g])

  for the free set ``F = bad_set ∪ bom_set``, ``free_slots = Σ_{g in F}(H − [g bomb])``,
  ``t_free = A − Σ_{truthful} decls[j]``. The ``(H+1)^{-|F|}`` factor cancels for
  ``M = 0`` (``|F| = num_bad`` constant) but not for ``M = 1`` (it penalises a good
  bomb-holder's extra liar). Falls back to the uniform distribution over valid configs
  on a fully degenerate observation (§3.3). Returns a ``[N]*(num_bad+num_bom)`` tensor
  summing to 1, non-zero only on sorted-index cells.
  """
  num_players = decls.shape[0]
  H = int(hand_size)
  decls = decls.astype(int)
  total = int(np.sum(decls))
  probs = np.zeros([num_players] * (num_bad + num_bom))
  for bad_set in itertools.combinations(range(num_players), num_bad):
    for bom_set in itertools.combinations(range(num_players), num_bom):
      free = set(bad_set) | set(bom_set)
      t_free = int(active_wires) - (total - sum(int(decls[g]) for g in free))
      free_slots = sum(H - (1 if g in bom_set else 0) for g in free)
      if not (0 <= t_free <= free_slots):
        continue
      denom = 1.0
      for g in free:
        denom *= uf.C(int(decls[g]), H)
      if denom == 0:
        continue
      lie_factor = (H + 1.0) ** (-len(free))  # each free hand is a uniform liar (ADR 0007)
      probs[bad_set + bom_set] = lie_factor * uf.C(t_free, free_slots) / denom
  total_mass = np.sum(probs)
  if total_mass == 0:  # degeneracy: uniform over the valid configs (§3.3, ADR 0002)
    for bad_set in itertools.combinations(range(num_players), num_bad):
      for bom_set in itertools.combinations(range(num_players), num_bom):
        probs[bad_set + bom_set] = 1.0
    return probs / np.sum(probs)
  return probs / total_mass


def ProbCut(decls, prior, revealed, found, hand_size, active_wires, num_bad, num_bom):
  """Bayesian posterior over configurations after a cut (model.md §3.4):

    posterior(c) = prior(c) · L_config(c) / Σ_c′ prior(c′) · L_config(c′)

  conditioned on "no bomb cut yet". ``active_wires`` is the **post-cut remaining**
  safe-wire total. Returns the prior unchanged on a zero marginal (impossible
  observation) or once a configuration is already certain. Returns a
  ``[N]*(num_bad+num_bom)`` tensor summing to 1.
  """
  num_players = decls.size
  decls = decls.astype(int)
  lklhd = np.zeros([num_players] * (num_bad + num_bom))
  marginal = 0.0
  for bad_set in itertools.combinations(range(num_players), num_bad):
    for bom_set in itertools.combinations(range(num_players), num_bom):
      idx = bad_set + bom_set
      if prior[idx] == 1:  # configuration already certain; nothing to update
        return prior
      lklhd[idx] = L_config(decls, revealed, found, hand_size, active_wires, bad_set, bom_set)
      marginal += prior[idx] * lklhd[idx]
  if marginal == 0:  # observation impossible under every config: keep the prior
    return prior
  return prior * lklhd / marginal


def P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom):
  """Probability that cutting one of player i's face-down cards reveals a wire
  (model.md §3.5). Mix over the config posterior: in each config a player is truthful
  (``decls[i] − found[i]`` remaining wires) or a free hand (expected remaining wires
  under the §3.4.1 split posterior given the observation, the bomb hand using the
  must-not-draw weights). Each contribution is divided by the remaining face-down
  cards ``H − revealed[i]`` (still counting the bomb card) and gated to a feasible
  wire count. Players with no cards left score 0.
  """
  num_players = decls.size
  H = int(hand_size)
  decls = decls.astype(int)
  sum_found = int(np.sum(found))
  p_wire = np.zeros(num_players)
  for bad_set in itertools.combinations(range(num_players), num_bad):
    for bom_set in itertools.combinations(range(num_players), num_bom):
      p = probs[bad_set + bom_set]
      if p == 0:
        continue
      free = sorted(set(bad_set) | set(bom_set))
      bomb = bom_set[0] if num_bom else None
      slots = {g: H - (1 if g == bomb else 0) for g in free}
      free_slots = sum(slots.values())
      truthful = sum(int(decls[j]) for j in range(num_players) if j not in free)
      t_free = int(active_wires) + sum_found - truthful
      remaining = np.full(num_players, np.nan)  # nan = skip this hand for this config
      for j in range(num_players):
        if j not in free:
          remaining[j] = decls[j] - found[j]
      sums = {g: 0.0 for g in free}
      norm = 0.0
      if 0 <= t_free <= free_slots:
        for split in _free_splits(free, t_free, slots):
          place = 1.0
          obs = 1.0
          for g in free:
            wg = split[g]
            place *= uf.C(wg, slots[g])
            obs *= (L_bomb_hand(int(found[g]), int(revealed[g]), wg, H) if g == bomb
                    else uf.Lklhd(H, wg, int(revealed[g]), int(found[g])))
          weight = place * obs
          if weight == 0:
            continue
          for g in free:
            sums[g] += weight * (split[g] - found[g])
          norm += weight
      if norm > 0:
        for g in free:
          remaining[g] = sums[g] / norm
      for i in range(num_players):
        cards_left = H - revealed[i]
        r = remaining[i]
        if cards_left > 0 and not np.isnan(r) and 0 <= r <= cards_left:
          p_wire[i] += p * r / cards_left
  return p_wire


# --- marginals (model.md §3.5) -------------------------------------------------

def Separate(probabilities, num_bad, num_bom):
  """Marginalise the config tensor into ``(prob_bad, prob_bom)``: the bad-set marginal
  ``[N]*num_bad`` (``P(bad set) = Σ_bomb probs``, the **persistent** belief) and the
  bomb marginal ``[N]*num_bom`` (``P(bomb holders) = Σ_bad probs``, read **per round**;
  never combined across rounds, §3.5). For ``num_bom = 0`` the bomb marginal is the
  scalar total."""
  num_players = probabilities.shape[0]
  prob_bad = np.zeros([num_players] * num_bad)
  prob_bom = np.zeros([num_players] * num_bom)
  for bad in itertools.combinations(range(num_players), num_bad):
    for bom in itertools.combinations(range(num_players), num_bom):
      prob_bad[bad] += probabilities[bad + bom]
      prob_bom[bom] += probabilities[bad + bom]
  return prob_bad, prob_bom


def DeMatrix(prob_bad):
  """Reduce a bad-set marginal ``[N]*num_bad`` to a per-player ``P(player i bad)``
  vector by summing every bad set that contains ``i``. The entries sum to ``num_bad``
  (each bad set contributes to its members)."""
  num_players = prob_bad.shape[0]
  num_bad = prob_bad.ndim
  line = np.zeros(num_players)
  for bad in itertools.combinations(range(num_players), num_bad):
    for i in bad:
      line[i] += prob_bad[bad]
  return line
