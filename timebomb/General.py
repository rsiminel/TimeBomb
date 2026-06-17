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

def _decl_weights(decls, hand_size, active_wires, num_bad, num_bom):
  """Unnormalised declaration weights over configurations (ADR 0007):

    w(config) = (H+1)^{-|F|} · C(free_slots, t_free) / Π_{g in F} C(H, decls[g])

  These are **absolute** — comparable across different bad counts ``num_bad`` (the
  per-round, B-independent constants ``Π_all C(H,decls)/C(N·H−M, A)`` and the ``1/N``
  bomb prior are dropped because they cancel in any posterior over configs *or* over B).
  ``ProbDeclaration`` renormalises these; the joint num_bad layer (ADR 0008) keeps them
  unnormalised to weigh one ``num_bad`` against another. Returns a
  ``[N]*(num_bad+num_bom)`` tensor.
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
  return probs


def ProbDeclaration(decls, hand_size, active_wires, num_bad, num_bom):
  """Prior over configurations from the round's declarations alone (model.md §3.3).

  Wraps ``_decl_weights`` (which carries the absolute, cross-B-comparable weights used
  by the joint num_bad layer of ADR 0008) and renormalises, with the §3.3/ADR 0002
  uniform fallback on a fully degenerate observation.

  Closed form (ADR 0007, with the lie-count factor kept):

    P(config) ∝ (H+1)^{-|F|} · C(free_slots, t_free) / Π_{g in F} C(H, decls[g])

  for the free set ``F = bad_set ∪ bom_set``, ``free_slots = Σ_{g in F}(H − [g bomb])``,
  ``t_free = A − Σ_{truthful} decls[j]``. The ``(H+1)^{-|F|}`` factor cancels for
  ``M = 0`` (``|F| = num_bad`` constant) but not for ``M = 1`` (it penalises a good
  bomb-holder's extra liar). Falls back to the uniform distribution over valid configs
  on a fully degenerate observation (§3.3). Returns a ``[N]*(num_bad+num_bom)`` tensor
  summing to 1, non-zero only on sorted-index cells.
  """
  probs = _decl_weights(decls, hand_size, active_wires, num_bad, num_bom)
  total_mass = np.sum(probs)
  if total_mass == 0:  # degeneracy: uniform over the valid configs (§3.3, ADR 0002)
    num_players = decls.shape[0]
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


# --- cross-round combination (ADR 0005) ----------------------------------------

def _badset_mask(shape):
  """Boolean mask of the valid sorted bad-set cells of a ``[N]*num_bad`` tensor."""
  num_players, num_bad = shape[0], len(shape)
  mask = np.zeros(shape, dtype=bool)
  for bad in itertools.combinations(range(num_players), num_bad):
    mask[bad] = True
  return mask


def CombineProbs(probabilities_list, eps=1e-9):
  """Combine per-round P(bad set) marginals into one accumulated belief (ADR 0005).

  Roles are fixed while wires and the bomb are re-dealt each round, so the per-round
  bad-set marginals are conditionally independent evidence about the same fixed bad
  set: the exact posterior is their elementwise product, renormalised. The bomb
  marginal is per-round and is **never** passed in here (§3.5).

  Robust form (the same hardenings prototyped in `TwoBadGuysOneBomb`): each round's
  vector is ``eps``-floored toward uniform (so one round's hard 0 cannot permanently
  eliminate a bad set) and the product is accumulated in log-space then softmax-
  normalised (so many rounds / large N cannot underflow to uniform). Numerically
  identical to the plain product for normal play. Falls back to uniform over the
  ``C(N, num_bad)`` sets only for a genuinely all-zero round.
  """
  if probabilities_list == []:
    return np.array([])
  shape = probabilities_list[0].shape
  mask = _badset_mask(shape)
  uniform = mask / mask.sum()
  log_acc = np.zeros(shape)
  for round_probs in probabilities_list:
    r = np.where(mask, np.asarray(round_probs, dtype=float), 0.0)
    s = r.sum()
    r = r / s if s > 0 else uniform.copy()  # defensively normalise each round
    r = (1 - eps) * r + eps * uniform        # eps-floor: no set ever hard-zeroed
    log_acc[mask] += np.log(r[mask])
  shifted = log_acc[mask] - log_acc[mask].max()  # log-space => underflow-proof
  weights = np.zeros(shape)
  weights[mask] = np.exp(shifted)
  return weights / weights.sum()


# --- the quantities-only four-stat cut panel (model.md §3.5, ADR 0006) ----------

def H(probs):
  """Shannon entropy in bits of a 1-D probability vector (0 for certain, log2(k) for
  uniform over k outcomes). Distinct from the hand size ``H`` of §2 (the code reuses
  the letter, as model.md warns); masses <= 1e-12 are skipped to avoid 0*log0."""
  h = 0.0
  for p in np.asarray(probs).ravel():
    if p > 1e-12:
      h -= p * np.log2(p)
  return h


def EntropyBad(probabilities, num_bad, num_bom):
  """Shannon entropy (bits) of the role posterior P(bad set) — the genuine role
  uncertainty over the ``C(N, num_bad)`` candidate bad sets (model.md §3.5). The bomb
  axis is marginalised out first (via ``Separate``), so it measures persistent-role
  uncertainty, not per-round bomb noise. Maximum ``log2(C(N, num_bad))``."""
  prob_bad, _ = Separate(probabilities, num_bad, num_bom)
  return H(prob_bad[_badset_mask(prob_bad.shape)])


def NextHBad(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom):
  """Stat 3 — the 1-ply expected post-cut role entropy (model.md §3.5, ADR 0006).

  For each player with a face-down card, average ``EntropyBad`` over the two
  non-terminal outcomes of cutting one uniformly random face-down card: a wire (weight
  ``P_wire``; the safe-wire total drops by one) or a non-wire dud (weight
  ``1 - P_wire``). Both branches use the bomb-aware ``ProbCut`` conditioned on "no bomb
  cut yet", so detonation is excluded — the information stats ignore bomb risk by design
  (ADR 0006). Lower = the cut teaches more about the fixed roles. Players with no
  face-down card left are ``np.nan``.
  """
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  exp_h = np.full(num_players, np.nan)
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1, num_bad, num_bom)
      h_wire = EntropyBad(probs_wire, num_bad, num_bom)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires, num_bad, num_bom)
      h_dud = EntropyBad(probs_dud, num_bad, num_bom)
    exp_h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
  return exp_h


def H_Min(decls, probs, revealed, found, hand_size, active_wires, stop, num_bad, num_bom):
  """Information-greedy min-entropy lookahead over the role posterior (model.md §3.5):
  the minimum expected ``EntropyBad`` reachable in ``stop`` maximally-informative cuts.
  Each cut branches into a wire (weight ``P_wire``; safe-wire total -1) and a dud, both
  via the bomb-aware ``ProbCut`` (ignores bomb risk, ADR 0006). Cost is ``O((2N)^stop)``
  ``ProbCut`` calls, so callers cap the depth for large N (a beam/analytic approximation
  is the follow-up). Returns ``EntropyBad`` when ``stop <= 0`` or nothing is cuttable.
  """
  if stop <= 0:
    return EntropyBad(probs, num_bad, num_bom)
  num_players = decls.size
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  best = None
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1, num_bad, num_bom)
      h_wire = H_Min(decls, probs_wire, revealed + reveal, found + reveal,
                     hand_size, active_wires - 1, stop - 1, num_bad, num_bom)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires, num_bad, num_bom)
      h_dud = H_Min(decls, probs_dud, revealed + reveal, found,
                    hand_size, active_wires, stop - 1, num_bad, num_bom)
    expected = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
    if best is None or expected < best:
      best = expected
  return EntropyBad(probs, num_bad, num_bom) if best is None else best


def RoundHorizonH(decls, probs, revealed, found, hand_size, active_wires,
                  num_bad, num_bom, max_depth=None):
  """Stat 4 — the round-horizon expected role entropy under info-greedy continuation
  (model.md §3.5, ADR 0006). For each player with a face-down card, value opening with
  a cut there (its wire/dud outcomes) then ``H_Min`` for the rest of the round. A round
  is N cuts, so the horizon is ``cuts_left = N - sum(revealed)``; ``max_depth`` caps it
  for a responsive display. Lower = the cut best opens an information-gathering line.
  Ships with two caveats (ADR 0006): it is an information *potential*, and the rollout
  ignores bomb risk — always read beside stat 2. Players with no card left are ``nan``.
  """
  num_players = decls.size
  cuts_left = num_players - int(np.sum(revealed))
  depth = cuts_left if max_depth is None else min(cuts_left, max_depth)
  p_wire = P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  round_h = np.full(num_players, np.nan)
  for cutee in range(num_players):
    if revealed[cutee] >= hand_size:
      continue
    reveal = np.zeros(num_players, dtype=int)
    reveal[cutee] = 1
    h_wire = h_dud = 0.0
    if p_wire[cutee] > 1e-9 and active_wires > 0:
      probs_wire = ProbCut(decls, probs, revealed + reveal, found + reveal,
                           hand_size, active_wires - 1, num_bad, num_bom)
      h_wire = H_Min(decls, probs_wire, revealed + reveal, found + reveal,
                     hand_size, active_wires - 1, depth - 1, num_bad, num_bom)
    if p_wire[cutee] < 1 - 1e-9:
      probs_dud = ProbCut(decls, probs, revealed + reveal, found,
                          hand_size, active_wires, num_bad, num_bom)
      h_dud = H_Min(decls, probs_dud, revealed + reveal, found,
                    hand_size, active_wires, depth - 1, num_bad, num_bom)
    round_h[cutee] = p_wire[cutee] * h_wire + (1 - p_wire[cutee]) * h_dud
  return round_h


def CutPanel(decls, probs, revealed, found, hand_size, active_wires,
             num_bad, num_bom, max_depth=None):
  """Assemble the quantities-only four-stat cut panel (model.md §3.5, ADR 0006).

  Returns an ``N x 4`` array whose row i (for a player with a face-down card) is
  ``[P(safe wire), P(bomb), 1-ply E[H(bad)], round-horizon H(bad)]`` — exploit, risk,
  immediate role-info, strategic role-info. Rows for players with no card left are all
  ``np.nan``. The stats are deliberately *not* combined into one score: the
  explore/exploit/risk tradeoff needs a risk appetite that belongs to the human.
  """
  num_players = decls.size
  p_safe = P_wire(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  if num_bom:
    _, prob_bom = Separate(probs, num_bad, num_bom)
    p_bomb = np.asarray(prob_bom).reshape(-1)[:num_players] if num_bom == 1 else np.zeros(num_players)
  else:
    p_bomb = np.zeros(num_players)
  one_ply = NextHBad(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom)
  horizon = RoundHorizonH(decls, probs, revealed, found, hand_size, active_wires,
                          num_bad, num_bom, max_depth)
  panel = np.full((num_players, 4), np.nan)
  for i in range(num_players):
    if revealed[i] >= hand_size:
      continue
    panel[i] = [p_safe[i], p_bomb[i], one_ply[i], horizon[i]]
  return panel


def PrintPanel(players, decls, probs, revealed, found, hand_size, active_wires,
               num_bad, num_bom, max_depth=3):
  """Pretty-print the four-stat cut panel (model.md §3.5) before a human chooses a cut.
  Stat 4's round-horizon entropy is depth-capped at ``max_depth`` (the exact lookahead
  is exponential, see ``H_Min``)."""
  decls = np.asarray(decls)
  panel = CutPanel(decls, probs, revealed, found, hand_size, active_wires,
                   num_bad, num_bom, max_depth)
  print("  cut panel   [ P(safe wire) | P(bomb) | 1-ply H(bad) | round-horizon H(bad) ]")
  for i in range(len(players)):
    if np.all(np.isnan(panel[i])):
      print(f"    {players[i]:<8} (no face-down cards left)")
    else:
      ps, pb, dh, rh = panel[i]
      print(f"    {players[i]:<8} {ps:11.3f}   {pb:6.3f}   {dh:9.3f}   {rh:14.3f}")


# --- joint inference over the number of bad guys (ADR 0008) --------------------

def NUM_BAD_PRIOR(num_players):
  """Prior P(num_bad) from the role-card composition by player count. For most counts
  the bad count is fixed; for N=4 it is 1 or 2 and for N=7 it is 2 or 3 (the deal leaves
  it uncertain). Returns a dict {num_bad: probability}."""
  if num_players == 4:
    return {1: 2 / 5, 2: 3 / 5}
  if num_players in (5, 6):
    return {2: 1.0}
  if num_players == 7:
    return {2: 3 / 8, 3: 5 / 8}
  if num_players == 8:
    return {3: 1.0}
  raise ValueError("Time Bomb supports 4-8 players")


def _cut_likelihoods(decls, revealed, found, hand_size, active_wires, num_bad, num_bom):
  """The ``L_config`` likelihood of the cut observation for every configuration, as a
  ``[N]*(num_bad+num_bom)`` tensor (the per-config factors ``ProbCut`` multiplies the
  prior by)."""
  num_players = decls.size
  lk = np.zeros([num_players] * (num_bad + num_bom))
  for bad_set in itertools.combinations(range(num_players), num_bad):
    for bom_set in itertools.combinations(range(num_players), num_bom):
      lk[bad_set + bom_set] = L_config(decls, revealed, found, hand_size,
                                       active_wires, bad_set, bom_set)
  return lk


def RoundLogU(decls, revealed, found, hand_size, total_active, active_wires, num_bad, num_bom):
  """Per-round log unnormalised bad-set marginal ``log u_r(S; B)`` (ADR 0008): the
  absolute declaration weight (at the round-start ``total_active``) times the cut
  likelihood (at the round's final cut state), summed over the bomb axis. ``-inf`` marks
  a bad set ruled out this round. Accumulating these across rounds and feeding them to
  ``JointBadBelief`` gives the joint posterior over how many bad guys there are.
  """
  decl_w = _decl_weights(decls, hand_size, total_active, num_bad, num_bom)
  cut_lk = _cut_likelihoods(decls, revealed, found, hand_size, active_wires, num_bad, num_bom)
  u_bad, _ = Separate(decl_w * cut_lk, num_bad, num_bom)  # sum over the bomb axis
  mask = _badset_mask(u_bad.shape)
  log_u = np.full(u_bad.shape, -np.inf)
  pos = mask & (u_bad > 0)
  log_u[pos] = np.log(u_bad[pos])
  return log_u


def JointBadBelief(log_u_by_b, prior_b):
  """Combine per-num_bad accumulated ``log u`` tensors into one belief (ADR 0008).

  ``log_u_by_b`` maps each candidate ``num_bad`` to ``Σ_r log u_r(S; B)`` (from
  ``RoundLogU``); ``prior_b`` maps it to ``P(num_bad)``. Returns
  ``(p_bad, p_num_bad, p_set_by_b)``:

    P(num_bad | D) ∝ P(num_bad) · (1/C(N,num_bad)) · Σ_S exp(Σ_r log u_r(S; B))
    P(S | num_bad, D) ∝ exp(Σ_r log u_r(S; B))
    P(player i bad) = Σ_B P(num_bad | D) · Σ_{S ∋ i} P(S | num_bad, D)

  All sums are in log-space (logsumexp/softmax) to avoid underflow. The ``1/C(N,B)``
  prior over which subset is bad is essential — a larger ``B`` spreads its prior over more
  subsets. Falls back to ``prior_b`` if every candidate is ruled out (fully degenerate).
  """
  num_players = next(iter(log_u_by_b.values())).shape[0]
  bs = list(log_u_by_b)
  log_ev = {}
  p_set = {}
  for b in bs:
    log_u = log_u_by_b[b]
    mask = _badset_mask(log_u.shape)
    vals = log_u[mask]
    if not np.any(np.isfinite(vals)):  # this bad-count ruled out entirely
      log_ev[b] = -np.inf
      p_set[b] = (mask / mask.sum())
      continue
    m = vals[np.isfinite(vals)].max()
    lse = m + np.log(np.sum(np.exp(vals - m)))  # Σ_S exp(Σ log u); exp(-inf)=0
    log_ev[b] = np.log(prior_b[b]) - np.log(uf.C(b, num_players)) + lse
    w = np.zeros(log_u.shape)
    w[mask] = np.exp(vals - m)
    p_set[b] = w / w.sum()
  evs = np.array([log_ev[b] for b in bs])
  if not np.any(np.isfinite(evs)):  # everything degenerate: fall back to the prior
    p_num_bad = {b: prior_b[b] for b in bs}
  else:
    mm = evs[np.isfinite(evs)].max()
    ww = np.where(np.isfinite(evs), np.exp(evs - mm), 0.0)
    ww = ww / ww.sum()
    p_num_bad = {bs[k]: float(ww[k]) for k in range(len(bs))}
  p_bad = np.zeros(num_players)
  for b in bs:
    p_bad += p_num_bad[b] * DeMatrix(p_set[b])
  return p_bad, p_num_bad, p_set


# --- simulation + interactive play ---------------------------------------------

def DistributeWires(num_players, hand_size, active_wires, num_bom):
  """Deal ``num_bom`` bombs to random hands then ``active_wires`` wires uniformly among
  the remaining non-bomb **slots** — the multivariate-hypergeometric deal the inference
  assumes (model.md §3.1/§3.4.1). Returns ``(wires, bombs)`` integer vectors.

  This must place wires uniformly over *slots*, not over *players*: a "pick a random
  player, add a wire if under capacity" loop is uniform over players and only matches
  the slot-uniform model when every hand has equal capacity. The bomb hand has one fewer
  slot, so the player-uniform shortcut over-deals wires to it and makes the simulated
  games diverge from the model — which shows up as a systematic `P(bad)` miscalibration
  (see ``Calibration.py``). Simulation-only helper for ``PlayAuto``.
  """
  bombs = np.zeros(num_players, dtype=int)
  for h in sample(range(num_players), num_bom):
    bombs[h] = 1
  slots = [(g, s) for g in range(num_players) for s in range(hand_size - int(bombs[g]))]
  wires = np.zeros(num_players, dtype=int)
  for (g, _s) in sample(slots, int(active_wires)):  # uniform over non-bomb slots
    wires[g] += 1
  return wires, bombs


def CutRandom(decls, probs, revealed, found, hand_size, active_wires, num_bad, num_bom):
  """Cut a uniformly random face-down card (ignores the belief). The baseline policy
  for the ``PlayAuto`` simulation; the quantities-only panel (``CutPanel``) is the
  assistant's recommendation substrate, not an autonomous policy (ADR 0006)."""
  num_players = revealed.size
  cutee = randrange(num_players)
  while revealed[cutee] >= hand_size:
    cutee = randrange(num_players)
  return cutee


def PlayAuto(num_players=4, initial_hand_size=5, verbosity=0, cut_strategy=CutRandom):
  """Simulate one full game with the joint num_bad model, tracking the belief.

  Samples the true bad count from ``NUM_BAD_PRIOR`` and a random bad set, deals the bomb
  and wires each round under the uniform-lie model, and cuts via ``cut_strategy`` until
  the good guys clear every wire, the bomb is cut, or time runs out. Accumulates
  ``RoundLogU`` per candidate num_bad and reduces to the joint belief each round
  (``JointBadBelief``). Returns ``(good_guys_won, p_bad, roles, p_num_bad)``.
  """
  prior_b = NUM_BAD_PRIOR(num_players)
  candidate_bs = list(prior_b)
  true_num_bad = candidate_bs[0] if len(candidate_bs) == 1 else \
      np.random.choice(candidate_bs, p=[prior_b[b] for b in candidate_bs])
  roles = np.zeros(num_players, dtype=int)
  for i in sample(range(num_players), int(true_num_bad)):
    roles[i] = 1
  if verbosity > 0:
    print("Roles:", roles, " true num_bad:", int(true_num_bad))
  log_u_by_b = {b: np.zeros([num_players] * b) for b in candidate_bs}
  num_bom = 1
  hand_size = initial_hand_size
  active_wires = num_players
  p_bad = np.full(num_players, sum(b * prior_b[b] for b in candidate_bs) / num_players)
  while hand_size > 1:
    total_active = active_wires
    wires, bombs = DistributeWires(num_players, hand_size, active_wires, num_bom)
    declarations = wires.copy()
    for i in range(num_players):
      if roles[i] == 1 or bombs[i] == 1:  # bad or bomb-holder => uniform lie
        declarations[i] = randint(0, hand_size)
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    bomb_cut = False
    for _ in range(num_players):
      probs = ProbDeclaration(declarations, hand_size, total_active, candidate_bs[0], num_bom)
      cutee = cut_strategy(declarations, probs, revealed, found, hand_size,
                           active_wires, candidate_bs[0], num_bom)
      randy = randint(1, hand_size - revealed[cutee])
      if bombs[cutee] == 1 and randy == hand_size - revealed[cutee]:
        bomb_cut = True  # game over; do NOT fold this cut into the role belief --
        break            # its likelihood conditions on "no bomb drawn", now false
      if randy <= wires[cutee] - found[cutee]:
        found[cutee] += 1
        active_wires -= 1
      revealed[cutee] += 1
      if active_wires <= 0:
        break
    # Fold this round's evidence into the joint belief (ADR 0008).
    for b in candidate_bs:
      log_u_by_b[b] = log_u_by_b[b] + RoundLogU(declarations, revealed, found,
                                                hand_size, total_active, active_wires, b, num_bom)
    p_bad, p_num_bad, _ = JointBadBelief(log_u_by_b, prior_b)
    if verbosity > 1:
      print("  P(bad):    ", np.round(p_bad, 3))
      print("  P(num_bad):", {b: round(p_num_bad[b], 3) for b in candidate_bs})
    if bomb_cut:
      if verbosity > 0:
        print("The Bomb was detonated. Bad guys win!")
      return (0, p_bad, roles, p_num_bad)
    if active_wires <= 0:
      if verbosity > 0:
        print("Good guys win!")
      return (1, p_bad, roles, p_num_bad)
    hand_size -= 1
  if verbosity > 0:
    print("Out of time. Bad guys win!")
  return (0, p_bad, roles, JointBadBelief(log_u_by_b, prior_b)[1])


def Play(players=["Alice", "Bob", "Clara", "Darryl"], initial_hand_size=5):
  """Interactive assistant for a real game: prompts for declarations and cut results and
  prints the joint belief (per-player P(bad), P(num_bad), P(bomb)) and the cut panel
  after each event. Handles the player counts where the bad count is uncertain
  (N=4, N=7) via the joint num_bad model (ADR 0008)."""
  num_players = len(players)
  prior_b = NUM_BAD_PRIOR(num_players)
  candidate_bs = list(prior_b)
  num_bom = 1
  log_u_by_b = {b: np.zeros([num_players] * b) for b in candidate_bs}
  hand_size = initial_hand_size
  active_wires = num_players
  while hand_size > 1:
    print("\n\n Round", initial_hand_size - hand_size + 1)
    total_active = active_wires
    declarations = np.zeros(num_players, dtype=int)
    for i in range(num_players):
      declarations[i] = int(input("How many wires does " + players[i] + " say they have? "))
    found = np.zeros(num_players, dtype=int)
    revealed = np.zeros(num_players, dtype=int)
    for cut in range(num_players):
      probs = ProbDeclaration(declarations, hand_size, total_active, candidate_bs[0], num_bom)
      probs = ProbCut(declarations, probs, revealed, found, hand_size, active_wires,
                      candidate_bs[0], num_bom)
      # joint belief readout (this round folded in provisionally for display)
      provisional = dict(log_u_by_b)
      for b in candidate_bs:
        provisional[b] = log_u_by_b[b] + RoundLogU(declarations, revealed, found,
                                                   hand_size, total_active, active_wires, b, num_bom)
      p_bad, p_num_bad, _ = JointBadBelief(provisional, prior_b)
      _, p_bomb = Separate(probs, candidate_bs[0], num_bom)
      print(" P(bad):    ", np.round(p_bad, 3))
      print(" P(num_bad):", {b: round(p_num_bad[b], 3) for b in candidate_bs})
      print(" P(bomb):   ", np.round(np.asarray(p_bomb).reshape(-1), 3))
      PrintPanel(players, declarations, probs, revealed, found, hand_size,
                 active_wires, candidate_bs[0], num_bom)
      cutee_str = input("\nWhose wire has been cut? ")
      while cutee_str not in players:
        cutee_str = input("You must have made a typo. Who? ")
      cutee = players.index(cutee_str)
      revealed[cutee] += 1
      shown = int(input("Did you reveal:\n 0- an inactive wire\n 1- an active wire\n 2- the bomb\n"))
      while shown not in (0, 1, 2):
        shown = int(input("Sorry, I'm looking for a 0, a 1 or a 2 here. "))
      if shown == 2:
        print("The Bomb was detonated. Bad guys win!")
        return
      if shown == 1:
        found[cutee] += 1
        active_wires -= 1
      if active_wires <= 0:
        print("All wires have been cut. Good guys win!")
        return
    for b in candidate_bs:
      log_u_by_b[b] = log_u_by_b[b] + RoundLogU(declarations, revealed, found,
                                                hand_size, total_active, active_wires, b, num_bom)
    hand_size -= 1
  print("Out of time. Bad guys win!")
