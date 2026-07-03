"""Solver-driven bot: panel adapter + policy layer (specs/002-host-local-game, R5).

Constitution I: every probability here comes from ``General.py``; this module only
ranks, samples, and thresholds those numbers -- it computes none of its own. The panel
adapter mirrors ``web/replay.py``'s belief-folding pattern (``General.Play``'s
reference orchestration: fold every completed round via ``RoundLogU``, then
provisionally fold the round in progress) but works from a live ``PublicState``
snapshot instead of a replayed event log, so it lives in ``tbgame/`` rather than
depending on ``web/`` (the wrong direction -- web depends on the engine, not vice
versa).
"""
import random as _random

import numpy as np

import General as tb

from .base import Agent
from ..state import legal_targets

INITIAL_HAND_SIZE = 5
# Lookahead depth for the info stat: the bot only needs pSafe/pBomb to decide, so a
# shallow depth keeps every decision well under the 2 s budget (SC-002) regardless of
# player count, unlike v1's per-player-count DEPTH_CAP (which also serves the display
# "horizon" stat's accuracy -- not a concern for policy-only use).
_MAX_DEPTH = 0


def _round_state(pub, r):
  """Reconstruct round ``r``'s hand_size/revealed/found from PublicState's cumulative
  fields (cut_log carries every cut ever made, tagged by round) -- everything
  ``RoundLogU`` needs to fold that round's evidence, without the adapter holding any
  state of its own."""
  n = pub.num_players
  hand_size = INITIAL_HAND_SIZE - r
  revealed = np.zeros(n, dtype=int)
  found = np.zeros(n, dtype=int)
  for c in pub.cut_log:
    if c["round"] == r:
      revealed[c["target"]] += 1
      if c["result"] == "wire":
        found[c["target"]] += 1
  return hand_size, revealed, found


def panel_from_public(pub):
  """``PublicState`` -> ``{"pBad": [...], "panel": [...] | None}`` -- per-seat P(bad)
  and, once this round has declarations, the four-stat cut panel (P(safe), P(bomb),
  one-ply/horizon info) for seats with a face-down card. ``General.py`` calls only."""
  n = pub.num_players
  prior_b = pub.num_bad_prior
  candidate_bs = list(prior_b)
  num_bom = pub.num_bom
  log_u_by_b = {b: np.zeros([n] * b) for b in candidate_bs}

  active = n   # round 0 always starts with active_wires == num_players (a fixed rule)
  for r in range(pub.round_index):
    if r >= len(pub.declaration_history):
      break
    decls = np.array(pub.declaration_history[r], dtype=int)
    hand_size, revealed, found = _round_state(pub, r)
    round_start_active = active
    active -= int(np.sum(found))
    for b in candidate_bs:
      log_u_by_b[b] = log_u_by_b[b] + tb.RoundLogU(
          decls, revealed, found, hand_size, round_start_active, active, b, num_bom)

  declared = pub.declarations
  if declared is None or all(d is None for d in declared):
    p_bad, _, _ = tb.JointBadBelief(log_u_by_b, prior_b)
    return {"pBad": [float(p) for p in p_bad], "panel": None}

  decls = np.array([0 if d is None else d for d in declared], dtype=int)
  revealed, found = np.array(pub.revealed), np.array(pub.found)
  b0 = candidate_bs[0]
  probs = tb.ProbDeclaration(decls, pub.hand_size, pub.round_start_active, b0, num_bom)
  probs = tb.ProbCut(decls, probs, revealed, found, pub.hand_size, pub.active_wires,
                     b0, num_bom)
  provisional = {}
  for b in candidate_bs:
    provisional[b] = log_u_by_b[b] + tb.RoundLogU(
        decls, revealed, found, pub.hand_size, pub.round_start_active, pub.active_wires,
        b, num_bom)
  p_bad, _, _ = tb.JointBadBelief(provisional, prior_b)
  panel = tb.CutPanel(decls, probs, revealed, found, pub.hand_size, pub.active_wires,
                      b0, num_bom, max_depth=_MAX_DEPTH)
  rows = []
  for i in range(n):
    if np.all(np.isnan(panel[i])):
      rows.append(None)
    else:
      p_safe, p_bomb, one_ply, horizon = panel[i]
      rows.append({"pSafe": float(p_safe), "pBomb": float(p_bomb),
                   "onePly": float(one_ply), "horizon": float(horizon)})
  return {"pBad": [float(p) for p in p_bad], "panel": rows}


class SolverBot(Agent):
  """Decisions are policy rules over the shared solver's numbers plus this seat's own
  ``PrivateView`` -- never a probability computed outside ``General.py`` (research R5).
  Deception ability is FR-016; the quality bar is "beats random" (SC-006), not "beats
  the arena LLMs"."""

  name = "solver_bot"

  def __init__(self, panel_adapter=panel_from_public, rng=None):
    self._panel_adapter = panel_adapter
    self._rng = rng or _random.Random()   # own, unseeded RNG (research R2)

  def declare(self, view):
    pub, priv = view.public, view.private
    if priv.my_role == 0:
      return priv.my_wires
    offset = self._rng.choice([-2, -1, 1, 2])
    return max(0, min(pub.hand_size, priv.my_wires + offset))

  def choose_cut(self, view):
    pub, priv = view.public, view.private
    legal = legal_targets(pub, priv.my_index)
    if not legal:
      return None
    rows = self._panel_adapter(pub).get("panel")
    if not rows:
      return self._rng.choice(legal)
    reverse = priv.my_role == 0   # good: chase high P(safe); bad: chase low P(safe)
    scored = sorted(legal, key=lambda j: (rows[j] or {}).get("pSafe", 0.0), reverse=reverse)
    # Sample among the (at most two) best-ranked targets -- enough to keep this
    # non-deterministic without diluting the panel's signal away (research R5).
    return self._rng.choice(scored[:min(2, len(scored))])

  def claim(self, view):
    pub, priv = view.public, view.private
    p_bad = self._panel_adapter(pub).get("pBad")
    if not p_bad:
      return None
    ranked = sorted((j for j in range(pub.num_players) if j != priv.my_index),
                    key=lambda j: p_bad[j], reverse=True)
    if not ranked:
      return None
    suspect = ranked[0]
    if priv.my_role == 0 and p_bad[suspect] > 0.6:
      return {"kind": "distrust", "target": suspect}
    if priv.my_role == 1:
      return {"kind": "self_honest", "target": None}
    return None
