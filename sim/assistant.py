"""Public-info assistant readout for arena agents (the ``<assistant_readout>`` block).

Adapter from the arena's ``PublicState`` to the solver's belief, in exactly the
orchestration pattern of ``web/replay.py`` (itself ``General.Play``'s): fold each
completed round's evidence with ``RoundLogU``, fold the in-progress round in
provisionally, read P(bad) / P(num_bad) from ``JointBadBelief`` and the per-player
next-cut wire/bomb chances from ``P_wire`` / ``Separate``. Zero probability formulas
live here -- every number is a raw ``General.py`` output, reshaped and rounded. The
web assistant's lookahead entropy stats are deliberately left out: they exist to feed
a human's risk appetite, and here they would only spend tokens.

The readout uses the PUBLIC record only (declarations + cut results), so it is the
same panel every player could compute at a real table -- showing it to everyone, bad
guys included, adds inference power but leaks nothing. A player's own private
knowledge (their role, their hand) is theirs to fold on top.
"""

import numpy as np

import General as gen


class PanelAssistant:
  """``panel(pub)`` -> the readout dict rendered into ``<assistant_readout>``, or None
  when there is no public evidence yet (the first round's blind declaration phase).
  Stateless between calls except for a per-position memo (the engine builds N identical
  views per decision point; the readout is the same for all of them)."""

  def __init__(self):
    self._cache = {}

  def panel(self, pub):
    # Everything the readout depends on, cheap to key on. Declarations are fixed within
    # a round once made, so the None-mask (blind vs sighted phase) covers them.
    key = (pub.round_index, tuple(d is None for d in pub.declarations),
           tuple(pub.revealed), tuple(pub.found), len(pub.cut_log))
    if key not in self._cache:
      self._cache[key] = self._compute(pub)
    return self._cache[key]

  # -- the General.Play / web-replay call pattern ----------------------------

  def _compute(self, pub):
    n, names = pub.num_players, pub.player_names
    prior_b = dict(pub.num_bad_prior)
    bs = list(prior_b)
    num_bom = pub.num_bom

    # Fold every completed round's evidence (ADR 0008), reconstructing each round's
    # final cut state from the public cut log. Wire totals: one wire per player at
    # game start, minus one per wire found.
    log_u = {b: np.zeros([n] * b) for b in bs}
    active = n
    for r, decls in enumerate(pub.declaration_history):
      hand = pub.hand_size + (pub.round_index - r)
      revealed, found = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
      for c in pub.cut_log:
        if c["round"] == r:
          revealed[c["target"]] += 1
          found[c["target"]] += c["result"] == "wire"
      d = np.asarray(decls, dtype=int)
      total = active
      active -= int(found.sum())
      for b in bs:
        log_u[b] = log_u[b] + gen.RoundLogU(d, revealed, found, hand, total, active,
                                            b, num_bom)

    if any(v is None for v in pub.declarations):
      # Blind declaration phase: only the completed rounds speak.
      if not pub.declaration_history:
        return None
      p_bad, p_num_bad, _ = gen.JointBadBelief(log_u, prior_b)
      return self._readout(names, p_bad, p_num_bad, bs, cut_rows=None)

    # A round in progress: fold it provisionally and compute the cut stats on its live
    # posterior, conditioning the panel on the leading num_bad candidate as the web
    # assistant does (exact whenever the role deal fixes the count, e.g. 6 players).
    d = np.asarray(pub.declarations, dtype=int)
    revealed = np.asarray(pub.revealed, dtype=int)
    found = np.asarray(pub.found, dtype=int)
    b0 = bs[0]
    probs = gen.ProbDeclaration(d, pub.hand_size, pub.round_start_active, b0, num_bom)
    probs = gen.ProbCut(d, probs, revealed, found, pub.hand_size, pub.active_wires,
                        b0, num_bom)
    provisional = {b: log_u[b] + gen.RoundLogU(d, revealed, found, pub.hand_size,
                                               pub.round_start_active, pub.active_wires,
                                               b, num_bom)
                   for b in bs}
    p_bad, p_num_bad, _ = gen.JointBadBelief(provisional, prior_b)

    p_wire = gen.P_wire(d, probs, revealed, found, pub.hand_size, pub.active_wires,
                        b0, num_bom)
    if num_bom:
      _, prob_bom = gen.Separate(probs, b0, num_bom)
      p_bomb = np.asarray(prob_bom).reshape(-1)[:n]
    else:
      p_bomb = np.zeros(n)
    cut_rows = {names[i]: {"P(wire)": _r3(p_wire[i]), "P(bomb)": _r3(p_bomb[i])}
                for i in range(n) if revealed[i] < pub.hand_size}
    return self._readout(names, p_bad, p_num_bad, bs, cut_rows)

  def _readout(self, names, p_bad, p_num_bad, bs, cut_rows):
    out = {"P(bad)": {names[i]: _r3(p_bad[i]) for i in range(len(names))}}
    if len(bs) > 1:                              # only when the deal leaves it uncertain
      out["P(num_bad)"] = {str(b): _r3(p) for b, p in p_num_bad.items()}
    if cut_rows is not None:
      out["next cut"] = cut_rows
    return out


def _r3(x):
  return round(float(x), 3)
