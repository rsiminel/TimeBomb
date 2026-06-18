# -*- coding: utf-8 -*-
"""Consistency checks for real-table input (model.md §3.6 — resilience to model-breaking play).

The solver assumes the uniform-lie model holds exactly, but a real table breaks it: a player
miscounts, declares more wires than their hand holds, or reports a cut result that is impossible
given the declarations. The belief math already degrades *safely* — ``ProbDeclaration`` falls
back to a uniform prior when no configuration can explain the declarations, and ``ProbCut``
returns the prior unchanged on an impossible observation — but *silently*, so a miscount throws
away a round of evidence with the displayed panel looking normal.

These helpers let the interactive ``Play`` loops detect the problem and tell the human, on a
**hybrid** policy:

- **Re-prompt** for clearly invalid numeric entry — a declaration outside ``[0, H]``, a
  non-integer, an out-of-range cut-result code, or cutting an already fully-revealed hand
  (``prompt_int``, ``valid_declaration``, ``can_cut``).
- **Warn and continue** for entry that is individually valid but *jointly* impossible under the
  rules — declarations no configuration can explain (``declarations_feasible``), or a cut result
  impossible given the declarations (the caller checks that ``ProbCut`` returned the prior
  object unchanged). The safe fallback still applies; the warning just stops the silent loss.

Pure functions (``prompt_int`` aside), reusable by the future web port (C1).
"""
import numpy as np

WARN = "⚠️ "  # warning sign, prefixing the two diagnostics below

DECL_WARNING = (WARN + "These declarations can't all be true under the rules (likely a "
                "miscount) -- continuing with no declaration evidence this round.")
CUT_WARNING = (WARN + "That result is impossible given the declarations -- belief left "
               "unchanged; check the entry.")


def valid_declaration(value, hand_size):
  """A declared wire count is valid iff it is a whole number in ``[0, hand_size]`` — a hand
  cannot hold fewer than 0 or more than ``hand_size`` wires."""
  try:
    v = int(value)
  except (TypeError, ValueError):
    return False
  return v == value and 0 <= v <= hand_size


def can_cut(revealed_i, hand_size):
  """A hand can still be cut iff it has at least one face-down card left."""
  return revealed_i < hand_size


def prompt_int(prompt, lo, hi, _input=input):
  """Prompt until the reply parses as an integer in ``[lo, hi]``, re-asking otherwise. The
  interactive guard for declarations (``lo=0, hi=H``) and cut-result codes (the module's
  allowed set); it also replaces the bare ``int(input(...))`` that crashes on non-integer
  input. ``_input`` is injectable so the loop can be unit-tested without real stdin."""
  while True:
    raw = _input(prompt)
    try:
      value = int(raw)
    except (TypeError, ValueError):
      print("  Please enter a whole number between %d and %d." % (lo, hi))
      continue
    if lo <= value <= hi:
      return value
    print("  That must be between %d and %d." % (lo, hi))


def declarations_feasible(decls, hand_size, active_wires, num_bad, num_bom):
  """Can *any* configuration explain these declarations under the uniform-lie model?

  True iff the declaration prior has positive mass before normalisation — i.e.
  ``ProbDeclaration`` would *not* hit its uniform degeneracy fallback. Computed from the
  canonical ``General._decl_weights``: the sum of the unnormalised config weights is
  convention-independent, so this single predicate is correct for every variant (each is a
  fixed ``(num_bad, num_bom)`` projection of General). A ``False`` means the declarations are
  jointly impossible (e.g. a miscount), even when each is individually in ``[0, H]``.

  Using General as the canonical *model* here is fine — the CLAUDE.md rule against oracling on
  General is about *tests*, not production diagnostics. Imported lazily to avoid a circular
  import (``General.Play`` imports this module)."""
  import General as gen
  weights = gen._decl_weights(np.asarray(decls), hand_size, active_wires, num_bad, num_bom)
  return float(np.sum(weights)) > 0.0
