# ADR 0004: Uniform-lie bomb model for the `*OneBomb` variants

- **Status:** Accepted
- **Date:** 2026-06-16
- **Scope:** model.md §2, §3.1, §3.8; roadmap.md; TODO.md Axis A4
- **Settles** the bomb sub-model that [0003](0003-defer-bomb-model.md) deferred.

## Context

The bomb variants (`OneBadGuyOneBomb`, `B=1, M=1`) introduce a second hidden variable
— which player holds the bomb this round — and need three things specified: how a
bomb-holder declares, how a cut updates beliefs when the bomb occupies a card slot, and
how `P(bomb)` is read out and carried (or not) across rounds. The behavioural model
inherited from model.md §2 assumed a *strategic* bomb-holder (good-with-bomb
under-declares to deter cuts; bad-with-bomb over-declares to bait them), but that imposes
`decls ≤ wires` / `decls ≥ wires` constraints that can yield dead-end null observations
and is more complex to implement and validate.

## Options considered

**Bomb-holder declaration model:**
- **A — Strategic uniform:** good+bomb ~ `Uniform{0..wires}`, bad+bomb ~
  `Uniform{wires..H}`. Strongest declaration signal for `P(bomb)`; matches the original
  §2 story; assumes opponents play that strategy and risks null observations.
- **B — Uniform-lie (chosen):** a player declares truthfully iff good *and* bomb-free;
  every other player (any bad guy, or a good guy holding the bomb) declares
  `Uniform{0..H}`. Simplest; no `≤`/`≥` constraints; still couples bad/bomb because a
  good guy forced to lie is evidence for "bad *or* bomb".
- **C — Parametric:** a tunable under/over-declare bias to fit later. Far-future.

**Bomb persistence:** per-round re-deal vs. fixed-for-game.

**Cut recommendation:** expected-value tradeoff vs. hard-avoid threshold vs. defer.

## Decision

- **Declaration model: B, the uniform-lie bomb model.** Truthful iff good and
  bomb-free; otherwise `Uniform{0..H}`. The strategic model (A) is the eventual target,
  to be A/B-tested against B once B is trusted; the parametric model (C) is a far-future
  research item. Both are recorded in model.md §3.7 and roadmap.md.
- **Persistence: per-round re-deal.** Roles are fixed for the game, but wires and the
  bomb are re-dealt each round. So `P(bad)` accumulates across rounds via `CombineProbs`
  while `P(bomb)` is read from the current round only and **never** combined.
- **Cut strategy: deferred.** Axis A4 covers the probability model only (declaration
  likelihood, cut likelihood, `P(bomb)` readout). The risk-aware cut strategy is a
  separate later task.

## Consequences

- model.md gains §3.8 with the closed-form declaration prior
  `C(2H−1, …)/(C(H,d_b)·C(H,d_h))` (and `C(H−1, …)/C(H,d_b)` on the `b=h` diagonal),
  the bomb-as-must-not-draw cut likelihood, and the readout/persistence rules. §2 is
  rewritten to the uniform-lie bomb model.
- The bomb occupying a slot (`H−1` wire-able slots in the bomb hand) is load-bearing in
  both the prior and the cut likelihood — it changes the maximum wires a hand can hold
  and the feasible outcomes of a cut, so it cannot be ignored.
- `CombineProbs` must marginalise out the bomb before combining rounds; mixing the bomb
  into the cross-round product would be a bug.
- A reusable comparison hook: once the uniform-lie variant is validated, benchmark it
  against the strategic model (A) on bad-guy and bomb identification accuracy.
