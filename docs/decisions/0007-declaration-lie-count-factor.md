# ADR 0007: The declaration prior must keep the lie-count factor `(H+1)^{−|F|}`

- **Status:** Accepted
- **Date:** 2026-06-17
- **Scope:** model.md §3.3; `ProbDeclaration` in `OneBadGuyOneBomb`, `TwoBadGuysOneBomb`,
  and `General`; their declaration oracles; roadmap.md; TODO.md Axis A1
- **Corrects** an omission in the uniform-lie declaration prior of [0001](0001-uniform-lie-declaration-prior.md)
  / [0004](0004-uniform-lie-bomb-model.md); **prerequisite** for joint `num_bad`
  inference ([0008](0008-joint-num-bad-inference.md)).

## Context

The uniform-lie declaration prior (§3.3) marks a configuration `(S, h)`, pins the truthful
hands to their declared counts, and scores the multivariate-hypergeometric probability of
the forced deal:

```
P(config | decls)  ∝  C(free_slots, t_free) / Π_{g ∈ F} C(H, decls[g])
```

The derivation also produces a factor `(H+1)^{−1}` for **each free (liar) hand** — the
probability that hand, declaring uniformly over `{0,…,H}`, announced what it did — i.e.
`(H+1)^{−|F|}` for the `|F| = |S ∪ {h}|` free hands. ADR 0001/0004 dropped it as "a constant
that cancels in normalisation."

That is true **only when `|F|` is constant across the configurations being compared**. It
holds for `M = 0` (`|F| = B` for every bad set, so the factor is global and cancels — the
no-bomb variants B1/B2 are correct as written). It **fails for `M = 1`**: a configuration
where a bad guy holds his own bomb has `|F| = B`, but one where a *good* guy holds the bomb
has `|F| = B + 1` (that good bomb-holder is an extra liar). Dropping the factor then
over-weights every bomb-on-good configuration by `(H+1)` relative to the bomb-on-bad ones.

The bug was invisible because the variants' **brute-force declaration oracles made the same
omission**, so module and oracle agreed with each other while both diverged from the true
generative model.

## Options considered

- **A — Keep dropping it (status quo).** Simplest, but wrong for `M ≥ 1`: it systematically
  mis-ranks where the bomb is (and, after marginalising, perturbs `P(bad)`), and makes the
  per-configuration weights non-comparable across hypotheses — which silently breaks any
  future cross-hypothesis comparison (joint `num_bad`).
- **B — Restore the `(H+1)^{−|F|}` factor (chosen).** One extra factor per configuration,
  exactly as the generative model dictates; reduces to the old formula wherever `|F|` is
  constant, so B1/B2 are untouched.

**Verification.** A generative Monte Carlo (sample `S`, `h`, the slot-uniform deal, and the
uniform lies; condition on the observed declarations) was run as an independent ground
truth. For `N=4, H=3`, the **corrected** formula matches the MC `P(bomb | decls)` to within
sampling noise (< 0.001) while the **uncorrected** formula is off by ~0.016 — far beyond
noise. This is what promoted the omission from "plausibly cancels" to "confirmed bug".

## Decision

Restore the lie-count factor: the declaration prior is

```
P(config | decls)  ∝  (H+1)^{−|F|} · C(free_slots, t_free) / Π_{g ∈ F} C(H, decls[g])
```

with `|F| = |S ∪ BombSet|` the number of free (liar) hands. Apply it in every variant that
can have `M ≥ 1` (`OneBadGuyOneBomb`, `TwoBadGuysOneBomb`, `General`) **and in their
brute-force oracles**, so the oracle is true ground truth rather than a mirror of the
module. The no-bomb variants are left unchanged (the factor provably cancels there).

## Consequences

- B3/B4 are re-opened and re-validated (the playbook permits a foundations fix to re-open a
  "done" variant); their `P(bomb)`/`P(bad)` are now calibrated, and the beats-random
  thresholds are re-baselined to the corrected numbers.
- The per-configuration weights are now **absolute and cross-hypothesis-comparable**, which
  is the property [0008](0008-joint-num-bad-inference.md) relies on to compare different bad
  counts `B` against one another from the same declarations.
- model.md §3.3 is corrected: the factor cancels only for `M = 0`; it is load-bearing for
  `M ≥ 1`.
- A standing lesson for the test suite: an oracle that shares a modelling shortcut with the
  module under test cannot catch that shortcut. Declaration oracles now enumerate the lie
  step explicitly (and, where feasible, are cross-checked against a generative Monte Carlo).
