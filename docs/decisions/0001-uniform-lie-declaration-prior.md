# ADR 0001: Uniform-lie joint-Bayes declaration prior

- **Status:** Accepted
- **Date:** 2026-06-16
- **Scope:** `ProbDeclaration` in every variant; model.md §2, §3.3

## Context

`ProbDeclaration` turns the round's wire declarations into a prior over configurations
(who is bad). Doing this by Bayes requires a *generative* model of how a bad guy
chooses what to declare. The existing code instead weighted each candidate liar by a
**card-arrangement count** — `C(decls[i], excess)` / `C(H−decls[i], −excess)` — that
was never derived from a generative model, and model.md §2's informal "lies uniformly
at random" did not actually match it. This is the declaration-side twin of the
Binomial-½ wire-split heuristic already rejected for `ProbCut` (model.md §3.4.1): a
plausible-looking count that is not the posterior.

## Options considered

- **A — Uniform lie.** The bad guy declares uniformly over `{0,…,H}`, independent of
  his true wire count. The lie term is then constant and cancels, leaving a closed form
  driven purely by how deal-plausible each forced configuration is.
- **B — Keep the card-count heuristic.** Retain the existing weights. They equal the
  true posterior only up to a spurious per-candidate factor, so they are wrong in
  general.
- **C — Strategic / parametric lie.** Model the bad guy as preferring small or
  maximally-deceptive lies, with a tunable parameter. More realistic, but introduces a
  free parameter that must be fit and cannot be cleanly validated against a brute force.

## Decision

**Option A.** Under uniform dealing (multivariate-hypergeometric) and a uniform lie,
the prior is the likelihood of the unique deal each configuration forces:

```
B = 1:   P(bad = i | decls)   ∝  C(H, t_i) / C(H, decls[i]),   t_i = decls[i] − excess
B > 1:   P(bad set S | decls) ∝  C(B·H, Σ_S decls − excess) / Π_{b∈S} C(H, decls[b])
```

(the `B > 1` form follows by Vandermonde's identity and reduces to the `B = 1` form
when `|S| = 1`). This makes `ProbDeclaration` the exact joint-Bayes twin of `ProbCut`.
There is **no `excess = 0` special case**: for `B = 1` the formula already yields a
uniform prior when `excess = 0`; for `B > 1`, `excess = 0` is genuinely informative
(two lies can cancel in aggregate) and must not be flattened.

## Consequences

- The declaration prior and the cut update now share one wire-placement law (uniform
  placement, model.md §3.4.1).
- `OneBadGuyNoBomb` — already marked "done" — currently ships heuristic **B**. The two
  disagree (e.g. `N=3, H=3, A=4, decls=[2,2,1]`: A gives `(3/7, 3/7, 1/7)`, B gives
  `(0.4, 0.4, 0.2)`). It will need the swap plus a generative-oracle test, which may
  re-open variant 1 (TODO.md Axis A1).
- Option C is parked as an explicitly **far-future, unscheduled** refinement
  (roadmap.md), to weigh only after the pipeline is correct and trusted.
