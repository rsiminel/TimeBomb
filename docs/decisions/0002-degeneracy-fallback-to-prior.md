# ADR 0002: Fall back to the prior/uniform on a zero marginal

- **Status:** Accepted
- **Date:** 2026-06-16
- **Scope:** `ProbDeclaration` and `ProbCut` in every variant; model.md §3.3, §3.4

## Context

An observation can be impossible under *every* surviving hypothesis, driving the Bayes
normaliser (the marginal) to zero. The two functions handled this inconsistently:
`ProbCut` returned the incoming prior unchanged, while `ProbDeclaration` returned an
all-zeros vector — an unnormalisable non-distribution that violates the "always return
a valid probability distribution" done-bar.

## Options considered

- **Fall back to the prior / uniform.** On a zero marginal, return the incoming prior
  (`ProbCut`) or the uniform prior (`ProbDeclaration`, whose implicit prior is uniform).
  One principle: *do not update on impossible evidence.*
- **Raise an error.** Treat an all-impossible observation as a hard inconsistency and
  fail loudly.

## Decision

**Fall back to the prior/uniform.** Both functions always return a valid distribution;
the rule is identical in spirit (`ProbDeclaration`'s prior just happens to be uniform).

## Consequences

- No `NaN`/all-zeros output; the invariant "output is a probability distribution" holds
  unconditionally.
- The case is unreachable with self-consistent game data, so this is purely defensive; a
  loud failure was judged less useful than graceful degradation for an assistant.
- `ProbDeclaration`'s all-zeros branch is to be replaced (TODO.md Axis A3).
