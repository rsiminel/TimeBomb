# ADR 0005: Cross-round evidence combination (`CombineProbs`)

- **Status:** Accepted
- **Date:** 2026-06-16
- **Scope:** `CombineProbs` in every variant; model.md §3.8.3; roadmap.md; TODO.md Axis A5
- **Builds on** [0004](0004-uniform-lie-bomb-model.md) (persistence: `P(bad)` accumulates,
  `P(bomb)` is per-round).

## Context

Roles are fixed for the whole game, but wires and the bomb are **re-dealt
independently each round** (model.md §3.1). Each round yields a per-round belief about
the fixed bad guy(s). We need to specify how those per-round beliefs combine into one
accumulated `P(bad)` — and to justify it, because it is the one cross-round operation
every variant shares and it had never been written down.

`CombineProbs` currently multiplies the per-round `P(bad)` vectors elementwise and
renormalises. The question raised: is that the right model, or should it (a) be more
robust, and (b) weight rounds by how informative they are?

## Options considered

**Combination rule:**
- **A — Elementwise product + renormalise (chosen).** Treat rounds as conditionally
  independent evidence about the fixed roles; multiply and renormalise.
- **B — Average / mixture of per-round posteriors.** Strictly less informative; throws
  away the accumulation that independent evidence licenses.
- **C — Recency or hand-tuned weighting** `∏ᵣ P(·|Dᵣ)^{wᵣ}`. A tempering exponent;
  departs from exact Bayes (see "weighting" below).

**Robustness of the product:** raw product vs. ε-floored / log-space accumulation.

**Per-round informativeness weighting:** add a coefficient `wᵣ` vs. rely on the
endogenous vector shape.

## Decision

- **Rule: A, elementwise product + renormalise — and it is *exact*, not a heuristic.**
  The redeal makes the rounds conditionally independent given the role assignment `R`:
  `P(D₁,…,D_R | R) = ∏ᵣ P(Dᵣ | R)`. Each round the module computes `P(bad=i | Dᵣ)`
  under a **uniform per-round role prior**, so the per-round vector is proportional to
  the per-round likelihood `L(Dᵣ | i)`. Multiplying and renormalising therefore yields
  `∏ᵣ L(Dᵣ | i)`, which — with a uniform game-start prior — is the exact posterior
  `P(R | all data)`. No alternative pooling improves on this under the model.

- **The per-round factor must stay a likelihood, not a prior-contaminated posterior.**
  `ProbCut` takes **that round's** `ProbDeclaration` output as its prior (never the
  accumulated cross-round belief), and the within-round cut refinement *replaces* the
  round's slot rather than chaining. If the accumulated belief were fed back in *and*
  the result multiplied, the prior would be counted once per round (`P(i)^R`). It
  cancels today only because the prior is uniform; the "fresh each round, combine by
  product" split keeps it correct if a non-uniform role prior is ever introduced.

- **`P(bomb)` is never combined.** The same redeal that licenses the product for the
  fixed roles *forbids* it for the per-round bomb holder (§3.8.3, ADR 0004).

- **Informativeness is not weighted, by design.** A round's discriminating power is the
  slope of `log L(Dᵣ | i)` across `i`: an uninformative round yields a near-uniform
  vector that is ≈ a no-op in the product, while an informative one yields a peaked
  vector that reshapes it — *automatically*, scaled by how far it departs from uniform
  (its KL-from-uniform). Combinatorial shifts over the game (shrinking `active_wires`
  and `hand_size` make late cuts more diagnostic) are likewise already carried by the
  likelihood. An external exponent `wᵣ` added to encode informativeness would
  **double-count** the slope that already self-scales, and is rejected.

- **A per-round weight encodes *distrust*, not information — deferred.** `wᵣ < 1` is a
  tempering exponent, justified only when a round's likelihood is *overconfident
  relative to reality* (model misspecification). The most suspect rounds are
  declaration-dominated ones (few/no cuts), which lean hardest on the uniform-lie
  idealisation rather than the exact hypergeometric. The principled fix for that is a
  better lie model, not a free coefficient. Per-round tempering is deferred and gated on
  an actual **calibration** finding; even then a single global temper (≈ the ε-floor
  below) is preferred before a per-round one.

## Consequences

- **Two robustness hardenings are warranted (tracked, not yet implemented):**
  - **ε-floor.** Mix each per-round vector with `ε · uniform` before multiplying, so a
    single round's hard `0` cannot *permanently* eliminate a player. A genuine `0` means
    "impossible observation"; under an idealised lie model a real game can produce an
    observation the model calls impossible, and one such round is currently
    unrecoverable.
  - **Log-space accumulation.** Sum `log` per-round vectors and softmax-normalise to
    avoid underflow over many rounds / large `N`. Underflow currently trips the
    `total == 0` branch, which dumps to uniform and silently discards real evidence; in
    log space, true contradiction essentially cannot arise from consistent observations,
    so that fallback stops being a correctness sink. The ε-floor is trivial to express
    in this form.
- These harden the *same* computation; they do not change what it computes. Land them in
  `General.py` (largest `N`, likeliest home of a future non-uniform lie model) rather
  than retrofitting the four pinned variants.
- Per-round informativeness weighting is explicitly **out of scope** unless a calibration
  test demonstrates systematic overconfidence; recorded here so it is not re-litigated.
