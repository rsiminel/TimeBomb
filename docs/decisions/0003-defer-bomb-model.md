# ADR 0003: Defer the bomb sub-model to the `*OneBomb` variants

- **Status:** Accepted
- **Date:** 2026-06-16
- **Scope:** model.md §1, §3.7; TODO.md Axis A4

## Context

The project's stated goal (model.md §1) includes a `P(bomb)` readout, and the
behavioural model (§2) says bomb-holders declare differently (a good guy with the bomb
under-declares; a bad guy with the bomb over-declares). But the current focus is the two
**no-bomb** variants (`OneBadGuyNoBomb`, `TwoBadGuysNoBomb`), for which the bomb
likelihoods are not needed. Specifying them now would be modelling ahead of demand.

## Options considered

- **Defer** the bomb declaration likelihood, the bomb cut likelihood, and the `P(bomb)`
  readout until the first bomb variant (`OneBadGuyOneBomb`, `B=1, M=1`).
- **Pin it down now** in the same modelling pass.

## Decision

**Defer.** The bomb sub-model is left as the single explicitly-open modelling gap
(model.md §3.7) and tracked as TODO.md Axis A4, to be settled when the bomb variants
begin.

## Consequences

- The no-bomb pipeline is unblocked without carrying an unused, unvalidated sub-model.
- model.md §1 advertises a `P(bomb)` output that no section yet derives; §3.7 records
  this as deliberate, not an oversight.
