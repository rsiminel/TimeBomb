# ADR 0008: Joint inference over the number of bad guys

- **Status:** Accepted
- **Date:** 2026-06-17
- **Scope:** `General.py` (the `Play`/`PlayAuto` orchestration and a new joint-belief
  helper); model.md §3.6 → a new §3.5.1; roadmap.md; TODO.md
- **Builds on** [0007](0007-declaration-lie-count-factor.md) (the absolute,
  cross-hypothesis declaration likelihood) and [0005](0005-cross-round-evidence-combination.md)
  (per-round product; `P(bomb)` never combined).

## Context

For most player counts the number of bad guys `B` is fixed and known (N=5,6 → 2; N=8 →
3), but for **N=4** it is 1 or 2 and for **N=7** it is 2 or 3 — the role-card deal leaves
`B` itself uncertain, with a known prior `P(B)` from the deck composition. The old
`General.py` handled this with a **fixed-weight mixture** (`pos_bad = [[1, 2/5], [2, 3/5]]`
for N=4): it kept one config tensor per candidate `B`, combined their `P(bad)` readouts
with constant weights, and **never updated those weights from evidence**. That is a
modelling gap — the declarations and cuts are informative about *how many* bad guys there
are, not just which, and a confident single-bad-guy round should shift mass toward `B = 1`.

## Options considered

- **A — Fixed-weight mixture (status quo).** Simple, but discards real evidence about `B`;
  the reported `P(bad)` is an arbitrary blend rather than a posterior.
- **B — Joint Bayes over `(B, S, h)` (chosen).** Treat the bad count as another hidden
  variable and let the same evidence update it. Roles `(B, S)` are fixed for the game; the
  bomb holder `h` is per-round (§3.1, ADR 0004).

## Decision

Infer the joint posterior over `(B, S, h)` and marginalise. With rounds conditionally
independent given the fixed roles (redeal, ADR 0005):

```
P(B, S | D₁..D_R)  ∝  P(B) · (1 / C(N, B)) · Π_r u_r(S; B)
```

where `P(S | B) = 1/C(N, B)` is the uniform prior over which `B`-subset is bad, and

```
u_r(S; B)  =  Σ_h  ŵ_decl(S, h) · L_config(S, h ; round-r final cuts)
```

is the round's **unnormalised** bad-set marginal: the bomb holder `h` summed out of the
per-configuration weight, declaration prior × cut likelihood. `ŵ_decl` is `ProbDeclaration`'s
*unnormalised* weight `(H+1)^{−|F|}·C(free_slots, t_free)/Π_{g∈F} C(H, decls[g])` — and it is
exactly the **lie-count factor of [ADR 0007]** that makes `u_r` an *absolute*, cross-`B`
comparable quantity. The per-round, `B`-independent constants dropped from it
(`Π_all C(H, decls) / C(N·H−1, A)`, the `1/N` bomb prior) are common to every `B` and cancel
in the `B`-posterior.

Read-outs:

```
P(S | B, D)  ∝  Π_r u_r(S; B)                      (CombineProbs, per B)
P(B | D)     ∝  P(B) · (1/C(N,B)) · Σ_S Π_r u_r(S; B)
P(player i bad)  =  Σ_B P(B | D) · Σ_{S ∋ i} P(S | B, D)
```

Two load-bearing details:

- **The `1/C(N, B)` prior is essential, not cosmetic.** A larger `B` spreads its prior over
  `C(N, B)` more subsets, each individually less likely a priori; without the `1/C(N, B)`
  the higher bad-count would be systematically over-favoured.
- **Computed in log-space.** `Σ_r log u_r(S; B)` accumulated per `B`, then `logsumexp` over
  `S` for the evidence and `softmax` over `B` for `P(B | D)` — the unnormalised products
  underflow otherwise (the same reason `CombineProbs` is log-space, ADR 0005).

`P(bomb)` stays per-round (ADR 0004/0005): it is read from a single round's joint and never
enters the cross-round `u_r` product.

## Consequences

- `General.py` carries a small joint-belief helper over the list of per-`B` tensors;
  `Play`/`PlayAuto` replace the fixed `pos_bad` weights with the evidence-updated `P(B | D)`.
- The cut panel and `P(bad)` read-out become posterior-weighted over `B`. For the
  fixed-`B` player counts (one candidate `B`) the scheme collapses exactly to the
  single-`B` pipeline, so those are unchanged.
- Validated by a generative oracle/Monte Carlo that samples `B` from its prior too (not
  just `S, h`): the inferred `P(B | D)` and per-player `P(bad)` must track the true
  generative posterior.
- This is the modelling step ADR 0007 was a prerequisite for; with the absolute likelihood
  in hand it is a marginalisation, no new behavioural assumption.
