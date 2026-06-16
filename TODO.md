# TODO

Open work items, split along two axes:

- **Axis A — Foundations:** cross-cutting model work shared by every variant.
- **Axis B — Variant pipeline:** the per-variant cleanup, simplest first.

Stable model docs live in [docs/model.md](docs/model.md); plan, status, and the
foundations priority order in [docs/roadmap.md](docs/roadmap.md).

## ⚠️ `General.py` is NOT ground truth

`General.py` is bug-ridden (it shares the `P_wire` bug and likely others). It is
the eventual unification target, but until it is itself cleaned up it must **never**
be used as a correctness oracle — doing so just launders its bugs into the
variants. The only valid ground truth is an **independent `math.comb` brute force
derived from `docs/model.md`** (as in `test_OneBadGuyNoBomb.py`). Same for
`web/app.py`.

---

## Axis A — Foundations (cross-cutting)

Not owned by any single variant; these touch the shared likelihoods. The **modelling
choices are settled** (see [docs/roadmap.md](docs/roadmap.md#cross-cutting-foundations)
and [model.md §3.3](docs/model.md)); what remains below is implementation and
brute-force validation — no code has been touched yet.

### A1 — Declaration prior (`ProbDeclaration`, model.md §3.3) — *model decided: uniform lie*

- [x] **Lie support fixed:** bad guy declares uniformly over `{0..H}` ⇒ joint-Bayes
      prior `P(bad=i) ∝ C(H, t_i)/C(H, decls[i])` (closed-form `B>1` generalisation).
- [x] **Validate against a generative brute force.** Independent `itertools.product`
      oracle (deal → truthful goods → uniform bad-guy lie) in
      `test_OneBadGuyNoBomb.py` confirms the closed form equals the enumerated
      posterior `P(i bad | decls)`.
- [x] **Re-check `OneBadGuyNoBomb`.** It shipped the old card-count heuristic and
      genuinely diverged (e.g. `[2,2,1]` → `[0.4,0.4,0.2]` vs correct `[3/7,3/7,1/7]`).
      Swapped `ProbDeclaration` to the uniform-lie closed form; generative-oracle test
      added (TDD: red → green).

### A2 — `B > 1` prior and `P_wire` marginal — *model decided: §3.4.1 uniform placement*

- [x] Declaration prior and `P_wire` for `B>1` use the §3.4.1 multivariate-
      hypergeometric model; **no `excess==0` special case** for `B>1`.
- [ ] Implement the closed-form `B>1` declaration prior and the per-hand `P_wire`
      marginal.

### A3 — Degeneracy convention — *decided: fall back to prior/uniform*

- [ ] Apply uniformly: `ProbCut` returns the incoming prior, `ProbDeclaration`
      returns uniform, on a zero marginal. Replace `ProbDeclaration`'s all-zeros.
      *(Done in `OneBadGuyNoBomb`; apply to each remaining variant as it is cleaned.)*

### A3b — Backfill the end-to-end accuracy test on `OneBadGuyNoBomb` — ✅ done

- [x] Added `test_inference_beats_random_chance` to `test_OneBadGuyNoBomb.py` (400
      games, N=5): belief puts ~0.95 on the true bad guy and ~0.96 top-1 accuracy vs
      the 0.20 random baseline. Variant 1 now meets the full done bar.

### A4 — Bomb model — *model decided: uniform-lie bomb model (§3.8, ADR-0004)*

- [x] **Specified** the bomb sub-model in `docs/model.md` §3.8: declaration prior
      `C(2H−1, …)/(C(H,d_b)·C(H,d_h))` (bomb eats one wire slot; `C(H−1, …)` on the
      `b=h` diagonal), the cut likelihood with the bomb as a must-not-draw card
      conditioned on "no bomb cut yet", and the `P(bad)`/`P(bomb)` readouts. §2
      rewritten to the uniform-lie bomb model; persistence pinned (P(bad) accumulates,
      P(bomb) per-round). Recorded as ADR-0004.
- [x] **Implement + validate** in `OneBadGuyOneBomb` (Axis B2): independent generative
      `(b, h)`-enumeration oracle (bomb as must-not-draw card), all three functions
      migrated to §3.8, confirmed against the oracle and **two** beats-random
      simulations (P(bad) ~0.59 and per-round P(bomb) ~0.30 vs 0.167 baseline). Derived
      from `docs/model.md`, not `General.py`.
- [ ] **Deferred (not now):** the strategic bomb-declaration model (under/over-declare)
      to A/B-test against this, and the risk-aware cut strategy.

### A5 — Cross-round combination (`CombineProbs`, ADR 0005) — *model decided: exact product*

The elementwise-product-and-renormalise rule is **exact Bayes**, not a heuristic: the
redeal makes rounds conditionally independent given the fixed roles, and the per-round
factor is a likelihood (uniform per-round role prior), so the product is the true
posterior. Informativeness is **not** weighted — it is already carried endogenously by
each round's vector shape (KL-from-uniform); an external exponent would double-count.

- [x] **Justification recorded** as [ADR 0005](docs/decisions/0005-cross-round-evidence-combination.md):
      exact product, per-round factor must stay a likelihood (not a prior-contaminated
      posterior), `P(bomb)` never combined, no informativeness weighting.
- [ ] **Robustness — ε-floor.** Mix each per-round vector with `ε · uniform` before
      multiplying so a single round's hard `0` cannot *permanently* eliminate a player
      under lie-model misspecification. Land in `General.py`.
- [ ] **Robustness — log-space accumulation.** Sum `log` per-round vectors and
      softmax-normalise to avoid underflow over many rounds / large `N` (which currently
      trips the `total == 0` → uniform branch and discards real evidence). Land in
      `General.py`; makes the ε-floor trivial to express.
- [ ] **Deferred (gated on calibration):** per-round tempering `wᵣ` to down-weight
      *distrusted* (declaration-dominated) rounds — only if a calibration test shows
      systematic overconfidence, and a single global temper is preferred first.

---

## Axis B — Variant pipeline

### B1 — `TwoBadGuysNoBomb.py` — ✅ done

The belief state is a lower-triangular matrix over *pairs* of bad guys. All findings
below were confirmed numerically against the independent split-enumeration oracle,
then fixed (TDD: red → green). The wire-split model is the §3.4.1 uniform placement
(closed-form multivariate hypergeometric); the old `C(bg_wires,k)` Binomial-½
weighting was wrong (e.g. `bg=2,H=2`: `(¼,½,¼)` vs correct `(⅙,⅔,⅙)`).

**Built the oracle**
- [x] `test_TwoBadGuysNoBomb.py` `math.comb` reference for the §3.4.1 model:
      generative declaration prior (`itertools.product`), per-pair cut posterior, and
      expected-wire marginal — each summing the split with hypergeometric weights, so
      structurally independent of the module's collapsed closed form.

**Verified & fixed against the oracle**
- [x] `ProbCut` — kept the joint-Bayes pair structure; replaced the
      `Σ_k C(bg_wires,k)·…` loop with the §3.4.1 closed form (new `L_bad_pair`
      helper). The static audit's "clean" was masked by normalisation.
- [x] `ProbDeclaration` — settled uniform-lie pair prior
      `C(2H, t_ij)/(C(H,d_i)·C(H,d_j))`; no `excess==0` special case (Axis A2);
      degeneracy falls back to uniform-over-pairs (A3).
- [x] `P_wire` bug A — good-guy branch now feasibility-gated (no negative/`>1`).
- [x] `P_wire` bug B — replaced the whole split loop with the pooled marginal
      `(bg − f_i − f_j)/(2H − rev_i − rev_j)`, dropping the spurious `+ found[i]`.

**Cleaned up & locked in**
- [x] `PlayAuto` — `revealed`/`found` now `dtype=int`.
- [x] `H_Min` — `-1` sentinel replaced with the `OneBadGuyNoBomb` `min_cutee = 0`
      pattern.
- [x] `test_TwoBadGuysNoBomb.py` (11 tests, brute-force-backed + an end-to-end
      accuracy test: ~0.94 on true bad guys vs 0.33 random baseline) + docstrings on
      every public function; `docs/roadmap.md` updated; committed.

### B2 — Later variants

- [x] **`OneBadGuyOneBomb.py`** (`B=1, M=1`) — ✅ done. All three functions migrated to
      §3.8 over the `N×N` `(bad, bomb)` config space, validated against an independent
      `(b, h)`-enumeration oracle (bomb as a must-not-draw card) and two beats-random
      simulations. `CombineProbs` accumulates only the P(bad) row marginal; the
      per-round P(bomb) column is never combined (§3.8.3). The old strategic heuristics,
      the `tabulate`/`DisplayProbs`/`CombineNonHomoProbs` dead code, and the `uf.C`
      negative-argument trap are gone.
- [ ] **`TwoBadGuysOneBomb.py`** (`B=2, M=1`) — ⏭ next; same playbook over the
      `(bad pair, bomb)` config space, combining A2 (§3.4.1 pair split) + §3.8.
- [ ] **`General.py`** — reconcile to the canonical forms; the end target. Carries
      the `P_wire` ungated-good-branch bug (and likely more — not yet trusted).

### B3 — Downstream (blocked until the backend is finalised)

- [ ] **`web/`** — re-port the math; `web/app.py` duplicates a `General.py`-style impl.
- [ ] **`AI.py`** — retrain / benchmark the REINFORCE agent against the cleaned-up
      analytic strategies (`CutMaxScore`, `CutRandom`).

---

## Done

- [x] **`OneBadGuyNoBomb.py`** — meets the full done bar (see
      [docs/roadmap.md](docs/roadmap.md#1-onebadguynobombpy--done)). `ProbDeclaration`
      now ships the uniform-lie joint-Bayes prior, validated against an independent
      generative oracle (Axis A1); degeneracy falls back to uniform (A3).
