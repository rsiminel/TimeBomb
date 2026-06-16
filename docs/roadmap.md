# Time Bomb Assistant — Roadmap & Status

The plan and current status. For the math see [model.md](model.md); for the
live task list see [../TODO.md](../TODO.md).

## Roadmap

The backend probability code is cleaned up **one variant at a time**, simplest
first. Each variant is brought to the "definition of done" bar below before the
next begins. The four hardcoded variants are stepping stones; `General.py` is the
canonical implementation they converge toward.

| # | Module                  | Config       | Status   |
| - | ----------------------- | ------------ | -------- |
| 1 | `OneBadGuyNoBomb.py`    | `B=1, M=0`   | ✅ done   |
| 2 | `TwoBadGuysNoBomb.py`   | `B=2, M=0`   | ✅ done   |
| 3 | `OneBadGuyOneBomb.py`   | `B=1, M=1`   | ✅ done   |
| 4 | `TwoBadGuysOneBomb.py`  | `B=2, M=1`   | ⏭ next   |
| 5 | `General.py`            | arbitrary    | pending  |

Downstream, **unblocked only after the backend is done**:

| # | Module    | Status                                                      |
| - | --------- | ---------------------------------------------------------- |
| 6 | `web/`    | on hold — Flask API + browser assistant                    |
| 7 | `AI.py`   | on hold — REINFORCE cut agent vs. the analytic strategies  |

## Definition of done (per variant)

1. **Works** — runs without errors; `ProbDeclaration` and `ProbCut` always return
   a valid probability distribution (entries in `[0,1]`, summing to 1).
2. **Tested** — an automated test suite asserts the distribution invariants across
   a parameter sweep (entries in `[0,1]`, no `NaN`, sums to 1), plus targeted
   known-value cases that pin the model, plus the edge cases that break a naive
   `ProbCut` (impossible observations). Math is checked against an *independent*
   brute-force reference so a shared bug cannot hide.
3. **Predictively useful** — an end-to-end test runs full simulated games and
   confirms the final belief identifies the true bad guy(s) **far above the random
   baseline** (each player's `P(bad)` marginal averages `B/N` under no information).
   Oracle-correctness (criterion 2) proves the math is implemented right; this proves
   the *model itself* is informative — a distinct guarantee the oracle cannot give.
4. **Documented** — every public function has a docstring stating its inputs,
   outputs, and the model assumption it encodes; the module header explains the
   variant.
5. **Reconciled** — known discrepancies resolved: the canonical joint-Bayes
   `ProbCut` (§3.4), the correct `P_wire` denominator and gating (§3.5), and no
   dead/duplicate code.

## Cross-cutting foundations

Some model work is **not owned by a single variant** — it touches the shared
likelihoods and so cuts across the whole pipeline. This track runs alongside the
per-variant work and, where noted, can re-open a variant already marked done. The
gaps themselves are catalogued in [model.md §3.7](model.md#37-open-modelling-gaps);
the task breakdown is [TODO.md Axis A](../TODO.md#axis-a--foundations-cross-cutting);
the rationale behind each settled choice is recorded as an ADR in
[decisions/](decisions/).

**Modelling decisions — settled (implementation + brute-force validation remain):**

1. **Declaration prior (§3.3): uniform-lie joint-Bayes.** The bad guy declares
   uniformly at random; the prior is the multivariate-hypergeometric likelihood of
   the deal each configuration forces — `P(bad=i) ∝ C(H, t_i)/C(H, decls[i])`, with a
   closed-form `B > 1` generalisation. Replaces the old card-count heuristic.
   (Implemented and validated against a generative oracle in `OneBadGuyNoBomb`; the
   remaining variants still ship the heuristic until cleaned.)
2. **`B > 1` prior and `P_wire` marginal** use the §3.4.1 uniform-placement
   (multivariate-hypergeometric) model, so prior and cut update share one
   wire-placement law. No `excess = 0` special case for `B > 1`.
3. **Degeneracy:** on a zero marginal, fall back to the prior/uniform — never an
   unnormalisable all-zeros vector.
4. **Bomb sub-model (§3.8): uniform-lie bomb model.** A player declares truthfully iff
   good *and* bomb-free; everyone else lies uniformly over `{0..H}`. The config is the
   pair `(bad, bomb)`; the prior is the closed form
   `C(2H−1, …)/(C(H,d_b)·C(H,d_h))` (the bomb eats one wire slot), the cut likelihood
   treats the bomb as a must-not-draw card conditioned on "no bomb cut yet", and the
   bomb is **per-round**: `P(bad)` accumulates across rounds, `P(bomb)` does not. See
   [decisions/0004](decisions/0004-uniform-lie-bomb-model.md).

**Deferred refinements (not scheduled).**

- **Strategic / parametric lie models.** Replace the uniform lie — for the no-bomb bad
  guy *and* for bomb-holders (§3.8: good-with-bomb under-declares, bad-with-bomb
  over-declares) — with a strategic or tunable-bias model. Each adds a free parameter
  to fit and validate; weigh only well after the pipeline is correct and trusted. Once
  the uniform-lie bomb variant is validated, A/B-test the strategic bomb model against
  it on bad-guy and bomb identification accuracy.
- **Risk-aware cut strategy.** Fold `P(bomb)` into the cut recommendation (expected
  wire progress vs. bomb risk, §3.6). The bomb sub-model itself stops at the `P(bomb)`
  readout.

## Status detail

### 1. `OneBadGuyNoBomb.py` — ✅ done

Meets all four criteria. Highlights:

- `ProbCut` collapsed from three implementations to the single canonical
  joint-Bayes form; the legacy per-player and "mathematically justified" variants
  removed.
- `P_wire` denominator corrected to remaining face-down cards, and both branches
  feasibility-gated (fixing a negative-probability bug found during testing).
- `ProbDeclaration` migrated from the old card-count heuristic to the uniform-lie
  joint-Bayes prior (§3.3), validated against an independent `itertools.product`
  generative oracle; degeneracy falls back to uniform (Cross-cutting foundations
  items 1 and 3).
- `PlayAuto` integer-array crash, `ProbSus` `NameError`, and `Play` input
  validation all fixed.
- Docstrings on every public function; `test_OneBadGuyNoBomb.py` (21 tests) checks
  the math against independent `math.comb` brute-force references, plus an end-to-end
  accuracy test: over 400 games the belief puts ~0.95 on the true bad guy (~0.96 top-1
  accuracy) vs the 0.20 random baseline.

### 2. `TwoBadGuysNoBomb.py` — ✅ done

`B=2`: the belief state is a lower-triangular matrix over *pairs* of bad guys.
Meets all four criteria. Highlights:

- All three model functions migrated from the old Binomial-½ wire split to the
  §3.4.1 uniform-placement (multivariate-hypergeometric) closed forms: the
  `ProbDeclaration` pair prior, a new `L_bad_pair` helper collapsing the cut
  likelihood, and the pooled-marginal `P_wire`.
- `P_wire` good-guy branch feasibility-gated (the ungated negative-probability bug)
  and the spurious `+ found[i]` split index removed; `ProbDeclaration` degeneracy
  now falls back to uniform-over-pairs.
- `PlayAuto` integer arrays and the `H_Min` `-1` sentinel fixed; docstrings on every
  public function.
- `test_TwoBadGuysNoBomb.py` (11 tests) checks the math against an independent
  split-enumeration `math.comb` oracle (the generative declaration prior, the
  per-pair cut posterior, and the expected-wire marginal), plus an end-to-end
  accuracy test: over 400 simulated games the belief puts ~0.94 on the true bad
  guys vs ~0.03 on the good ones (random baseline 0.33).

### 3. `OneBadGuyOneBomb.py` — ✅ done

`B=1, M=1`: introduces the Bomb. The belief state is the full `N×N` matrix
`probs[b][h]` = P(player `b` bad, player `h` holds the bomb), diagonal allowed. Meets
all five criteria. Highlights:

- All three model functions migrated to the §3.8 uniform-lie bomb model: the
  `ProbDeclaration` closed form `C(2H−1, …)/(C(H,d_b)·C(H,d_h))` (with `C(H−1, …)` on
  the `b=h` diagonal), a new `L_config`/`L_bomb_hand` pair giving the cut likelihood
  with the bomb as a must-not-draw card conditioned on "no bomb yet", and the
  bomb-aware `P_wire` (split-posterior expected wires, denominator still counts the
  bomb card). The old strategic heuristics (good-bomb under-declares, bad-bomb
  over-declares) and the `uf.C` negative-argument trap are gone.
- `PlayAuto` now deals the bomb first then wires among the remaining slots and
  generates declarations under the uniform-lie model; `DisplayProbs`/`tabulate` and the
  dead `CombineNonHomoProbs` dropped; integer arrays throughout; `H_Min` `−1` sentinel
  fixed. `CombineProbs` accumulates only the P(bad) row marginal — the per-round
  P(bomb) column is never combined (§3.8.3).
- `test_OneBadGuyOneBomb.py` (10 tests) checks the math against an independent
  `(b, h)`-enumeration `math.comb` oracle (split-summed declaration prior, cut
  likelihood, and `P_wire` marginal), plus **two** end-to-end accuracy tests: over 400
  games the combined belief puts ~0.59 on the true bad guy (top-1 ~0.69) vs the 0.167
  baseline — weaker than the no-bomb variants because a good guy forced to lie by the
  bomb looks bad — and the per-round P(bomb) column puts ~0.30 on the true holder
  (top-1 ~0.42) vs the same baseline.

### 4. `TwoBadGuysOneBomb.py` — ⏭ next

`B=2, M=1`: combines the `B>1` pair structure (§3.4.1) with the bomb sub-model (§3.8).
Same playbook over the `(bad pair, bomb)` config space.
