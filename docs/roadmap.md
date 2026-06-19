# Time Bomb Assistant — Roadmap & Status

The plan and current status. For the math see [model.md](model.md); for the
live task list see [../TODO.md](../TODO.md).

## Roadmap

The backend probability code was cleaned up **one variant at a time**, simplest
first, each hardcoded variant brought to the "definition of done" bar below. That work
is complete: **`General.py` is the backend solver** and subsumes the four variants (each
is its `(num_bad, num_bom)` projection).
The table below is the historical record of that journey.

| # | Module                  | Config       | Status   |
| - | ----------------------- | ------------ | -------- |
| 1 | `OneBadGuyNoBomb.py`    | `B=1, M=0`   | ✅ done   |
| 2 | `TwoBadGuysNoBomb.py`   | `B=2, M=0`   | ✅ done   |
| 3 | `OneBadGuyOneBomb.py`   | `B=1, M=1`   | ✅ done   |
| 4 | `TwoBadGuysOneBomb.py`  | `B=2, M=1`   | ✅ done   |
| 5 | `General.py`            | arbitrary    | ✅ done   |

The backend is now complete: all five modules implement the same validated uniform-lie
model, and `General.py` subsumes the four variants (each is its `(num_bad, num_bom)`
projection) while adding **joint inference over the bad count** for the player counts where
it is uncertain (N=4, N=7; [ADR 0008](decisions/0008-joint-num-bad-inference.md)).

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
gaps themselves are catalogued in [model.md §3.6](model.md#36-open-modelling-gaps);
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
4. **Bomb sub-model (§3.2–§3.4): uniform-lie bomb model.** A player declares truthfully iff
   good *and* bomb-free; everyone else lies uniformly over `{0..H}`. The config is the
   pair `(bad, bomb)`; the prior is the closed form
   `C(2H−1, …)/(C(H,d_b)·C(H,d_h))` (the bomb eats one wire slot), the cut likelihood
   treats the bomb as a must-not-draw card conditioned on "no bomb cut yet", and the
   bomb is **per-round**: `P(bad)` accumulates across rounds, `P(bomb)` does not. See
   [decisions/0004](decisions/0004-uniform-lie-bomb-model.md).
5. **Cross-round combination (`CombineProbs`): exact elementwise product.** The redeal
   makes rounds conditionally independent given the fixed roles, and each per-round
   factor is a likelihood (uniform per-round role prior), so multiply-and-renormalise is
   the *exact* posterior — not a heuristic. The per-round factor must stay a likelihood
   (never feed the accumulated belief back as the round's prior); `P(bomb)` is never
   combined; round informativeness is **not** weighted (it is already carried by each
   round's vector shape — an external coefficient would double-count). Two robustness
   hardenings (ε-floor against permanent `0`-pinning under misspecification; log-space
   accumulation against underflow) are warranted and land in `General.py` — now
   prototyped and tested in `TwoBadGuysOneBomb` (B4) to de-risk that work. See
   [decisions/0005](decisions/0005-cross-round-evidence-combination.md).

**Deferred refinements (not scheduled).**

- **Strategic / parametric lie models.** Replace the uniform lie — for the no-bomb bad
  guy *and* for bomb-holders (§3.6: good-with-bomb under-declares, bad-with-bomb
  over-declares) — with a strategic or tunable-bias model. Each adds a free parameter
  to fit and validate; weigh only well after the pipeline is correct and trusted. Once
  the uniform-lie bomb variant is validated, A/B-test the strategic bomb model against
  it on bad-guy and bomb identification accuracy.
- **Cut recommendation — quantities-only four-stat panel (§3.5, ADR 0006).** Present
  per player: P(safe wire), P(bomb), 1-ply ΔH(bad), and round-horizon H(bad) — exploit,
  risk, immediate-info, strategic-info — leaving the explore/exploit/risk integration to
  the human. Risk-awareness is the raw `P(bomb)` readout (no ad-hoc penalty: under the
  P(win) objective the bomb is a terminal-0 state). Calibration of `P(bad)`/`P(bomb)` is
  a prerequisite for trusting the panel.
- **Horizon-weighted VOI lookahead (open, ADR 0006).** Upgrade stat 4's objective from
  end-of-round entropy to a λ-free win-prob gain weighted by *cuts remaining* (role info
  is durable, so early-round exploration compounds — what a 1-ply VOI cannot see). Needs
  a simulation-estimated sensitivity coefficient; objective choice still to debate.
- **Per-round tempering in `CombineProbs`.** A weight `wᵣ` down-weighting *distrusted*
  (declaration-dominated) rounds — only if a calibration test reveals systematic
  overconfidence, and a single global temper is preferred before a per-round one. This
  encodes model distrust, not informativeness (which is already handled); see
  [decisions/0005](decisions/0005-cross-round-evidence-combination.md).

## Engineering & infrastructure (deferred)

Not modelling decisions — project plumbing and robustness, scheduled after the backend
is correct and trusted.

- **Resilience to model-breaking play (§3.6).** Real tables violate the uniform-lie
  model: miscounts, arithmetically impossible declarations, house-rule deals, strategic
  liars producing ~0-probability observations. Three hardenings; two are done.
  - **ε-floor + log-space `CombineProbs`** (ADR 0005) — done in `General` and
    `TwoBadGuysOneBomb`. The three simpler variants keep the bare product, and that is now
    **intentional**: the variants are frozen oracles (their `Play` loops are vestigial), so
    porting it is *retired*, not pending — the real-table solver is `General.py`.
  - **Graceful response to inconsistent input** — done (`timebomb/Consistency.py`,
    `tests/test_consistency.py`): the interactive `Play` re-prompts clearly invalid numeric
    entry and warns (once per round, continuing with the safe fallback) on jointly impossible
    declarations or an impossible cut result, so a miscount no longer silently discards a
    round's evidence. Hybrid policy, reusable by the future web port.
- **Broader test coverage.** The suites pair an independent `math.comb` brute force
  (correctness) with an end-to-end beats-random simulation (predictive usefulness).
  - **Calibration — done** (`tests/calibration.py` harness + `tests/test_calibration.py`):
    `P(bad)`, declaration-time `P(bomb)`, and `P(num_bad)` (N=4/N=7, the cross-`B`
    absolute-weight guard of ADR 0007/0008) all match empirical frequencies (ECE ≈ 0.01),
    so the cut panel's risk numbers are trustworthy and A5 tempering stays unwarranted.
    Building it caught two simulator/model mismatches in `General.PlayAuto` — a
    player-uniform (not slot-uniform) wire deal, and folding the bomb-detonating cut as a
    "no-bomb" observation — both fixed.
  - **Cross-variant consistency — done** for the panel/marginals (`General` at `(2,1)` vs
    `TwoBadGuysOneBomb`); worth extending to every projection.
  - **Property-based / fuzz tests + regression fixtures — done** (`tests/test_properties.py`):
    structural properties that hold for any correct implementation regardless of the numbers —
    player-relabeling **equivariance**, `CombineProbs` round **permutation invariance**, the
    no-observation `ProbCut` **identity** — plus distribution-invariant sweeps at `N`/`hand_size`
    past the brute force's reach, and pinned golden vectors generated from the *independent*
    oracle. These catch index/axis/ordering/scale bugs a value-by-value oracle cannot.
- **Principled `beats_random` thresholds — done.** The end-to-end tests no longer assert
  hand-tuned cutoffs; `tests/baseline_stats.py` derives a self-calibrating bar from the
  run's own sample (a one-sided 5σ z-test of a per-game statistic against the
  no-information baseline). The robust cross-variant claim is the *paired* gap
  `P(bad|bad) − P(bad|good) > 0`, which holds even where a bomb-holding good guy lifts
  `P(bad|good)` to ≈baseline.
- **Test-suite speed — done.** The simulation-heavy suite runs across all cores by default
  (`pytest-xdist`, `-n auto` in `pytest.ini`); the calibration sweeps are parametrized and
  the joint test split per `N` so xdist schedules them as independent units; and one panel
  test's uncapped `O((2N)^stop)` lookahead (157s, range-checks only) is depth-capped.
  Full suite ≈ 70s, down from ≈ 4.5 min serial.
- **Packaging.** Turn the repo into an installable package (`pyproject.toml`, a
  `timebomb` distribution, console entry points for `Play`/`PlayAuto`) so it no longer
  relies on `PYTHONPATH=timebomb` and a hand-rolled `.venv`. Pins the
  numpy/scipy/pytest-xdist dependencies and makes the test/CI setup reproducible.

## Status detail

### Variants 1–4 — ✅ done, now FROZEN

All four hardcoded variants reached the definition-of-done bar:
`ProbDeclaration`/`ProbCut`/`P_wire` on the validated uniform-lie model, each checked
against an *independent* `math.comb` oracle and a beats-random simulation, docstrings
throughout, dead code removed. They are projections of `General.py`; the per-variant
derivations and cleanups live in the ADRs and git history. All backend work happens in `General.py`.

| Variant | Config | Belief state | Independent test oracle |
| --- | --- | --- | --- |
| 1 `OneBadGuyNoBomb` | `B=1, M=0` | length-`N` vector | `itertools.product` generative prior |
| 2 `TwoBadGuysNoBomb` | `B=2, M=0` | lower-triangular pair matrix | split-enumeration `math.comb` |
| 3 `OneBadGuyOneBomb` | `B=1, M=1` | `N×N` `(bad, bomb)` matrix | `(b,h)`-enumeration, bomb must-not-draw |
| 4 `TwoBadGuysOneBomb` | `B=2, M=1` | `N×N×N` `(pair, bomb)` tensor | `(b1,b2,h)`-enumeration, explicit split |

Variant 4 also pre-implemented the four-stat cut panel and the robust ε-floor/log-space
`CombineProbs` as the verified references `General.py` was then built from.

### 5. `General.py` — ✅ done

Rewritten from the bug-ridden strategic-lie implementation (which also could not import —
`sympy`/`tabulate`) to the validated uniform-lie model over the general `(bad set S, bomb
holder h)` configuration space, for arbitrary `num_bad` and `num_bom ∈ {0,1}`. Highlights:

- **Reconciliation that surfaced a real bug.** Deriving the absolute likelihood for joint
  `num_bad` inference exposed the dropped `(H+1)^{−|F|}` lie factor in the declaration
  prior — fixed at the root across model.md §3.3, the `*OneBomb` variants, and their oracles
  ([ADR 0007](decisions/0007-declaration-lie-count-factor.md)); verified by a generative
  Monte Carlo. B1/B2 were unaffected.
- **Core math** (`ProbDeclaration`/`ProbCut`/`P_wire`) is the corrected uniform-lie closed
  form, vectorised within the `itertools.combinations` config loop; the §3.4.1 `L_bad`
  collapse and the must-not-draw bomb term generalise the variant helpers. Validated against
  an independent fully-generative `math.comb` oracle (incl. the lie factor) across
  `(num_bad,num_bom) ∈ {(1,0),(2,0),(1,1),(2,1)}`, and shown identical to the corrected
  `TwoBadGuysOneBomb` on its projection.
- **Robust `CombineProbs`** (ε-floor + log-space, A5/ADR 0005) and the **four-stat cut
  panel** (`CutPanel`/`NextHBad`/`RoundHorizonH`/`H_Min`/`EntropyBad`, A6/§3.5) ported and
  generalised from the B4 reference.
- **Joint `num_bad` inference** ([ADR 0008](decisions/0008-joint-num-bad-inference.md), §3.5.1):
  for N=4/N=7 the bad count updates from evidence — `P(B|D) ∝ P(B)·(1/C(N,B))·Σ_S Π_r u_r(S;B)`
  in log-space — instead of the old fixed-weight `pos_bad` mixture.
- `test_General.py`: the generative oracle sweeps, panel invariants + a
  cross-variant panel check against B4, `CombineProbs` robustness, the joint posterior vs an
  independent re-derivation, within-`B` agreement with `CombineProbs`, and beats-random at
  N=4/N=7. Open issue carried forward: the round-horizon lookahead is `O((2N)^stop)` — a
  beam/analytic approximation is the follow-up (depth-capped for display today).

### 6. `web/` — fix up the website.

Re-port the cleaned `General.py` math behind the
Flask API (`web/app.py` currently duplicates an old, bug-ridden `General.py`-style
implementation) and update the browser assistant to present the quantities-only
four-stat panel (§3.5) instead of a single dictated cut. The UI is the natural home
for a real-table assistant: enter declarations and cut results, read the belief.

### 7. `AI.py` — create the AI.
Train and benchmark the REINFORCE cut agent against
the cleaned-up analytic strategies (`CutMaxScore`, `CutRandom`, the info-greedy
lookahead). Goal: learn a cut policy that beats the hand-written heuristics, and use
it as an empirical yardstick for the horizon-weighted VOI question (§3.5/§3.6).
