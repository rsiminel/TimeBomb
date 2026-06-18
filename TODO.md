# TODO

A worklist of **open** items. Completed work is pruned aggressively — git history, the
ADRs in [docs/decisions/](docs/decisions/), and the status section of
[docs/roadmap.md](docs/roadmap.md) preserve the what and why. Stable model docs live in
[docs/model.md](docs/model.md).

Axes: **A** foundations (cross-cutting model) · **B** variant pipeline · **C** downstream
(web + AI) · **D** engineering & infrastructure.

## Testing ground truth

The only valid correctness oracle is an **independent `math.comb` brute force derived from
`docs/model.md`** (as in `test_OneBadGuyNoBomb.py`) — never test a module against another
*implementation*. The four hardcoded variants are **frozen** independent oracles that
cross-check the backend solver `General.py`; their entire worth
is that they share no code with it, so **they must not be modified**. Each module's suite
pairs the brute force with an end-to-end *beats-random* simulation (self-calibrating
thresholds in `tests/baseline_stats.py`) and the cross-module calibration checks in
`tests/test_calibration.py`.

## Axis A — Foundations ✅ done

The model is settled, implemented, and brute-force validated across all variants: the
uniform-lie declaration prior (incl. the ADR-0007 `(H+1)^{−|F|}` lie factor), the §3.4.1
multivariate-hypergeometric wire split, the bomb sub-model, the degeneracy fallback, the
exact-product `CombineProbs` (ε-floor + log-space), the four-stat cut panel, and joint
`num_bad` inference. See [model.md §3](docs/model.md) and ADRs 0001–0008.

## Axis B — Variant pipeline ✅ done (variants now frozen)

Variants 1–4 and `General.py` are all at the definition-of-done bar (see
[roadmap.md status](docs/roadmap.md#status-detail)). `General.py` subsumes the four; each
hardcoded variant is its `(num_bad, num_bom)` projection.

## Axis C — Downstream (blocked on the now-finished backend)

### C1 — `web/` — fix up the website
- [ ] Re-port the cleaned `General.py` math (`web/app.py` duplicates an old
      `General.py`-style impl) and present the quantities-only four-stat panel (§3.5, the
      `CutPanel`) in the browser assistant rather than a single dictated cut.

### C2 — `AI.py` — create the AI
- [ ] Retrain / benchmark the REINFORCE cut agent against the cleaned analytic strategies
      (`CutRandom`, the info-greedy lookahead); use it as the empirical yardstick for the
      horizon-weighted VOI question (model.md §3.6, ADR 0006). Note: `AI.py`'s own simulator
      still has the capped-lie deal bug (`min(hand_size, active_wires)`) — fix when it comes
      off hold.

## Axis D — Engineering & infrastructure

### D3 — Resilience to model-breaking play
- [x] **Graceful response to inconsistent input** (`timebomb/Consistency.py`,
      `tests/test_consistency.py`): the interactive `Play` loops now re-prompt clearly invalid
      numeric entry (declaration outside `[0, H]`, bad cut-result code, cutting an empty hand)
      and warn — once per round, continuing with the safe fallback — on declarations that are
      jointly impossible under the rules or a cut result impossible given the declarations.
      Feasibility = `General._decl_weights(...).sum() > 0` (convention-independent); the
      impossible-cut signal is `ProbCut` returning the same prior object. See model.md §3.6.
- [ ] Still open: port the ε-floor + log-space `CombineProbs` (ADR 0005, in `General` and
      `TwoBadGuysOneBomb`) to the simpler variants `OneBadGuyNoBomb`/`TwoBadGuysNoBomb`/
      `OneBadGuyOneBomb`, which still use the bare product (one impossible round can permanently
      hard-zero a hypothesis).

### D4 — Broader test coverage
- [x] **Calibration** — `tests/calibration.py` (harness) + `tests/test_calibration.py`:
      `P(bad)`, declaration-time `P(bomb)`, and `P(num_bad)` (N=4/N=7, the cross-`B`
      absolute-weight guard) all match empirical frequencies. Cross-variant panel/marginal
      consistency (`General` vs the variants) is covered too.
- [ ] Still open: property-based / fuzz tests (e.g. Hypothesis) over the distribution
      invariants, and regression fixtures pinning known belief vectors.

### D5 — Packaging
- [ ] Make the repo an installable package (`pyproject.toml`, `timebomb` distribution,
      `Play`/`PlayAuto` console entry points) so it drops the `PYTHONPATH=timebomb` +
      hand-rolled `.venv` setup and pins numpy/scipy/pytest-xdist for reproducible CI.

## Deferred modelling refinements (not scheduled)

- **Strategic / parametric lie model** (§3.6): good-with-bomb under-declares, bad-with-bomb
  over-declares. Adds a free parameter; A/B-test against the uniform-lie model once trusted.
- **Risk-aware cut strategy** (§3.6): fold `P(bomb)` into a single recommendation. Held by
  the quantities-only philosophy (ADR 0006) — the panel stops at the raw `P(bomb)`.
- **Horizon-weighted VOI lookahead** (§3.6, ADR 0006 "Open"): upgrade panel stat 4 from
  end-of-round entropy to a λ-free win-prob gain weighted by cuts-remaining; needs a
  simulation-estimated sensitivity coefficient.
- **Round-horizon cut lookahead cost**: stat 4 is `O((2N)^stop)`, depth-capped for display;
  a beam/analytic approximation is the follow-up.
- **Per-round tempering in `CombineProbs`** (ADR 0005): only if a calibration test reveals
  systematic overconfidence — it does not, so this stays unwarranted.
