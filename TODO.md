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

### A4 — Bomb model (deferred to the `*OneBomb` variants)

- [ ] Specify the bomb-holder declaration likelihood (good-with-bomb under-declares,
      bad-with-bomb over-declares), the cut likelihood, and the `P(bomb)` readout in
      `docs/model.md`. §1 promises `P(bomb)` but no section models it yet.

---

## Axis B — Variant pipeline

### B1 — `TwoBadGuysNoBomb.py` (current variant)

A static audit is done; findings below are **candidates** — confirm each
numerically against the independent brute-force oracle before fixing. The belief
state is a 2-D array over *pairs* of bad guys. The wire-split model is decided
(model.md §3.4.1: uniform placement → closed-form multivariate hypergeometric; the
code's `C(bg_wires,k)` Binomial-½ weighting is wrong — e.g. `bg=2,H=2`: code
`(¼,½,¼)` vs correct `(⅙,⅔,⅙)`).

**Build the oracle**
- [ ] `math.comb` reference for the §3.4.1 model: hypergeometric likelihood, exact
      posterior over pairs, expected-wire. Ground truth for everything below. **Do
      not** derive it from `General.py`.

**Verify & fix against the oracle**
- [ ] `ProbCut` — joint-Bayes over the pair space is structurally right, but replace
      the `Σ_k C(bg_wires,k)·…` loop with the §3.4.1 closed form. (The static audit
      called this "clean" — it was not.)
- [ ] `ProbDeclaration` — apply the settled prior (Axis A1/A2); verify per-pair
      weighting against the oracle.
- [ ] `P_wire` bug A — good-guy branch (line ~210) is ungated → emits a **negative**
      (or `>1`) probability when `found[i] != decls[i]`. Same fix as
      `OneBadGuyNoBomb`. *(Confirmed by inspection.)*
- [ ] `P_wire` bug B — `j_wires = bg_wires - i_wires - found[j]` carries a spurious
      `+ found[i]`; should be `bg_wires - k - found[j]`. *(Confirmed; only bites when
      `found[i] != 0`.)*

**Cleanup & lock in**
- [ ] `PlayAuto` — `revealed`/`found` are float arrays; use `dtype=int` (convention).
- [ ] `H_Min` — `min_cutee = -1` sentinel can leak an out-of-range index into the
      returned path; align with the `OneBadGuyNoBomb` version.
- [ ] `test_TwoBadGuysNoBomb.py` (brute-force-backed) + docstrings on every public
      function, update `docs/roadmap.md`, then commit.

### B2 — Later variants

- [ ] **`OneBadGuyOneBomb.py`** (`B=1, M=1`) — same playbook; needs Axis A4.
- [ ] **`TwoBadGuysOneBomb.py`** (`B=2, M=1`) — same playbook; needs A2 + A4.
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
