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
| 2 | `TwoBadGuysNoBomb.py`   | `B=2, M=0`   | ⏭ next   |
| 3 | `OneBadGuyOneBomb.py`   | `B=1, M=1`   | pending  |
| 4 | `TwoBadGuysOneBomb.py`  | `B=2, M=1`   | pending  |
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
3. **Documented** — every public function has a docstring stating its inputs,
   outputs, and the model assumption it encodes; the module header explains the
   variant.
4. **Reconciled** — known discrepancies resolved: the canonical joint-Bayes
   `ProbCut` (§3.4), the correct `P_wire` denominator and gating (§3.5), and no
   dead/duplicate code.

## Status detail

### 1. `OneBadGuyNoBomb.py` — ✅ done

Meets all four criteria. Highlights:

- `ProbCut` collapsed from three implementations to the single canonical
  joint-Bayes form; the legacy per-player and "mathematically justified" variants
  removed.
- `P_wire` denominator corrected to remaining face-down cards, and both branches
  feasibility-gated (fixing a negative-probability bug found during testing).
- `PlayAuto` integer-array crash, `ProbSus` `NameError`, and `Play` input
  validation all fixed.
- Docstrings on every public function; `test_OneBadGuyNoBomb.py` (16 tests) checks
  the math against independent `math.comb` brute-force references.

### 2. `TwoBadGuysNoBomb.py` — ⏭ next

`B=2`: the belief state becomes a 2-D array over *pairs* of bad guys. Apply the
same playbook — audit `ProbDeclaration` / `ProbCut` / `P_wire` against the model,
collapse to the canonical forms, add a brute-force-backed test suite and
docstrings. See [../TODO.md](../TODO.md).
