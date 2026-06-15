# TODO

Open work items. Stable model docs live in [docs/model.md](docs/model.md); plan
and status in [docs/roadmap.md](docs/roadmap.md).

## Backend (current track — one variant at a time)

- [ ] **`TwoBadGuysNoBomb.py`** — bring to the done bar: audit `ProbDeclaration` /
      `ProbCut` / `P_wire` against the model, collapse to the canonical forms, add a
      brute-force-backed test suite and docstrings.
- [ ] **`OneBadGuyOneBomb.py`** — same playbook (`B=1, M=1`).
- [ ] **`TwoBadGuysOneBomb.py`** — same playbook (`B=2, M=1`).
- [ ] **`General.py`** — reconcile to the canonical forms; serves as the end target.

## Known bugs to carry forward

- [ ] **`General.py` `P_wire`** — the good-guy branch
      `(1 − P(bad))·(decls − found)` is **not** feasibility-gated, so it can emit a
      negative probability when `found > decls` (the same bug fixed in
      `OneBadGuyNoBomb`). Check the `OneBomb` variants' `P_wire` for the same issue.

## Downstream (blocked until the backend is finalised)

- [ ] **`web/`** — re-port the math after the backend is finalised; the Flask app
      (`web/app.py`) currently duplicates a `General.py`-style implementation.
- [ ] **`AI.py`** — retrain / benchmark the REINFORCE agent against the cleaned-up
      analytic strategies (`CutMaxScore`, `CutRandom`).

## Done

- [x] **`OneBadGuyNoBomb.py`** — meets the done bar (see
      [docs/roadmap.md](docs/roadmap.md#1-onebadguynobombpy--done)).
