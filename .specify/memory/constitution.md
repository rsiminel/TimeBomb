<!--
Sync Impact Report
- Version change: (template) → 1.0.0 (initial ratification)
- Scope: the web/ assistant only — backend rules live in CLAUDE.md and docs/, not here
- Modified principles: three defined from scratch (template's five-slot layout reduced)
- Added sections: Core Principles (I–III), Constraints, Governance
- Removed sections: template Section 3 placeholder (workflow covered by Constraints)
- Templates requiring updates:
  ✅ plan-template.md / spec-template.md / tasks-template.md — gates resolve against
     this file at plan time; no edits needed
- Follow-up TODOs: none
-->

# TimeBomb Web Assistant Constitution

## Core Principles

### I. The Solver Owns the Math

Every probability shown in the browser comes from `timebomb/General.py`. The web layer
contains zero probability formulas — no duplicated, re-derived, or "ported" math. If the
UI needs a quantity the solver doesn't expose, the solver grows an interface (as a
separate backend change); the web layer never grows a formula.

*Rationale*: the current `web/app.py` is a stale copy of an old solver — exactly the
failure mode that motivates this rework.

### II. Quantities, Not Commands (ADR 0006)

The site presents the solver's calibrated numbers — the four-stat `CutPanel` — and never
collapses them into a single dictated cut. The player makes the call; the site supplies
honest numbers.

### III. Web Work Stays in web/

Building the site MUST NOT modify anything under `timebomb/` or `tests/`. Missing solver
interfaces are raised as backend tasks, not patched inline.

## Constraints

- The web app's tests verify presentation against live `General.py` output (what the
  browser shows equals what the solver returns) — they do not re-test the math itself.
- Keep the stack simple and dependency-light; prefer the existing shape (small Python
  backend + static frontend) unless the spec justifies otherwise.
- Commit verified work in clean, well-scoped chunks.

## Governance

Amendments bump the version per semver (MAJOR: principle removed/redefined; MINOR:
principle added/expanded; PATCH: clarification) in a commit that states the change.
Every plan's Constitution Check gate verifies Principles I–III; a violation needs a
justification in the plan's Complexity Tracking table or the plan does not proceed.

**Version**: 1.0.0 | **Ratified**: 2026-07-03 | **Last Amended**: 2026-07-03
