<!--
Sync Impact Report
- Version change: 1.0.0 → 2.0.0 (MAJOR: Principles II and III redefined)
- Scope change: "the web/ assistant" → the whole web app (v1 assistant at /assistant
  + natively hosted game at /play); backend rules still live in CLAUDE.md and docs/
- Modified principles:
  I. The Solver Owns the Math → I. The Solver Owns the Math, the Engine Owns the Rules
     (extended: rule adjudication joins probability math on the never-in-web list)
  II. Quantities, Not Commands → rescoped to the assistant surface; AI opponents are
     players, not the panel, and their policies decide declarations, claims, and cuts
  III. Web Work Stays in web/ → shared rules-engine carve-out added (extraction from
     sim/engine.py); timebomb/ and tests/ remain untouchable
- Added sections: none (Constraints extended with hidden-information delivery rule)
- Removed sections: none
- Templates requiring updates:
  ✅ plan-template.md / spec-template.md / tasks-template.md — gates resolve against
     this file at plan time; no edits needed
- Follow-up TODOs: none
-->

# TimeBomb Web App Constitution

## Core Principles

### I. The Solver Owns the Math, the Engine Owns the Rules

Every probability shown in the browser comes from `timebomb/General.py`. The web layer
contains zero probability formulas — no duplicated, re-derived, or "ported" math. If the
UI needs a quantity the solver doesn't expose, the solver grows an interface (as a
separate backend change); the web layer never grows a formula.

Likewise, game rules — dealing, cut resolution, win judgment, and the construction of
hidden-information per-player views — live in the shared rules-engine package (promoted
from `sim/engine.py`), never in the web layer. The web layer presents state and forwards
player intents; it MUST NOT adjudicate them.

*Rationale*: v1's predecessor was a stale in-page copy of an old solver — the failure
mode this principle exists to prevent. Hosting the game adds a second temptation
(re-implementing rules in the UI); the same rule covers both.

### II. Quantities, Not Commands — on the Assistant Surface (ADR 0006)

The stats panel presents the solver's calibrated numbers — the four-stat `CutPanel` —
and never collapses them into a single dictated cut. The human player makes the call;
the panel supplies honest numbers.

This principle governs the assistant surface only. AI opponents are *players*, not the
panel: their agent policies MUST decide their own declarations, claims, and cuts, and
doing so is not a violation. No AI decision path may leak back into the panel shown to
humans (e.g. surfacing a bot's chosen target as a recommendation).

*Rationale*: the assistant informs; players — human or AI — play.

### III. Web Work Stays in web/, Plus the Shared-Engine Carve-Out

Building the site MUST NOT modify anything under `timebomb/` or `tests/` (the backend
solver and its suite). Missing solver interfaces are raised as backend tasks, not
patched inline.

Carve-out: web feature work MAY create and modify the shared rules-engine package
extracted from `sim/engine.py`, and MAY adapt `sim/` to import from it — provided
`sim/`'s own tests keep passing after every such change. The engine package is shared
infrastructure, not web code: it MUST stay importable and testable without the web app.

*Rationale*: the referee already exists and is arena-tested; duplicating it inside
`web/` would create two rule books. The carve-out lets web work promote it without
opening the door to solver edits.

## Constraints

- The web app's tests verify presentation against live `General.py` output (what the
  browser shows equals what the solver returns) — they do not re-test the math itself.
- Hidden information is enforced server-side: a browser is only ever sent the view of
  the player (or shared hotseat surface) it is acting for. Secrets MUST NOT be shipped
  to the client and hidden by the UI.
- Keep the stack simple and dependency-light; prefer the existing shape (small Python
  backend + static frontend, no build step) unless the spec justifies otherwise.
- Commit verified work in clean, well-scoped chunks.

## Governance

Amendments bump the version per semver (MAJOR: principle removed/redefined; MINOR:
principle added/expanded; PATCH: clarification) in a commit that states the change.
Every plan's Constitution Check gate verifies Principles I–III; a violation needs a
justification in the plan's Complexity Tracking table or the plan does not proceed.

**Version**: 2.0.0 | **Ratified**: 2026-07-03 | **Last Amended**: 2026-07-03
