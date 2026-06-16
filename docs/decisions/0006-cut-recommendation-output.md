# ADR 0006: Cut-recommendation output — the four-stat panel

- **Status:** Accepted (panel); the horizon-weighted VOI upgrade path is **Open**
- **Date:** 2026-06-16
- **Scope:** the cut recommendation (`General.CutMaxScore`, `H`/`NextH`/`H_Min`);
  model.md §3.6; roadmap.md deferred refinements; TODO.md
- **Builds on** [0004](0004-uniform-lie-bomb-model.md) and
  [0005](0005-cross-round-evidence-combination.md) (the redeal → durable roles,
  ephemeral bomb).

## Context

Once the belief is computed, the assistant must help the player choose whose hand to
cut. This was framed as an explore (reduce uncertainty about who is bad) vs. exploit
(cut to make progress / avoid the bomb) problem, with risk-aware cutting a third
concern. The question: what is the mathematically justified way to combine these, and
what should the assistant actually output?

## Options considered

**Objective framing:**
- **A — Three objectives traded off** (score, entropy, risk) via a scalarization
  `score + λ·info − C·risk`. Rejected: the terms are in incommensurable units (expected
  wires, bits, probability), so every weight is arbitrary; bandit (UCB/Thompson) regret
  guarantees do not transfer to a finite-horizon hidden-role team game.
- **B — One objective: P(win), a POMDP** (chosen framing). Information has no intrinsic
  value; it is worth exactly its effect on future win probability, valued automatically
  by the Bellman equation. Explore/exploit is an artifact of approximating an
  intractable POMDP. Bomb risk dissolves too: cutting the bomb is a terminal state of
  value 0, so risk-aversion is built in, parameter-free — no ad-hoc penalty needed.

**Who integrates explore/exploit/risk:**
- **Quantities only (chosen).** Present calibrated, individually-justified inputs; the
  human owns the policy (they hold table-talk, tells, meta-strategy, and their own risk
  appetite — the utility function we decline to impose). This also makes the explore
  metric *tractable*: with no need to combine units, expected information gain in bits
  (computed in one step from the belief + cut likelihood) suffices — no win-prob value
  function required.
- Quantities + a single recommendation under a stated objective; or a fully autonomous
  policy. Both deferred — they commit a utility function on the player's behalf.

## Decision

**The four-stat panel.** Per player `i`, under "one uniformly random face-down card of
`i` is cut this turn":

1. **P(safe wire)** — exploit / immediate progress (`P_wire`).
2. **P(bomb)** — catastrophe risk, raw probability (the risk-aware piece; no tradeoff
   baked in).
3. **1-ply ΔH(bad)** — expected post-cut entropy of `P(bad)`; immediate role-info,
   cheap, honestly myopic.
4. **Round-horizon H(bad)** — expected end-of-round entropy of `P(bad)` under
   info-greedy continuation (the min-entropy lookahead); the headline explore stat,
   capturing a cut's value as the opening of an information-gathering line.

**Dropped:** a combined "expected score (bomb→0)" — it collapses to stat 1 when the bomb
is priced like a dud, and otherwise smuggles in the bomb-vs-wire risk weight that belongs
to the human. **Dropped:** any `P(bomb)`-entropy / `EIG_bomb` stat — ephemeral (re-dealt,
ADR 0005) and perverse (minimizing it courts detonation, since the most bomb-
discriminating cut is cutting the suspected bomb hand).

**Stat 4 ships with two caveats:** it is an information *potential* (the player does not
control every cut, so info-greedy continuation is counterfactual), and the rollout
ignores bomb risk, so it must always be displayed beside stat 2.

## Consequences

- model.md §3.6 is rewritten from "two interchangeable policies" to the
  quantities-only panel plus the upgrade path.
- **Calibration becomes load-bearing**, not optional: the panel only helps if `P(bad)`
  and `P(bomb)` are calibrated (a miscalibrated risk number is worse than none). This
  promotes the deferred calibration check from ADR 0005 to a prerequisite for trusting
  the panel.
- Single-recommendation and autonomous-policy modes remain available later without
  contradicting this ADR (the panel is their substrate).

## Open — horizon-weighted VOI (synthesized, to debate later)

Recorded so the explore/exploit discussion can resume without re-deriving it.

- **The tension is real and long-horizon.** Roles are fixed, so information learned in
  round 1 keeps paying off through round 4 — its true value is the *sum of exploit
  improvements over all remaining cuts*. A **1-ply VOI lookahead is structurally blind
  to this** (it prices only the next cut), so it under-explores early. Minimizing
  role-entropy is a cheap surrogate that *does* capture "invest in durable knowledge
  now" — i.e. entropy-min is poor-man's full-horizon role-VOI, **not** a rival terminal
  objective. (Infomax alone is never a terminal goal; it is valuable only through this
  instrumental channel.)
- **The principled, λ-free form:**
  `value(i) ≈ exploit(i) + [sensitivity of a future cut's P_wire to role-certainty] ×
  [cuts remaining in the game]`. The explore weight is an *observable* (future cuts that
  benefit), not a tuning knob; it decays to 0 on the last cut, automatically yielding
  "explore early, exploit late". The only empirical quantity is the sensitivity
  coefficient, estimable by simulation.
- **Upgrade path:** stat 4 is the entropy-surrogate special case; graduate its objective
  from "end-of-round entropy" to "horizon-weighted win-prob gain" rather than rewriting.
  The user is interested in implementing this VOI lookahead; whether end-of-round entropy
  or the horizon-weighted win-prob surrogate is the right objective — and how to estimate
  the sensitivity coefficient — is the open question for the next debate.
