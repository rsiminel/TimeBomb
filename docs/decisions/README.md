# Architecture Decision Records

This folder records the **modelling and design decisions** behind the Time Bomb
assistant — one file per decision, numbered in the order they were made. Each ADR is a
small, *immutable* record of a single choice: the context that forced it, the options
on the table, the decision, and its consequences. Superseding a decision means adding a
*new* ADR that supersedes the old one — not editing history.

Why keep these: [roadmap.md](../roadmap.md) tracks *where the project is*; these ADRs
track *why it got there*, so settled questions are not silently re-litigated and the
rejected alternatives (and the conditions under which they'd be reconsidered) stay on
record. The mathematics each decision feeds into lives in [model.md](../model.md).

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-uniform-lie-declaration-prior.md) | Uniform-lie joint-Bayes declaration prior | Accepted |
| [0002](0002-degeneracy-fallback-to-prior.md) | Fall back to prior/uniform on a zero marginal | Accepted |
| [0003](0003-defer-bomb-model.md) | Defer the bomb sub-model to the `*OneBomb` variants | Accepted |

Format: lightweight [Nygard-style](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
ADRs — Context / Options considered / Decision / Consequences.
