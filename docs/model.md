# Time Bomb Assistant — Model & Mathematics

The stable reference for the probability model shared by every variant. For the
plan and per-variant status see [roadmap.md](roadmap.md); for open work items see
[../TODO.md](../TODO.md); for orientation see the [README](../README.md).

## 1. Goal

Given only the *public* information at the table — how many wires each player
**declares**, and what each **cut** reveals — maintain, for every player, a
calibrated estimate of:

- **P(bad)** — probability the player is a bad guy,
- **P(bomb)** — probability the player holds the Bomb,
- **P(wire)** — probability that cutting one of their face-down cards reveals an
  active (Safe) wire,

and surface these as calibrated **decision inputs** for the next cut. The assistant is
quantities-only: it informs the cut, it does not dictate it (§3.5).

## 2. Game model (the rules the math depends on)

A game is played over several **rounds**. Notation used throughout the docs and
the code:

| Symbol         | Code name      | Meaning                                                             |
| -------------- | -------------- | ------------------------------------------------------------------- |
| `N`            | `num_players`  | Number of players.                                                  |
| `H`            | `hand_size`    | Cards in each player's hand **this round** (starts at 5, −1/round). |
| `A`            | `active_wires` | Safe wires still face-down across all hands.                        |
| `B`            | `num_bad`      | Number of bad guys. Known from player count (see below).            |
| `M`            | `num_bom`      | Number of bombs in play (0 or 1 in the current variants).           |
| `decls[i]`     | `declarations` | Wires player `i` *claims* to hold this round.                       |
| `wires[i]`     | `wires`        | Wires player `i` *actually* holds (hidden; sim-only).               |
| `revealed[i]`  | `revealed`     | Cards already cut from player `i`'s hand this round.                |
| `found[i]`     | `found`        | Active wires found in player `i`'s hand this round.                 |

**Bad-guy count `B` is fixed by the player count `N`** (the rules pin it, so `B` is
public information, not something to infer):

| Players `N` | Bad guys `B` |
| ----------- | ------------ |
| 4           | 1            |
| 5–7         | 2            |
| 8           | 3            |

(There is always exactly one Bomb, `M = 1`, in the full game; the `M = 0` variants are
stepping stones, not real configurations.)

**Round flow.** Each round: wires are dealt; every player declares a wire count;
then `N` cuts happen (each cut flips one face-down card of some player), updating
`revealed`/`found`. After `N` cuts the round ends and `H` decreases by 1.

**Win/lose.** Good guys win when all `A` active wires are cut. Bad guys win if the
Bomb is cut, or if time runs out (`H` reaches 1 with wires uncut).

**Behavioral assumptions (the lie model).** These define the likelihoods. A player
declares **truthfully if and only if they are good *and* bomb-free**; in every other
case they declare **uniformly at random** over `{0,…,H}`, independent of their true
wire count:

- A good guy **without** the bomb declares **truthfully**: `decls[i] = wires[i]`.
- A good guy **with** the bomb declares **uniformly at random** over `{0,…,H}`
  (holding the bomb makes even a good guy declare like a liar).
- A bad guy declares **uniformly at random** over `{0,…,H}`, **whether or not** he
  holds the bomb.

(For the `OneBadGuyNoBomb` variant only the first and third bullets apply, and the
bomb cases never arise.)

This **uniform-lie bomb model** is a deliberate simplification (firm modelling
choice; §3.3 derives the prior it implies). It still gives `P(bomb)` real signal —
a good guy forced to lie is evidence for "bad *or* holding the bomb", coupling the
two hidden variables — while avoiding the `decls ≤ wires` / `decls ≥ wires`
constraints of a *strategic* model that can produce dead-end null observations. The
richer **strategic** model (good-with-bomb under-declares, bad-with-bomb
over-declares) is the eventual target, to be A/B-tested against this once trusted; a
fully **parametric** lie bias is a far-future research item. Both are tracked in
[roadmap.md](roadmap.md) and recorded in [decisions/0004](decisions/0004-uniform-lie-bomb-model.md).

## 3. Mathematical models

The whole game (`M = 1`) couples two hidden facts, so the model is written for the
**general configuration** below; the `M = 0` and single-bad-guy variants fall out as
projections. Each section gives the general form first, then the instances.

### 3.1 State

The belief state is a probability array over **configurations**. A configuration names
the two hidden facts the assistant infers: **which players are bad** (the bad set `S`,
`|S| = B`) and, when a bomb is in play, **which player holds it this round** (`h`). For
`B` bad guys and one bomb it is a `(B+1)`-dimensional array; `b = h` is allowed (a bad
guy may be dealt his own bomb). The hardcoded variants are projections of this object:

- `OneBadGuyNoBomb` (`B=1, M=0`): a length-`N` vector `probs[i] = P(player i bad)`.
- `TwoBadGuysNoBomb` (`B=2, M=0`): a lower-triangular `N×N` matrix over bad *pairs*.
- `OneBadGuyOneBomb` (`B=1, M=1`): an `N×N` matrix `probs[b][h] = P(b bad, h holds bomb)`.
- `TwoBadGuysOneBomb` (`B=2, M=1`): an `N×N×N` tensor `probs[b1][b2][h]` (`b1 > b2`).

Every variant normalises to sum 1.

**The bomb is a card.** It occupies one of a hand's `H` slots, so the hand holding it
has only `H − 1` slots available for wires. The deal places the bomb uniformly
(`P(bomb in hand h) = 1/N`) and then the `A` wires among the remaining slots, all
independent of the secret roles.

**Persistence across rounds.** Character roles are fixed for the whole game, but the
wires *and the bomb* are reshuffled and re-dealt every round. So the **bad-set** belief
accumulates across rounds (via `CombineProbs`) while the **bomb holder** is a per-round
latent: `P(bomb)` is read from the current round's joint only and is **never** combined
across rounds (§3.5).

### 3.2 Hypergeometric likelihood

The atom shared by every computation. A hand of `n` cards contains `m` wires; `k` cards
are cut and `p` wires are found. The probability of that observation is the
hypergeometric PMF:

```
Lklhd(n, m, k, p) = C(m, p) · C(n − m, k − p) / C(n, k)
```

(`UsefulFunctions.Lklhd` writes this in the equivalent permutation form.)

**The bomb hand needs one variant of this atom.** A hand holding `w` wires, one bomb,
and `H − 1 − w` blanks, from which `k` cards are cut revealing `p` wires **and no bomb**
(the must-not-draw event — cutting the bomb would end the game), has likelihood

```
L_bomb(w, p, k) = C(w, p) · C(H − 1 − w, k − p) / C(H, k)
```

— the hypergeometric over the `H − 1` non-bomb slots, conditioned on none of the `k`
draws being the single bomb card. A bomb-free hand (all `H` slots wire/blank) is just
`Lklhd`.

### 3.3 Prior from declarations — `ProbDeclaration`

Convert the round's declarations into a prior over configurations by Bayes on the
**configuration space**, exactly parallel to `ProbCut` (§3.4). Under the **uniform-lie
model** (§2) a hand is *truthful* iff its owner is good **and** bomb-free; every truthful
hand declares its real wire count, while every liar — each bad guy, and the bomb holder
when good — declares uniformly over `{0,…,H}`, contributing a factor `(H+1)^{−1}` each, so
`(H+1)^{−|F|}` for the `|F|` free (liar) hands. This factor cancels in normalisation **only
when `|F|` is constant across configurations** — which holds for `M = 0` (`|F| = B` always)
but **not** for `M = 1`, where `|F| = B` on the self-bomb diagonal (a bad guy holds his own
bomb) and `|F| = B + 1` when a good guy holds the bomb (one extra good liar). Dropping it
there over-weights the bomb-on-good configurations by `(H+1)` and miscalibrates
`P(bomb)`/`P(bad)` (ADR 0007, verified by a generative Monte Carlo).

Marking a configuration — bad set `S`, bomb holder `h` — pins every truthful hand to its
declared count and leaves the **free hands** `F = S ∪ {h}` to hold the remaining wires:

```
t_free = A − Σ_{j ∉ F} decls[j]
```

placed uniformly among the free hands' **non-bomb** slots. Each free hand offers `H`
slots, except the bomb hand, which offers `H − 1`:

```
free_slots = Σ_{g ∈ F} (H − [g = h]) = |F|·H − 1     (and |F|·H with no bomb, M = 0)
```

Counting the consistent deals and collapsing the wire split via Vandermonde's identity
(§3.4.1) gives a single closed form:

```
P(config | decls)  ∝  (H+1)^{−|F|} · C(free_slots, t_free)  /  Π_{g ∈ F} C(H, decls[g])
```

with `C(n, k) = 0` for `k < 0` or `k > n` (zeroing any configuration that cannot account
for the wires). For `M = 0` the `(H+1)^{−|F|} = (H+1)^{−B}` is a global constant and drops
out in normalisation, so the no-bomb instances below are unchanged; for `M = 1` it must be
kept. Every variant is an instance of this one formula:

- **`B=1, M=0`** — `F = {i}`, `free_slots = H`; with `excess = Σ decls − A` and
  `t_i = decls[i] − excess`:  `P(bad=i) ∝ C(H, t_i) / C(H, decls[i])`. When `excess = 0`
  every weight is `1` → uniform, recovering "fully consistent declarations carry no
  information".
- **`B>1, M=0`** — `F = S`, `free_slots = B·H`:
  `P(S) ∝ C(B·H, Σ_{b∈S} decls[b] − excess) / Π_{b∈S} C(H, decls[b])`. Here `excess = 0`
  does **not** give a uniform prior (two lies can cancel in aggregate while some bad sets
  stay more deal-plausible), so there is **no `excess = 0` special case**.
- **`M=1`** — `F = S ∪ {h}`; the bomb hand contributes `H − 1` slots, dropping
  `free_slots` by one, and the `(H+1)^{−|F|}` lie factor is kept (it differs between the
  `h ∈ S` and `h ∉ S` cases). E.g. `B=1`: `b ≠ h` (`|F| = 2`) gives
  `(H+1)^{−2}·C(2H−1, …)/(C(H,decls[b])·C(H,decls[h]))` and the self-bomb case `b = h`
  (`|F| = 1`) gives `(H+1)^{−1}·C(H−1, …)/C(H, decls[b])` — so a good bomb-holder hypothesis
  is penalised by the extra `1/(H+1)`.

**Degeneracy.** If every configuration has zero weight (declarations impossible under the
model — unreachable with consistent game data) fall back to the uniform prior, never an
unnormalisable all-zeros vector. This matches `ProbCut`'s "return the prior on zero
marginal" convention (§3.4).

### 3.4 Bayesian update from a cut — `ProbCut`

After each reveal, update the posterior by Bayes' rule on the **configuration space**.
The hypotheses are mutually exclusive and exhaustive, so

```
posterior(c) = prior(c) · L(c) / Σ_c′ prior(c′) · L(c′)
```

Cuts reveal `{wire, blank, bomb}`; cutting the bomb ends the game, so live inference
always **conditions on "no bomb cut yet"** — every observed cut is a wire or a blank, and
that fact is itself weak evidence about where the bomb is *not*. Conditioned on a
configuration `c = (S, h)` the observation factorises into independent per-hand draws:

- **Truthful hands** `j ∉ F`: wire count pinned to `decls[j]`, no bomb — the standard
  hypergeometric `Lklhd(H, decls[j], revealed[j], found[j])` (§3.2).
- **Free hands** `F = S ∪ {h}`: marginalise their wire split (total `t_free`) under the
  uniform-placement law (§3.4.1). The bad hands use `Lklhd`; the bomb hand uses the
  must-not-draw atom `L_bomb` (§3.2).

```
L(c) = [ Π_{j ∉ F} Lklhd(H, decls[j], revealed[j], found[j]) ] · L_free(c)
```

where `L_free` sums the split of `t_free` over the free hands — each split weighted by
its uniform-placement count and normalised by `C(free_slots, t_free)`. With **no bomb**
the free hands are exactly the bad set and the split collapses to the single closed form
`L_bad` of §3.4.1; **with a bomb** the bomb hand's must-not-draw term breaks the full
collapse, but the bad guys within `S` still collapse via `L_bad`, leaving only the
bomb-vs-rest split to sum explicitly (whether that residual sum has its own closed form
is an *implementation* question — the model is fully pinned here, and the brute-force
test oracle enumerates the split regardless).

This factorisation is **exact given the uniform-lie model**: conditioning on a
configuration pins every truthful hand's wire count and constrains the free hands to
`t_free` wires placed uniformly, and given fixed per-hand counts the cut draws are
independent across hands. A *strategic* model where a good-with-bomb under-declares would
break the truthful-hand pinning — there `L(c)` would have to marginalise the good's
hidden wire count too (deferred; §3.6).

**Why the joint form is canonical.** For `OneBadGuyNoBomb` the configuration space is
"which single player is bad", so `L` reduces to a length-`N` vector. This joint form (not
a per-player binary update) is the canonical implementation because:

- It is the literal statement of Bayes' theorem on the mutually-exclusive hypotheses, so
  its correctness is self-evident and easy to test.
- A common factor `Π_all Lklhd(good)` cancels in normalisation, making it equivalent to
  the intuitive `posterior[k] ∝ prior[k] · L_bad(k)/L_good(k)`.
- It **degrades gracefully**: a configuration whose observation is impossible gets zero
  posterior; the only failure mode (every hypothesis impossible, `marginal = 0`) is
  explicitly guarded by returning the prior.
- It **generalises**: configurations become combinations of bad players (and a bomb
  holder) rather than single indices, exactly as above.

A per-player binary update instead normalises each player independently, dividing by the
evidence average `prior[i]·P(obs | i bad) + (1−prior[i])·P(obs | i good)`. That
denominator is `0` when the observation is impossible under *both* of a player's
hypotheses, producing `NaN`. (Minimal repro: `decls=[1,1,2,0]`, `H=2`, cut one of player
2's two cards.) The joint form returns the correct distribution there.

#### 3.4.1 The wire-split model (uniform placement)

For `B = 1` and no bomb the single bad guy's wire count is fully determined:
`bad_wires = A + Σfound − Σ_good decls`. As soon as more than one hand is *free* — two
bad guys, or a bad guy and a separate bomb holder — only the **group total** `t_free`
(§3.3) is determined; how it splits between the free hands is not, and the model must say
how.

**The model — uniform placement.** The `t_free` wires are placed uniformly at random
among the free hands' combined non-bomb slots (every arrangement equally likely). This is
the same assumption already used within each individual hand, and it matches a shuffled
deal: the randomness is over which *cards* are wires, not over which player is "assigned"
each wire.

Under uniform placement the per-hand wire counts follow a multivariate hypergeometric,
and because the revealed cards are an exchangeable subset, a group of bomb-free bad hands
`G` (combined `|G|·H` slots, holding `g_wires` between them) **collapses to a single
closed form** — no summation over splits:

```
L_bad(G) = [ Π_{b ∈ G} C(revealed[b], found[b]) ]
           · C(|G|·H − Σ_b revealed[b],  g_wires − Σ_b found[b])
           / C(|G|·H, g_wires)
```

This reduces **exactly** to the single-hand `Lklhd(H, bad_wires, revealed, found)` when
`|G| = 1`. The bomb hand cannot join this collapse — it offers only `H − 1` wire slots
and carries the must-not-draw constraint (§3.2, `L_bomb`) — so when a bomb is in play the
split between the bad group and the bomb hand is summed explicitly while `L_bad` still
collapses the bad group internally.

### 3.5 Outputs & decision strategies

Once the posterior is updated, the assistant reduces it to the player-facing readouts.

**Marginals and persistence.** From the configuration array,

- `P(bad = i) = Σ_{c ∋ i} probs[c]` — the **persistent** role belief (sum over every
  configuration in which `i` is bad). This marginal is what `CombineProbs` accumulates
  across rounds: roles are fixed, so the per-round marginals are conditionally
  independent evidence and multiply elementwise (exact Bayes, not a heuristic — see
  [decisions/0005](decisions/0005-cross-round-evidence-combination.md)).
- `P(bomb = h) = Σ_{c : bomb=h} probs[c]` — read from the **current round only**. Because
  the bomb is re-dealt each round (§3.1), it must **never** enter `CombineProbs`; doing so
  would treat an independent per-round draw as persistent evidence.

**P(safe wire) — `P_wire`.** For ranking cut targets: the probability that cutting one of
player `i`'s unrevealed cards yields an active wire. Mix over the belief that `i` is good
vs. bad (vs. holding the bomb), dividing remaining wires by remaining face-down cards:

```
P_wire[i] = ( P(i good)·(decls[i] − found[i]) + Σ_configs P(config)·wires_remaining[i] )
            / (H − revealed[i])
```

Two correctness requirements, both load-bearing:

- The denominator is `H − revealed[i]` (remaining face-down cards), **not** the full hand
  `H` — a card already revealed can no longer be cut. In the bomb hand it still counts the
  face-down bomb card.
- Each branch contributes only when its implied remaining-wire count is feasible, in
  `[0, H − revealed[i]]`. An impossible hypothesis (e.g. a "good" player who has already
  had more wires found than they declared) contributes nothing rather than a negative
  count.

For a truthful hand the remaining wires are the fixed `decls[i] − found[i]`; for a free
hand (bad, or the bomb holder) they are the expected remaining wires under the §3.4.1
uniform-placement split posterior, averaged over configurations.

**The four-stat panel.** The cut decision is a finite-horizon POMDP whose only true
objective is **P(good team wins)**; "score", "information", and "bomb risk" are not
separate objectives but myopic proxies for it. Solving the POMDP exactly is intractable,
and the explore/exploit tension is an artifact of that approximation. The assistant is
therefore **quantities-only**: it presents calibrated, individually-justified decision
inputs and leaves the explore/exploit/risk integration (which requires a utility function
— risk appetite — that is the player's to own) to the human. See
[ADR 0006](decisions/0006-cut-recommendation-output.md). For each player `i`, under the
hypothesis that one uniformly random face-down card of `i`'s is cut this turn, display:

1. **P(safe wire)** — expected immediate progress (exploit), from `P_wire` above.
2. **P(bomb)** — catastrophe risk, from the `P(bomb)` marginal. Surfaced as a raw
   probability; no risk tradeoff is baked in.
3. **1-ply ΔH(bad)** — expected post-cut Shannon entropy of `P(bad)`; what *this single
   cut* teaches about the fixed roles. Cheap (`O(N × outcomes)`, reuses `ProbCut`);
   honestly myopic.
4. **Round-horizon H(bad)** — expected end-of-round entropy of `P(bad)` under an
   information-greedy continuation (the `H`/`NextH`/`H_Min` min-entropy lookahead). The
   headline explore stat: it values an opening cut as the first move of an
   information-gathering *line*, which (3) cannot see.

Deliberately **excluded:** a combined "expected score (bomb→0)" number (it either
collapses to stat 1 or smuggles in the bomb-vs-wire risk weight that belongs to the
human), and any `P(bomb)`-entropy / `EIG_bomb` stat (it decays at the round boundary —
the bomb is re-dealt, §3.1 — and minimizing it perversely *courts* detonation, since the
most bomb-discriminating cut is cutting the suspected bomb hand).

Stat 4 caveats, which it must ship with: it is an information *potential* (the player does
not control every cut in a round, so info-greedy continuation is counterfactual), and the
rollout **ignores bomb risk**, so it must always be displayed beside stat 2.

(`H(·)` here is Shannon entropy — the code reuses the name `H`, which §2's table also uses
for the hand size; they are unrelated.)

### 3.6 Open modelling gaps

The declaration prior (§3.3), the wire-split (§3.4.1), the bomb sub-model (§3.2/§3.3/§3.4),
the `P_wire` marginal (§3.5), and the degeneracy convention are all **firm modelling
choices**; what remains for them is implementation and brute-force validation (see
[roadmap.md](roadmap.md) Cross-cutting foundations and [../TODO.md](../TODO.md) Axis A).
What is deliberately *not* yet modelled:

- **Strategic / parametric declarations** — the richer model where a good-with-bomb
  *under-declares* and a bad-with-bomb *over-declares* (rather than both lying uniformly,
  §2). It would break the truthful-hand pinning of §3.4 (the cut likelihood would have to
  marginalise a bomb-holding good guy's hidden wire count), and it adds a free parameter
  to fit and validate. The eventual target, to be A/B-tested against the uniform-lie
  model; a fully parametric lie bias is a further far-future refinement.
- **Risk-aware cut strategy** — folding `P(bomb)` into a single cut recommendation
  (trading expected wire progress against bomb risk). Under the quantities-only philosophy
  (§3.5) the panel stops at the raw `P(bomb)` readout and leaves the tradeoff to the human.

- **Horizon-weighted VOI (the explore/exploit "solution", open; to debate further).**
  Stat 4 of the panel uses end-of-round entropy as a *proxy* objective. The principled
  version values a cut in win-probability units and needs **no arbitrary weight λ**,
  because role information is a *durable* asset (roles are fixed; a bit learned in round 1
  improves cut-targeting in every remaining round), whereas bomb information is *ephemeral*
  (re-dealt each round). The value of an opening cut is then

  > immediate exploit(i) + [sensitivity of a future cut's `P_wire` to role-certainty] ×
  > [cuts remaining in the game]

  where the explore weight is not a tuning knob but an *observable* — how many future cuts
  will benefit from what is learned — which decays to zero on the last cut, automatically
  reproducing "explore early, exploit late". This is what makes early-round
  entropy-reduction genuinely better than myopic score-max and what a **1-ply VOI
  lookahead structurally cannot see** (it prices only the next cut's benefit, missing the
  cross-round compounding). The only empirical quantity is the sensitivity coefficient
  (estimable by simulation, not hand-tuned). Stat 4 is the entropy-surrogate special case;
  the upgrade swaps its objective from "end-of-round entropy" to "horizon-weighted win-prob
  gain". Build stat 4 first (self-contained), then graduate the objective rather than
  rewriting. See [ADR 0006](decisions/0006-cut-recommendation-output.md).

- **Resilience to model-breaking play.** The likelihoods assume the uniform-lie model
  holds exactly. Real tables violate it — a player miscounts, the declared totals are
  arithmetically impossible, a house rule shifts the deal, or a strategic liar produces
  observations the model assigns probability ~0. The degeneracy convention (§3.3/§3.4)
  keeps the belief *defined* in these cases (fall back to the prior/uniform rather than
  emit `NaN`), but a single impossible round can still permanently zero out a
  configuration under the elementwise `CombineProbs` product. Hardening this — an ε-floor
  so no round delivers an unrecoverable hard `0`, log-space accumulation against underflow,
  and a graceful response to inconsistent declarations — is a firm direction, not yet a
  fixed model; see [decisions/0005](decisions/0005-cross-round-evidence-combination.md)
  and [roadmap.md](roadmap.md).
