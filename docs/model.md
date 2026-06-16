# Time Bomb Assistant — Model & Mathematics

The stable reference for the probability model shared by every variant. For the
plan and per-variant status see [roadmap.md](roadmap.md); for open work items see
[../TODO.md](../TODO.md); for orientation see the [README](../README.md).

## 1. Goal

Given only the *public* information at the table — how many wires each player
**declares**, and what each **cut** reveals — maintain, for every player, a
calibrated estimate of:

- **P(bad)** — probability the player is a Terrorist ("bad guy" / "evil"),
- **P(bomb)** — probability the player holds the Bomb,
- **P(wire)** — probability that cutting one of their face-down cards reveals an
  active (Safe) wire,

and use these to recommend **which wire to cut next**.

## 2. Game model (the rules the math depends on)

A game is played over several **rounds**. Notation used throughout the docs and
the code:

| Symbol         | Code name      | Meaning                                                             |
| -------------- | -------------- | ------------------------------------------------------------------- |
| `N`            | `num_players`  | Number of players.                                                  |
| `H`            | `hand_size`    | Cards in each player's hand **this round** (starts at 5, −1/round). |
| `A`            | `active_wires` | Safe wires still face-down across all hands.                        |
| `B`            | `num_bad`      | Number of bad guys (Terrorists). Known from player count.           |
| `M`            | `num_bom`      | Number of bombs in play (0 or 1 in the current variants).           |
| `decls[i]`     | `declarations` | Wires player `i` *claims* to hold this round.                       |
| `wires[i]`     | `wires`        | Wires player `i` *actually* holds (hidden; sim-only).               |
| `revealed[i]`  | `revealed`     | Cards already cut from player `i`'s hand this round.                |
| `found[i]`     | `found`        | Active wires found in player `i`'s hand this round.                 |

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
choice; §3.8 derives the prior it implies). It still gives `P(bomb)` real signal —
a good guy forced to lie is evidence for "bad *or* holding the bomb", coupling the
two hidden variables — while avoiding the `decls ≤ wires` / `decls ≥ wires`
constraints of a *strategic* model that can produce dead-end null observations. The
richer **strategic** model (good-with-bomb under-declares, bad-with-bomb
over-declares) is the eventual target, to be A/B-tested against this once trusted; a
fully **parametric** lie bias is a far-future research item. Both are tracked in
[roadmap.md](roadmap.md) and recorded in [decisions/0004](decisions/0004-uniform-lie-bomb-model.md).

## 3. Mathematical models

### 3.1 State

The belief state is a probability array over **configurations** — an assignment of
which players are bad and which hold bombs. For `B` bad guys and `M` bombs it is a
`(B+M)`-dimensional array indexed `probs[bad_set + bomb_set]`. The simplest variant
(`OneBadGuyNoBomb`, `B=1, M=0`) collapses this to a length-`N` vector
`probs[i] = P(player i is the bad guy)`, normalized to sum to 1. Adding one bomb
(`OneBadGuyOneBomb`, `B=1, M=1`) makes it an `N×N` matrix `probs[b][h] =
P(player b is bad and player h holds the bomb)`, with `b = h` allowed (the bad guy
may be dealt his own bomb); see §3.8.

**Persistence across rounds.** Character roles are fixed for the whole game, but the
wires *and the bomb* are reshuffled and re-dealt every round. So the **bad-set**
belief accumulates across rounds (via `CombineProbs`) while the **bomb holder** is a
per-round latent: `P(bomb)` is read from the current round's joint only and never
combined across rounds (§3.8).

### 3.2 Hypergeometric likelihood

The atom shared by every computation. A hand of `n` cards contains `m` wires; `k`
cards are cut and `p` wires are found. The probability of that observation is the
hypergeometric PMF:

```
Lklhd(n, m, k, p) = C(m, p) · C(n − m, k − p) / C(n, k)
```

(`UsefulFunctions.Lklhd` writes this in the equivalent permutation form.)

### 3.3 Prior from declarations — `ProbDeclaration`

Convert the round's declarations into a prior over configurations by Bayes on the
**configuration space**, exactly parallel to `ProbCut` (§3.4). Marking which players
are bad pins every *good* hand to its declared wire count (good guys declare
truthfully), which forces the bad side's true wire total. The prior for a
configuration is then proportional to the multivariate-hypergeometric probability of
the unique deal it implies (every assignment of the `A` wires to the `N·H` card-slots
equally likely); the bad guys' declarations contribute only a constant factor under
the **uniform-lie** model (§2 — the bad guy declares uniformly over `{0,…,H}`
independent of his true count), which cancels in normalisation.

**Single bad guy (`B = 1`).** Let `excess = Σ decls − A`. Marking player `i` as bad
forces his true count to `t_i = decls[i] − excess`, giving the closed form

```
P(bad = i | decls)  ∝  C(H, t_i) / C(H, decls[i]),     t_i = decls[i] − excess
```

with `C(H, t) = 0` for `t < 0` or `t > H` (zeroing any candidate who cannot alone
account for the excess). When `excess = 0` every weight is `1` → uniform prior,
recovering the intuition that fully consistent declarations carry no information.

**Multiple bad guys (`B > 1`).** Marking the bad set `S` pins the good hands but
leaves the group's wire total `bg_wires = Σ_{b∈S} decls[b] − excess` to be split
among the `B·H` bad card-slots by the same uniform-placement model as §3.4.1.
Summing over splits via Vandermonde's identity collapses to a closed form:

```
P(bad set S | decls)  ∝  C(B·H,  Σ_{b∈S} decls[b] − excess)  /  Π_{b∈S} C(H, decls[b])
```

which reduces exactly to the `B = 1` formula when `|S| = 1`. Note that for `B > 1`,
`excess = 0` does **not** give a uniform prior — two lies can cancel in aggregate
while some bad sets remain more deal-plausible than others — so there is **no
`excess = 0` special case**; the closed form is used unconditionally.

**Degeneracy.** If every configuration has zero weight (declarations impossible under
the model — unreachable with consistent game data) fall back to the uniform prior,
never an unnormalisable all-zeros vector. This matches `ProbCut`'s "return the prior
on zero marginal" convention (§3.4).

### 3.4 Bayesian update from a cut — `ProbCut`

After each reveal, update the posterior by Bayes' rule on the **configuration
space** (which players are bad). The hypotheses are mutually exclusive and
exhaustive. Conditioned on a configuration `c`, every hand has a known wire count
— the bad candidate holds its *implied* count `bad_wires = A + Σfound − Σdecls +
decls[i]`, every truthful good guy holds exactly `decls[good]` — so the joint
observation factorises into independent per-hand hypergeometric draws:

```
L(c) = Lklhd(H, bad_wires, revealed[bad], found[bad])
       · Π_{good ∉ c} Lklhd(H, decls[good], revealed[good], found[good])

posterior(c) = prior(c) · L(c) / Σ_c′ prior(c′) · L(c′)
```

This factorisation is **exact given truthful good guys**: conditioning on a
configuration pins every good hand's wire count to `decls[good]` and the bad hand's
to its implied count, and given fixed per-hand counts the cut draws are independent
across hands. Variants where a good guy may hold the bomb and under-declare break
that pinning — there `L(c)` must marginalise over the good's hidden wire count
(deferred; §3.7, item 6).

**Why the joint form is canonical.** For `OneBadGuyNoBomb` the configuration space
is "which single player is bad", so `L` reduces to a length-`N` vector. This joint
form (not a per-player binary update) is the canonical implementation because:

- It is the literal statement of Bayes' theorem on the mutually-exclusive
  hypotheses, so its correctness is self-evident and easy to test.
- A common factor `Π_all Lklhd(good)` cancels in normalisation, making it
  equivalent to the intuitive `posterior[k] ∝ prior[k] · L_bad(k)/L_good(k)`.
- It **degrades gracefully**: a configuration whose observation is impossible gets
  zero posterior; the only failure mode (every hypothesis impossible,
  `marginal = 0`) is explicitly guarded by returning the prior.
- It **generalises** to multiple bad guys / bombs: configurations become
  `combinations` of players rather than single indices (see §3.4.1 for the extra
  wire-split modelling that `B > 1` requires).

A per-player binary update divides by `prior[i]·peh + (1−prior[i])·penh`, which is
`0` when an observation is impossible under *both* of a player's hypotheses,
producing `NaN`. (Minimal repro: `decls=[1,1,2,0]`, `H=2`, cut one of player 2's
two cards.) The joint form returns the correct distribution there.

### 3.4.1 Multiple bad guys: the wire-split model (B > 1)

For `B = 1` the single bad guy's wire count is fully determined:
`bad_wires = A + Σfound − Σ_good decls`. For `B > 1` only the bad *group*'s **total**

```
bg_wires = A + Σfound − Σ_good decls[good]
```

is determined; how it splits between the `B` bad hands is not, and the model must
say how.

**The model — uniform placement.** The `bg_wires` wires are placed uniformly at
random among the bad group's `B·H` combined card-slots (every arrangement equally
likely). This is the same assumption already used within each individual hand, and
it matches a shuffled deal: the randomness is over which *cards* are wires, not over
which bad guy is "assigned" each wire.

Under uniform placement the per-hand wire counts follow a multivariate
hypergeometric, and because the revealed cards are an exchangeable subset, the bad
group's contribution to the cut-likelihood **collapses to a single closed form** —
no summation over splits:

```
L_bad(bad set) = [ Π_{b ∈ bad set} C(revealed[b], found[b]) ]
                 · C(B·H − Σ_b revealed[b],  bg_wires − Σ_b found[b])
                 / C(B·H, bg_wires)
```

This reduces **exactly** to the single-hand `Lklhd(H, bad_wires, revealed, found)`
when `B = 1`, so it is the honest generalisation of §3.4. The full configuration
likelihood multiplies `L_bad` by the good guys' independent per-hand
hypergeometrics, and Bayes normalisation proceeds as in §3.4.

> **The current code uses a worse model — to be replaced.**
> `TwoBadGuysNoBomb.ProbCut` (and `General.py`) instead weight a split `k` of
> `bg_wires` between two bad hands by `C(bg_wires, k)`, normalised by
> `2^bg_wires` — a **Binomial(bg_wires, ½)** split, i.e. each wire flips an
> independent fair coin. That treats wires as capacity-free distinguishable objects
> and is only a large-`H` approximation to the correct hypergeometric split.
> Concretely, for `bg_wires = 2, H = 2` it gives split probabilities `(¼, ½, ¼)`
> instead of the correct `(⅙, ⅔, ⅙)`, over-crediting one bad guy hoarding the
> wires. Replace it with the closed form above (TODO.md, TwoBadGuysNoBomb Step 1).

### 3.5 Expected wire probability — `P_wire`

For ranking cut targets: the probability that cutting one of player `i`'s
unrevealed cards yields an active wire. Mix over the belief that `i` is good vs.
bad, dividing remaining wires by remaining face-down cards:

```
P_wire[i] = ( P(i good)·(decls[i] − found[i]) + P(i bad)·bad_wires_remaining[i] )
            / (H − revealed[i])
```

Two correctness requirements, both load-bearing:

- The denominator is `H − revealed[i]` (remaining face-down cards), **not** the
  full hand `H` — a card already revealed can no longer be cut.
- Each branch contributes only when its implied remaining-wire count is feasible,
  in `[0, H − revealed[i]]`. An impossible hypothesis (e.g. a "good" player who has
  already had more wires found than they declared) contributes nothing rather than
  a negative count.

For `B = 1`, `bad_wires_remaining[i]` is the fixed implied count minus `found[i]`.
For `B > 1` it is **not** a single number per hand: it is the marginal expected
remaining wires in bad hand `i` under the uniform-placement (multivariate-
hypergeometric) model of §3.4.1, averaged over configurations. `P_wire` must use
that marginal, not a per-hand fixed count. (The model is fixed by §3.4.1; computing
this marginal is an implementation task — TODO.md Axis A2.)

### 3.6 Decision strategies

The cut decision is a finite-horizon POMDP whose only true objective is **P(good team
wins)**; "score", "information", and "bomb risk" are not separate objectives but myopic
proxies for it. Solving the POMDP exactly is intractable, and the explore/exploit
tension is an artifact of that approximation. The assistant is therefore **quantities-
only**: it presents calibrated, individually-justified decision inputs and leaves the
explore/exploit/risk integration (which requires a utility function — risk appetite —
that is the player's to own) to the human. See [ADR 0006](decisions/0006-cut-recommendation-output.md).

**The four-stat panel.** For each player `i`, under the hypothesis that one uniformly
random face-down card of `i`'s is cut this turn, display:

1. **P(safe wire)** — expected immediate progress (exploit), from `P_wire`.
2. **P(bomb)** — catastrophe risk, from the `P(bomb)` column. Surfaced as a raw
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
the bomb is re-dealt, §3.8.3 — and minimizing it perversely *courts* detonation, since
the most bomb-discriminating cut is cutting the suspected bomb hand).

Stat 4 caveats, which it must ship with: it is an information *potential* (the player
does not control every cut in a round, so info-greedy continuation is counterfactual),
and the rollout **ignores bomb risk**, so it must always be displayed beside stat 2.

(`H(·)` here is Shannon entropy — the code reuses the name `H`, which §2's table also
uses for the hand size; they are unrelated.)

**Upgrade path — horizon-weighted VOI (open; to debate further).** Stat 4 uses
end-of-round entropy as a *proxy* objective. The principled version values a cut in
win-probability units and needs **no arbitrary weight λ**, because role information is a
*durable* asset (roles are fixed; a bit learned in round 1 improves cut-targeting in
every remaining round), whereas bomb information is *ephemeral* (re-dealt each round).
The value of an opening cut is then

> immediate exploit(i) + [sensitivity of a future cut's `P_wire` to role-certainty] ×
> [cuts remaining in the game]

where the explore weight is not a tuning knob but an *observable* — how many future cuts
will benefit from what is learned — which decays to zero on the last cut, automatically
reproducing "explore early, exploit late." This is what makes early-round
entropy-reduction genuinely better than myopic score-max and what a **1-ply VOI
lookahead structurally cannot see** (it prices only the next cut's benefit, missing the
cross-round compounding). The only empirical quantity is the sensitivity coefficient
(estimable by simulation, not hand-tuned). Stat 4 is the entropy-surrogate special case;
the upgrade swaps its objective from "end-of-round entropy" to "horizon-weighted win-prob
gain". Build stat 4 first (self-contained), then graduate the objective rather than
rewriting — this is the explore/exploit "solution" still under discussion.

### 3.7 Open modelling gaps

The declaration prior (§3.3), the `B > 1` wire-split (§3.4.1), the `B > 1` `P_wire`
marginal (§3.5), the degeneracy convention, and now the **bomb sub-model** (§3.8) are
all **firm modelling choices**; what remains for them is implementation and
brute-force validation (see [roadmap.md](roadmap.md) Cross-cutting foundations and
[../TODO.md](../TODO.md) Axis A). What is deliberately *not* yet modelled:

- **Strategic bomb declarations** — the richer model where a good-with-bomb
  *under-declares* and a bad-with-bomb *over-declares* (rather than both lying
  uniformly, §3.8). The eventual target, to be A/B-tested against the uniform-lie
  bomb model; a fully parametric lie bias is a further far-future refinement.
- **Risk-aware cut strategy** — folding `P(bomb)` into the cut recommendation
  (trading expected wire progress against bomb risk, §3.6). Deferred; the bomb
  sub-model stops at the `P(bomb)` readout.

### 3.8 The bomb (`B = 1, M = 1`)

Adding one bomb introduces a second hidden variable: besides which player is bad,
*which player holds the bomb this round*. A configuration is the pair `(b, h)` —
player `b` is bad, player `h` holds the bomb — and the belief state is the `N×N`
matrix `probs[b][h]`, with `b = h` allowed (§3.1). The bomb is a card like a wire: it
occupies one of a hand's `H` slots, so a hand holding it has only `H − 1` slots
available for wires. The deal places the bomb uniformly (`P(bomb in hand h) = 1/N`)
and then the `A` wires among the remaining slots, all independent of the secret
roles.

Under the **uniform-lie bomb model** (§2) a hand is *truthful* iff its owner is good
and bomb-free, i.e. every hand except `b` and `h`. The liars (`b`, and `h` when
distinct) declared uniformly, contributing only a constant factor.

#### 3.8.1 Declaration prior — `ProbDeclaration`

Marking `(b, h)` pins every truthful hand `j ∉ {b, h}` to its declared wires and
forces the liar hands' free wire total to `t_free = A − Σ_{j∉{b,h}} decls[j]`, which
must be placed in their **non-bomb** slots. Counting the consistent deals and
collapsing the split via Vandermonde's identity gives the closed form (with
`excess = Σ decls − A`):

```
b ≠ h:   P(b, h | decls)  ∝  C(2H−1, decls[b]+decls[h] − excess) / ( C(H, decls[b]) · C(H, decls[h]) )
b = h:   P(b, b | decls)  ∝  C(H−1,  decls[b] − excess)          /   C(H, decls[b])
```

with `C(n, k) = 0` for `k < 0` or `k > n`. The slot counts are the crux: two distinct
liar hands offer `2H − 1` non-bomb slots for the free wires (the bomb eats one), and
the self-bomb case `b = h` offers `H − 1`. This is the §3.3 prior generalised by one
bomb-occupied slot. If every configuration has zero weight, fall back to the uniform
`N×N` prior (degeneracy convention, §3.3).

#### 3.8.2 Cut likelihood — `ProbCut`

Cuts now reveal `{wire, blank, bomb}`. Cutting the bomb ends the game (bad guys win),
so live inference always **conditions on "no bomb cut yet"**: every observed cut is a
wire or a blank, and that fact is itself weak evidence about where the bomb is *not*.
Conditioned on `(b, h)` the likelihood factorises:

- **Truthful hands** `j ∉ {b, h}`: wires pinned to `decls[j]`, no bomb — the standard
  hypergeometric `Lklhd(H, decls[j], revealed[j], found[j])` of §3.2.
- **Liar / bomb hands** `b, h`: marginalise their wire split (total `t_free`) under
  the same uniform-placement law as §3.4.1. In the **bomb hand** `h` the bomb is a
  *must-not-draw* card, so its cut term is

  ```
  P(found_h wires, 0 bombs in revealed_h draws | w_h wires) =
        C(w_h, found_h) · C(H − 1 − w_h, revealed_h − found_h) / C(H, revealed_h)
  ```

  (a hypergeometric over `w_h` wires, one bomb, and `H − 1 − w_h` blanks, gated to
  draw no bomb). The bad hand `b` (when `b ≠ h`) uses the ordinary `Lklhd`.

The posterior is Bayes on the configuration space, `posterior(c) ∝ prior(c) · L(c)`,
returning the prior on a zero marginal (§3.4). Whether the liar/bomb marginal
collapses to a single closed form (as the bomb-free §3.4.1 does) is an
*implementation* question — the **model** is fully pinned here, and the brute-force
test oracle enumerates the split regardless.

#### 3.8.3 Readouts and persistence

- `P(bad = b) = Σ_h probs[b][h]` — the persistent role belief; this marginal is what
  `CombineProbs` accumulates across rounds.
- `P(bomb = h) = Σ_b probs[b][h]` — read from the **current round only**. Because the
  bomb is re-dealt each round (§3.1), it must **never** enter `CombineProbs`; doing so
  would treat an independent per-round draw as persistent evidence.
